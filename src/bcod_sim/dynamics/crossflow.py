"""Pure strip-theory cross-flow damping models in body FRD."""
from dataclasses import dataclass
import math
from typing import Protocol
import torch
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.dynamics.restoring import WrenchResult
from bcod_sim.state.vessel_state import VesselState

class CrossflowModel(Protocol):
    def validate(self,*,dtype:torch.dtype,device:torch.device)->None: ...
    def evaluate(self,state:VesselState,water_velocity_body:torch.Tensor|None=None)->WrenchResult: ...

@dataclass(frozen=True)
class NoCrossflow:
    model_name:str="none"
    def validate(self,*,dtype:torch.dtype,device:torch.device)->None: return None
    def evaluate(self,state:VesselState,water_velocity_body:torch.Tensor|None=None)->WrenchResult:
        return WrenchResult(state.nu_body.new_zeros(6),self.model_name,{"relative_crossflow_velocity":state.nu_body.new_zeros(2)})

HOERNER_RATIO=(.0108623,.176606,.353025,.451863,.472838,.492877,.493252,.558473,.646401,.833589,.988002,1.30807,1.63918,1.85998,2.31288,2.59998,3.00877,3.45075,3.7379,4.00309)
HOERNER_CD=(1.96608,1.96573,1.89756,1.78718,1.58374,1.27862,1.21082,1.08356,.998631,.87959,.828415,.759941,.691442,.657076,.630693,.596186,.586846,.585909,.559877,.559315)

def hoerner_coefficient(beam_m:float,draft_m:float)->float:
    if not(math.isfinite(beam_m) and math.isfinite(draft_m) and beam_m>0 and draft_m>0):
        raise PhysicalValidationError("Hoerner geometry must be positive and finite")
    ratio=beam_m/(2*draft_m)
    if ratio<HOERNER_RATIO[0]: raise PhysicalValidationError("Hoerner ratio below reference range")
    if ratio>=HOERNER_RATIO[-1]: return HOERNER_CD[-1]
    for i in range(1,len(HOERNER_RATIO)):
        if ratio<=HOERNER_RATIO[i]:
            x0,x1=HOERNER_RATIO[i-1],HOERNER_RATIO[i]; y0,y1=HOERNER_CD[i-1],HOERNER_CD[i]
            return y0+(ratio-x0)*(y1-y0)/(x1-x0)
    raise AssertionError

@dataclass(frozen=True)
class StripTheoryCrossflow:
    station_x_body_m:torch.Tensor
    station_beam_m:torch.Tensor
    station_draft_m:torch.Tensor
    water_density_kg_m3:float=1025.0
    drag_model:str="hoerner"
    include_vertical:bool=False
    model_name:str="strip_theory"

    @classmethod
    def constant_section(cls,length_m:float,beam_m:float,draft_m:float,strips:int,*,water_density_kg_m3:float=1025.0,include_vertical:bool=False,dtype:torch.dtype=torch.float64,device:torch.device|None=None):
        if not isinstance(strips,int) or strips<2 or not all(math.isfinite(x) and x>0 for x in (length_m,beam_m,draft_m)):
            raise PhysicalValidationError("Cross-flow requires positive geometry and >=2 strips")
        dx=length_m/strips
        x=torch.linspace(-length_m/2+dx/2,length_m/2-dx/2,strips,dtype=dtype,device=device)
        return cls(x,torch.full_like(x,beam_m),torch.full_like(x,draft_m),water_density_kg_m3,"hoerner",include_vertical)

    def validate(self,*,dtype:torch.dtype,device:torch.device)->None:
        vals=(self.station_x_body_m,self.station_beam_m,self.station_draft_m)
        if (len(self.station_x_body_m)<2 or any(v.ndim!=1 or v.shape!=self.station_x_body_m.shape for v in vals) or
            any(v.dtype!=dtype or v.device!=device or not torch.isfinite(v).all().item() for v in vals) or
            not torch.all(self.station_x_body_m[1:]>self.station_x_body_m[:-1]).item() or
            (self.station_beam_m<=0).any().item() or (self.station_draft_m<=0).any().item() or
            not math.isfinite(self.water_density_kg_m3) or self.water_density_kg_m3<=0 or self.drag_model!="hoerner"):
            raise PhysicalValidationError("Invalid strip-theory cross-flow definition")

    def evaluate(self,state:VesselState,water_velocity_body:torch.Tensor|None=None)->WrenchResult:
        water=state.nu_body.new_zeros(3) if water_velocity_body is None else water_velocity_body
        if water.shape!=(3,) or not torch.isfinite(water).all().item(): raise PhysicalValidationError("Invalid body water velocity")
        relative=state.nu_body[:3]-water; x=self.station_x_body_m
        edges=torch.empty(len(x)+1,dtype=x.dtype,device=x.device); edges[1:-1]=(x[:-1]+x[1:])/2
        edges[0]=x[0]-(x[1]-x[0])/2; edges[-1]=x[-1]+(x[-1]-x[-2])/2; dx=edges[1:]-edges[:-1]
        cd=state.nu_body.new_tensor([hoerner_coefficient(float(b),float(t)) for b,t in zip(self.station_beam_m,self.station_draft_m)])
        local_y=relative[1]+x*state.nu_body[5]
        strip_y=-.5*self.water_density_kg_m3*self.station_draft_m*cd*local_y.abs()*local_y*dx
        tau=state.nu_body.new_zeros(6); tau[1]=strip_y.sum(); tau[5]=(x*strip_y).sum()
        local_z=relative[2]+x*state.nu_body[4]
        if self.include_vertical:
            strip_z=-.5*self.water_density_kg_m3*self.station_draft_m*cd*local_z.abs()*local_z*dx
            tau[2]=strip_z.sum(); tau[4]=(x*strip_z).sum()
        return WrenchResult(tau,self.model_name,{"relative_crossflow_velocity":torch.stack((relative[1],state.nu_body[5])),"station_local_sway_mps":local_y,"drag_coefficient":cd,"include_vertical":self.include_vertical})

@dataclass(frozen=True)
class SectionalCrossflow:
    """Per-component strips; repeated X stations preserve multihull geometry."""
    station_x_body_m: torch.Tensor
    station_y_body_m: torch.Tensor
    station_beam_m: torch.Tensor
    station_draft_m: torch.Tensor
    station_dx_m: torch.Tensor
    water_density_kg_m3: float = 1025.
    cd_scale: float = 1.
    model_name: str = "sectional_stations"
    station_cd: torch.Tensor | None = None
    station_lift_base_kg_per_m: torch.Tensor | None = None
    moment_reference_x_m: float = 0.
    incidence_blend: bool = False
    shear_blend: bool = False
    translation_shear_v5: bool = False

    @classmethod
    def from_stations(cls, stations: list[dict], *, density: float = 1025.,
                      cd_scale: float = 1., dtype: torch.dtype = torch.float64):
        values = lambda key: torch.tensor([s[key] for s in stations], dtype=dtype)
        cd = values("cd") if all("cd" in s for s in stations) else None
        lift = values("lift_base_kg_per_m") if all("lift_base_kg_per_m" in s for s in stations) else None
        methods = {s.get("lateral_force_method") for s in stations}
        blended = methods in ({"incidence_blend_v3"}, {"shear_incidence_blend_v4"})
        shear = methods == {"shear_incidence_blend_v4"}
        translation_shear = methods == {"translation_shear_v5"}
        return cls(values("x_m"), values("y_m"), values("beam_m"),
                   values("draft_m"), values("dx_m"), density, cd_scale,
                   station_cd=cd, station_lift_base_kg_per_m=lift,
                   incidence_blend=blended, shear_blend=shear,
                   translation_shear_v5=translation_shear)

    def validate(self, *, dtype: torch.dtype, device: torch.device) -> None:
        arrays = (self.station_x_body_m, self.station_y_body_m, self.station_beam_m,
                  self.station_draft_m, self.station_dx_m)
        if (len(arrays[0]) < 3 or any(x.ndim != 1 or x.shape != arrays[0].shape or
            x.dtype != dtype or x.device != device or not torch.isfinite(x).all().item() for x in arrays)
            or any((x <= 0).any().item() for x in arrays[2:])
            or not math.isfinite(self.water_density_kg_m3) or self.water_density_kg_m3 <= 0
            or not math.isfinite(self.cd_scale) or self.cd_scale <= 0
            or not math.isfinite(self.moment_reference_x_m)):
            raise PhysicalValidationError("Invalid sectional cross-flow stations")
        for name, array in (("station_cd", self.station_cd),
                            ("station_lift_base_kg_per_m", self.station_lift_base_kg_per_m)):
            if array is not None and (array.shape != arrays[0].shape or array.dtype != dtype or
                array.device != device or not torch.isfinite(array).all().item() or
                (array < 0).any().item()):
                raise PhysicalValidationError(f"Invalid {name}")

    def evaluate(self, state: VesselState, water_velocity_body: torch.Tensor | None = None) -> WrenchResult:
        water = state.nu_body.new_zeros(3) if water_velocity_body is None else water_velocity_body
        if water.shape != (3,) or not torch.isfinite(water).all().item():
            raise PhysicalValidationError("Invalid water velocity")
        nu = state.nu_body
        local = nu[1] - water[1] + self.station_x_body_m * nu[5]
        ratio = self.station_beam_m / (2 * self.station_draft_m)
        cd = (torch.clamp(1.5 / torch.sqrt(torch.clamp(ratio, min=.05)), min=.55, max=2.6)
              if self.station_cd is None else self.station_cd) * self.cd_scale
        drag = -.5 * self.water_density_kg_m3 * cd * self.station_draft_m * local.abs() * local * self.station_dx_m
        local_u = nu[0] - water[0] - self.station_y_body_m * nu[5]
        lift = (torch.zeros_like(drag) if self.station_lift_base_kg_per_m is None else
                -self.station_lift_base_kg_per_m * local_u.abs() * local)
        if self.translation_shear_v5:
            projected = self.station_draft_m * self.station_dx_m
            mean_local = torch.sum(projected * local) / projected.sum()
            mean_u = torch.sum(projected * local_u.abs()) / projected.sum()
            speed2 = mean_u.square() + mean_local.square()
            translation_weight = torch.where(speed2 > 0,
                mean_local.square() / torch.clamp(speed2, min=1e-30),
                torch.zeros_like(speed2))
            shear = local - mean_local
            drag = -.5 * self.water_density_kg_m3 * cd * self.station_draft_m * local * (
                translation_weight * local.abs() + (1-translation_weight) * shear.abs()) * self.station_dx_m
            lift = (1-translation_weight) * lift
            weight = torch.ones_like(local) * translation_weight
        elif self.incidence_blend:
            speed2 = local_u.square() + local.square()
            incidence = torch.where(speed2 > 0, local.square() / torch.clamp(speed2, min=1e-30),
                                    torch.zeros_like(speed2))
            if self.shear_blend:
                projected = self.station_draft_m * self.station_dx_m
                mean_local = torch.sum(projected * local) / projected.sum()
                shear = local - mean_local
                shear2 = shear.square()
                shear_weight = torch.where(shear2 + mean_local.square() > 0,
                    shear2 / torch.clamp(shear2 + mean_local.square(), min=1e-30),
                    torch.zeros_like(shear2))
                weight = 1 - (1-incidence) * (1-shear_weight)
            else:
                weight = incidence
            drag, lift = weight * drag, (1-weight) * lift
        else:
            weight = torch.zeros_like(local)
        fy = drag + lift
        tau = nu.new_zeros(6)
        tau[1] = fy.sum()
        tau[5] = ((self.station_x_body_m-self.moment_reference_x_m) * fy).sum()
        return WrenchResult(tau, self.model_name, {"station_local_sway_mps": local,
                             "station_local_surge_mps": local_u,
                             "station_local_incidence_rad": torch.atan2(local, local_u.abs()),
                             "station_crossflow_weight": weight,
                             "drag_coefficient": cd, "component_offsets_y_m": self.station_y_body_m,
                             "section_x_m": self.station_x_body_m,
                             "section_drag_force_n": drag, "section_lift_force_n": lift,
                             "section_force_n": fy,
                             "section_yaw_moment_nm": (self.station_x_body_m-self.moment_reference_x_m)*fy,
                             "moment_reference_x_m": self.moment_reference_x_m})
