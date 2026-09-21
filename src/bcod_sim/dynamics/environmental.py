"""Provider-independent relative-air and simplified wave-kinematic loads."""
from dataclasses import dataclass
import math
import torch
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.dynamics.restoring import WrenchResult
from bcod_sim.frames.tensor import rotate_body_to_world,rotate_world_to_body
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.world.world import WorldSample

@dataclass(frozen=True)
class WindCoefficientPoint:
    angle_rad:float; cx:float; cy:float; ck:float; cn:float

@dataclass(frozen=True)
class RelativeWindLoads:
    frontal_area_m2:float
    lateral_area_m2:float
    reference_height_m:float
    coefficients:tuple[WindCoefficientPoint,...]
    air_density_kg_m3:float=1.225
    model_name:str="relative_wind_coefficients"
    def __post_init__(self):
        if self.frontal_area_m2<=0 or self.lateral_area_m2<=0 or not self.coefficients: raise PhysicalValidationError("Invalid wind model")
        if any(b.angle_rad<=a.angle_rad for a,b in zip(self.coefficients,self.coefficients[1:])): raise PhysicalValidationError("Wind coefficient angles must increase")
    def _coefficient(self,angle:float)->WindCoefficientPoint:
        wrapped=(angle+math.pi)%(2*math.pi)-math.pi; points=self.coefficients
        if wrapped<points[0].angle_rad or wrapped>points[-1].angle_rad: raise PhysicalValidationError("Apparent wind angle outside coefficient validity")
        for a,b in zip(points,points[1:]):
            if wrapped<=b.angle_rad:
                f=(wrapped-a.angle_rad)/(b.angle_rad-a.angle_rad)
                return WindCoefficientPoint(wrapped,*(getattr(a,k)+f*(getattr(b,k)-getattr(a,k)) for k in ("cx","cy","ck","cn")))
        return points[-1]
    def evaluate(self,state:VesselState,sample:WorldSample)->WrenchResult:
        vessel_world=rotate_body_to_world(state.nu_body[:3],state.q_body_to_ned); apparent=rotate_world_to_body(sample.wind_ned_mps[0]-vessel_world,state.q_body_to_ned)
        speed=torch.linalg.vector_norm(apparent[:2]); angle=math.atan2(float(apparent[1]),float(apparent[0])) if speed else 0.; c=self._coefficient(angle)
        q=.5*self.air_density_kg_m3*float(speed)**2
        tau=state.nu_body.new_tensor((q*self.frontal_area_m2*c.cx,q*self.lateral_area_m2*c.cy,0.,q*self.lateral_area_m2*self.reference_height_m*c.ck,0.,q*self.lateral_area_m2*self.reference_height_m*c.cn))
        return WrenchResult(tau,self.model_name,{"apparent_wind_body_mps":apparent,"apparent_speed_mps":float(speed),"apparent_angle_rad":angle})

@dataclass(frozen=True)
class KinematicWaveLoads:
    linear_drag:tuple[float,float,float]
    quadratic_drag:tuple[float,float,float]
    inertia_coefficients:tuple[float,float,float]
    model_name:str="kinematic_wave_drag"
    def evaluate(self,state:VesselState,sample:WorldSample)->WrenchResult:
        orbital=rotate_world_to_body(sample.wave_orbital_ned_mps[0],state.q_body_to_ned); acceleration=rotate_world_to_body(sample.wave_acceleration_ned_mps2[0],state.q_body_to_ned)
        relative=state.nu_body[:3]-orbital; linear=state.nu_body.new_tensor(self.linear_drag); quadratic=state.nu_body.new_tensor(self.quadratic_drag); inertia=state.nu_body.new_tensor(self.inertia_coefficients)
        force=-linear*relative-quadratic*relative.abs()*relative+inertia*acceleration
        tau=torch.cat((force,state.nu_body.new_zeros(3)))
        return WrenchResult(tau,self.model_name,{"wave_orbital_body_mps":orbital,"wave_acceleration_body_mps2":acceleration})
