"""Pure, configuration-driven hydrostatic wrench models in body FRD."""

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol

import torch

from bcod_sim.core.errors import OperatingEnvelopeError, PhysicalValidationError
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.dynamics.reference_points import transform_hydrostatic_stiffness


@dataclass(frozen=True)
class WrenchResult:
    tau_body: torch.Tensor
    model_name: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    validity: Mapping[str, object] = field(default_factory=dict)


class HydrostaticsModel(Protocol):
    def validate(self, *, dtype: torch.dtype, device: torch.device) -> None: ...
    def evaluate(self, state: VesselState, mass_kg: float, cg_frd_m: torch.Tensor) -> WrenchResult: ...


def rotate_world_to_body(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    w = q[0]
    xyz = -q[1:]
    t = 2 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)


def quaternion_to_rpy(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind()
    return torch.stack((torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                        torch.asin(torch.clamp(2*(w*y-z*x), -1.0, 1.0)),
                        torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def reference_point_transform(stiffness: torch.Tensor, reference_point_frd_m: torch.Tensor) -> torch.Tensor:
    """Transform source G to the plant origin with MSS/Fossen H(r)' G H(r)."""
    return transform_hydrostatic_stiffness(stiffness,reference_point_frd_m)


@dataclass(frozen=True)
class Hydrostatics:
    """Legacy gravity/constant-buoyancy model, retained without behavior change."""
    buoyancy_n: float
    buoyancy_center_frd_m: torch.Tensor
    gravity_mps2: float = 9.80665
    model_name: str = "constant_buoyancy"

    def validate(self, *, dtype: torch.dtype, device: torch.device) -> None:
        if (self.buoyancy_center_frd_m.shape != (3,) or self.buoyancy_center_frd_m.dtype != dtype or
            self.buoyancy_center_frd_m.device != device or not torch.isfinite(self.buoyancy_center_frd_m).all().item() or
            not math.isfinite(self.buoyancy_n) or self.buoyancy_n < 0 or
            not math.isfinite(self.gravity_mps2) or self.gravity_mps2 <= 0):
            raise PhysicalValidationError("Invalid hydrostatics")

    def evaluate(self, state: VesselState, mass_kg: float, cg_frd_m: torch.Tensor) -> WrenchResult:
        down = state.nu_body.new_tensor((0.0,0.0,1.0)); down_body = rotate_world_to_body(down,state.q_body_to_ned)
        weight = mass_kg*self.gravity_mps2*down_body; buoyancy = -self.buoyancy_n*down_body
        moment = torch.linalg.cross(cg_frd_m,weight)+torch.linalg.cross(self.buoyancy_center_frd_m,buoyancy)
        return WrenchResult(torch.cat((weight+buoyancy,moment)),self.model_name,
                            {"submerged_volume_m3":None,"center_of_buoyancy_frd_m":self.buoyancy_center_frd_m})

    def wrench(self, q_body_to_ned: torch.Tensor, mass_kg: float, cg_frd_m: torch.Tensor) -> torch.Tensor:
        state=VesselState(q_body_to_ned.new_zeros(3),q_body_to_ned,q_body_to_ned.new_zeros(6))
        return self.evaluate(state,mass_kg,cg_frd_m).tau_body


@dataclass(frozen=True)
class LinearHydrostatics:
    stiffness_6x6: torch.Tensor
    equilibrium_position_ned_m: torch.Tensor
    equilibrium_rpy_rad: torch.Tensor
    max_abs_roll_rad: float
    max_abs_pitch_rad: float
    allow_unstable: bool = False
    source_stiffness_6x6: torch.Tensor | None = None
    source_reference_point_frd_m: torch.Tensor | None = None
    model_name: str = "linear_matrix"

    def validate(self, *, dtype: torch.dtype, device: torch.device) -> None:
        tensors=(self.stiffness_6x6,self.equilibrium_position_ned_m,self.equilibrium_rpy_rad)
        if (self.stiffness_6x6.shape!=(6,6) or self.equilibrium_position_ned_m.shape!=(3,) or
            self.equilibrium_rpy_rad.shape!=(3,) or any(t.dtype!=dtype or t.device!=device for t in tensors) or
            not all(torch.isfinite(t).all().item() for t in tensors)):
            raise PhysicalValidationError("Linear hydrostatics tensors must be finite and match plant schema")
        if not torch.allclose(self.stiffness_6x6,self.stiffness_6x6.T,rtol=1e-9,atol=1e-10):
            raise PhysicalValidationError("Hydrostatic stiffness must be symmetric")
        if not all(math.isfinite(x) and x>0 for x in (self.max_abs_roll_rad,self.max_abs_pitch_rad)):
            raise PhysicalValidationError("Linear hydrostatic attitude validity limits must be positive")
        eig=torch.linalg.eigvalsh(self.stiffness_6x6); scale=max(1.0,torch.linalg.matrix_norm(self.stiffness_6x6).item())
        if eig[0].item() < -1e-10*scale and not self.allow_unstable:
            raise PhysicalValidationError("Hydrostatic stiffness has a materially negative eigenvalue")
        if (self.source_stiffness_6x6 is None)!=(self.source_reference_point_frd_m is None):
            raise PhysicalValidationError("Source stiffness and reference point provenance must be paired")

    @property
    def eigenvalues(self) -> tuple[float,...]:
        return tuple(float(x) for x in torch.linalg.eigvalsh(self.stiffness_6x6))

    def displacement(self,state:VesselState)->torch.Tensor:
        angular=wrap_angle(quaternion_to_rpy(state.q_body_to_ned)-self.equilibrium_rpy_rad)
        if abs(angular[0].item())>self.max_abs_roll_rad or abs(angular[1].item())>self.max_abs_pitch_rad:
            raise OperatingEnvelopeError("Linear hydrostatic attitude exceeds configured validity")
        return torch.cat((state.position_ned-self.equilibrium_position_ned_m,angular))

    def evaluate(self,state:VesselState,mass_kg:float,cg_frd_m:torch.Tensor)->WrenchResult:
        delta=self.displacement(state); tau=-(self.stiffness_6x6@delta)
        return WrenchResult(tau,self.model_name,{"delta_eta":delta,"stiffness_eigenvalues":self.eigenvalues,
                                                 "submerged_volume_m3":None,"center_of_buoyancy_frd_m":None})
