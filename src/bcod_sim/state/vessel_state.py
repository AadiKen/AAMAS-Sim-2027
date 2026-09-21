"""Single vessel state values; all coordinates NED/FRD and SI."""

from dataclasses import dataclass

import torch

from bcod_sim.core.errors import NonFiniteStateError, PhysicalValidationError


@dataclass(frozen=True)
class VesselState:
    position_ned: torch.Tensor  # [3]
    q_body_to_ned: torch.Tensor  # [w,x,y,z]
    nu_body: torch.Tensor  # [u,v,w,p,q,r]

    def __post_init__(self) -> None:
        for name, value, count in (("position_ned", self.position_ned, 3),
                                   ("q_body_to_ned", self.q_body_to_ned, 4),
                                   ("nu_body", self.nu_body, 6)):
            if value.shape != (count,) or not value.is_floating_point():
                raise PhysicalValidationError(f"{name} requires floating tensor [{count}]")
            if not torch.isfinite(value).all().item():
                raise NonFiniteStateError(f"{name} is nonfinite")
        if any(x.device != self.position_ned.device or x.dtype != self.position_ned.dtype
               for x in (self.q_body_to_ned, self.nu_body)):
            raise PhysicalValidationError("State tensors must share device and dtype")
        norm = torch.linalg.vector_norm(self.q_body_to_ned).item()
        if abs(norm - 1.0) > 1e-6:
            raise PhysicalValidationError("Orientation must be a normalized body-FRD-to-world-NED quaternion")

    def clone(self) -> "VesselState":
        return VesselState(self.position_ned.clone(), self.q_body_to_ned.clone(), self.nu_body.clone())
