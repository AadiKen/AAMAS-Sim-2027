"""Flattened global vessel table; no max-vessels padding."""

from dataclasses import dataclass

import torch

from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError
from bcod_sim.state.vessel_state import VesselState


@dataclass(frozen=True)
class VesselTable:
    env_id: torch.Tensor
    vessel_id: torch.Tensor
    position_ned: torch.Tensor
    q_body_to_ned: torch.Tensor
    nu_body: torch.Tensor
    rl_active: torch.Tensor
    physical_active: torch.Tensor
    actuator_state_ref: tuple[str | None, ...]

    def __post_init__(self) -> None:
        n = self.env_id.numel()
        if self.env_id.shape != (n,) or self.vessel_id.shape != (n,):
            raise PhysicalValidationError("Identity vectors must be flat")
        if self.env_id.dtype != torch.int64 or self.vessel_id.dtype != torch.int64:
            raise PhysicalValidationError("Identity vectors must be int64")
        if (self.position_ned.shape, self.q_body_to_ned.shape, self.nu_body.shape) != ((n, 3), (n, 4), (n, 6)):
            raise PhysicalValidationError("Canonical vessel table has invalid state shapes")
        if self.rl_active.shape != (n,) or self.physical_active.shape != (n,) or len(self.actuator_state_ref) != n:
            raise PhysicalValidationError("Vessel masks/ownership must match row count")
        if self.rl_active.dtype != torch.bool or self.physical_active.dtype != torch.bool:
            raise PhysicalValidationError("Activity masks must be bool")
        pairs = list(zip(self.env_id.tolist(), self.vessel_id.tolist()))
        if len(pairs) != len(set(pairs)):
            raise DuplicateIdentityError("Duplicate (env_id, vessel_id) in vessel table")
        for i in range(n):
            VesselState(self.position_ned[i], self.q_body_to_ned[i], self.nu_body[i])
