"""Hard operating limits; checked before and after each integration step."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.core.errors import OperatingEnvelopeError, PhysicalValidationError
from bcod_sim.state.vessel_state import VesselState


@dataclass(frozen=True)
class OperatingEnvelope:
    max_abs_nu: torch.Tensor  # [6], positive finite
    min_substep_s: float = 0.0
    max_substep_s: float = math.inf

    def validate(self, *, dtype: torch.dtype, device: torch.device) -> None:
        if (self.max_abs_nu.shape != (6,) or self.max_abs_nu.dtype != dtype or self.max_abs_nu.device != device or
            not torch.isfinite(self.max_abs_nu).all().item() or (self.max_abs_nu <= 0).any().item() or
            not math.isfinite(self.min_substep_s) or self.min_substep_s < 0 or
            self.max_substep_s <= 0 or self.max_substep_s < self.min_substep_s):
            raise PhysicalValidationError("Invalid operating envelope")

    def check(self, state: VesselState, dt_s: float) -> None:
        if dt_s < self.min_substep_s or dt_s > self.max_substep_s:
            raise OperatingEnvelopeError("Dynamics substep outside declared envelope")
        if (state.nu_body.abs() > self.max_abs_nu).any().item():
            raise OperatingEnvelopeError("Vessel velocity exceeds hard operating envelope")
