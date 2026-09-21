"""Explicit world-field-to-wrench contracts for the authoritative engine."""

from dataclasses import dataclass
import math
from typing import Mapping, Protocol

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.frames.tensor import rotate_body_to_world, rotate_world_to_body
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.world.world import WorldSample


ENVIRONMENT_TERMS = ("current", "wind", "wave", "wake")


class EnvironmentLoadModel(Protocol):
    def evaluate(self, state: VesselState, sample: WorldSample) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True)
class ExplicitZeroLoads:
    """Opt-in analytic fixture: world fields have no mechanical coupling."""

    def evaluate(self, state: VesselState, sample: WorldSample) -> Mapping[str, torch.Tensor]:
        if sample.current_valid is not None and not bool(sample.current_valid[0]):
            raise PhysicalValidationError("Current query lies outside the physical water column")
        return {name: state.nu_body.new_zeros((6,)) for name in ENVIRONMENT_TERMS}


@dataclass(frozen=True)
class LinearEnvironmentLoads:
    current_force_n_per_mps: float
    wind_force_n_per_mps: float
    wave_heave_n_per_m: float
    wake_force_n_per_mps: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) and x >= 0 for x in (
            self.current_force_n_per_mps, self.wind_force_n_per_mps,
            self.wave_heave_n_per_m, self.wake_force_n_per_mps)):
            raise PhysicalValidationError("Environment load coefficients must be finite and nonnegative")

    def evaluate(self, state: VesselState, sample: WorldSample) -> Mapping[str, torch.Tensor]:
        velocity_world = rotate_body_to_world(state.nu_body[:3], state.q_body_to_ned)
        current_body = rotate_world_to_body(sample.current_ned_mps[0] - velocity_world, state.q_body_to_ned)
        wind_body = rotate_world_to_body(sample.wind_ned_mps[0] - velocity_world, state.q_body_to_ned)
        wake_body = rotate_world_to_body(sample.wake_ned_mps[0], state.q_body_to_ned)
        zero = state.nu_body.new_zeros((3,))
        return {
            "current": torch.cat((self.current_force_n_per_mps * current_body, zero)),
            "wind": torch.cat((self.wind_force_n_per_mps * wind_body, zero)),
            "wave": torch.cat((torch.stack((zero[0], zero[1], self.wave_heave_n_per_m *
                                           (sample.wave_surface_ned_z_m[0] - state.position_ned[2]))), zero)),
            "wake": torch.cat((self.wake_force_n_per_mps * wake_body, zero)),
        }
