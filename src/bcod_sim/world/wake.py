"""Double-buffered deterministic shared wake baseline; no calibration claim."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError
from bcod_sim.frames.tensor import rotate_body_to_world
from bcod_sim.state.vessel_state import VesselState


@dataclass(frozen=True)
class WakeParameters:
    strength_per_surge: float
    lateral_radius_m: float
    downstream_decay_m: float
    source: str
    version: str

    def __post_init__(self) -> None:
        if (not all(math.isfinite(x) and x > 0 for x in
                    (self.strength_per_surge, self.lateral_radius_m, self.downstream_decay_m)) or
            not self.source or not self.version):
            raise PhysicalValidationError("Wake parameters require positive scales and provenance")


@dataclass(frozen=True)
class WakeEmission:
    env_id: int
    source_vessel_id: int
    origin_ned_m: tuple[float, float, float]
    forward_ned: tuple[float, float, float]
    surge_mps: float
    parameters: WakeParameters
    weight: float = 1.0

    def __post_init__(self) -> None:
        if (self.env_id < 0 or self.source_vessel_id < 0 or
            len(self.origin_ned_m) != 3 or len(self.forward_ned) != 3 or
            not all(math.isfinite(x) for x in (*self.origin_ned_m, *self.forward_ned, self.surge_mps)) or
            self.surge_mps < 0 or not math.isfinite(self.weight) or self.weight <= 0 or
            abs(math.sqrt(sum(x*x for x in self.forward_ned)) - 1) > 1e-8):
            raise PhysicalValidationError("Invalid wake emission")


class GaussianWakeEmitter:
    def __init__(self, env_id: int, source_vessel_id: int, parameters: WakeParameters) -> None:
        if env_id < 0 or source_vessel_id < 0:
            raise PhysicalValidationError("Wake emitter requires nonnegative owner IDs")
        self.env_id, self.source_vessel_id, self.parameters = env_id, source_vessel_id, parameters

    def emit(self, state: VesselState, *, weight: float = 1.0) -> WakeEmission:
        forward = rotate_body_to_world(state.position_ned.new_tensor((1, 0, 0)), state.q_body_to_ned)
        forward = torch.stack((forward[0], forward[1], forward.new_zeros(())))
        length = torch.linalg.vector_norm(forward).item()
        if length < 1e-12:
            raise PhysicalValidationError("Wake emitter heading has no horizontal projection")
        return WakeEmission(self.env_id, self.source_vessel_id, tuple(state.position_ned.tolist()),
                            tuple((forward / length).tolist()), max(0.0, state.nu_body[0].item()), self.parameters,
                            weight)


class WakeField:
    def __init__(self) -> None:
        self._current: tuple[WakeEmission, ...] = ()
        self._next: dict[tuple[int, int, int], WakeEmission] | None = None
        self.generation = 0

    def begin_step(self) -> None:
        if self._next is not None:
            raise PhysicalValidationError("Wake field step already open")
        self._next = {}

    def emit(self, emission: WakeEmission, *, substep_index: int = 0) -> None:
        if self._next is None:
            raise PhysicalValidationError("Wake emission requires open step")
        if substep_index < 0:
            raise PhysicalValidationError("Wake substep index must be nonnegative")
        key = (emission.env_id, emission.source_vessel_id, substep_index)
        if key in self._next:
            raise DuplicateIdentityError(f"Duplicate wake emitter: {key}")
        self._next[key] = emission

    def sample(self, positions_ned_m: torch.Tensor, *, env_id: int,
               receiver_vessel_id: int | None = None) -> torch.Tensor:
        if (env_id < 0 or positions_ned_m.ndim != 2 or positions_ned_m.shape[1] != 3 or
            not positions_ned_m.is_floating_point() or not torch.isfinite(positions_ned_m).all().item()):
            raise PhysicalValidationError("Wake query requires finite floating NED [N,3] positions")
        rows = []
        for position in positions_ned_m.tolist():
            values = [[], [], []]
            for emission in self._current:
                if emission.env_id != env_id or emission.source_vessel_id == receiver_vessel_id:
                    continue
                offset = tuple(p-o for p, o in zip(position, emission.origin_ned_m))
                along = sum(x*y for x, y in zip(offset, emission.forward_ned))
                downstream = -along
                if downstream < 0 or emission.surge_mps == 0:
                    continue
                cross = tuple(offset[i] - along * emission.forward_ned[i] for i in range(3))
                lateral_sq = sum(x*x for x in cross)
                params = emission.parameters
                magnitude = (emission.weight * params.strength_per_surge * emission.surge_mps *
                             math.exp(-downstream / params.downstream_decay_m) *
                             math.exp(-0.5 * lateral_sq / params.lateral_radius_m**2))
                for i in range(3):
                    values[i].append(-magnitude * emission.forward_ned[i])
            rows.append(tuple(math.fsum(component) for component in values))
        result = positions_ned_m.new_tensor(rows).reshape((-1, 3))
        if not torch.isfinite(result).all().item():
            raise PhysicalValidationError("Wake field produced nonfinite values")
        return result

    def swap(self) -> None:
        if self._next is None:
            raise PhysicalValidationError("Wake swap requires open step")
        self._current = tuple(value for _, value in sorted(self._next.items()))
        self._next = None
        self.generation += 1
