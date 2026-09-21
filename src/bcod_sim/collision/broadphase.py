"""Deterministic sweep-and-prune candidates per environment."""

from dataclasses import dataclass
import math
from typing import Any

import torch

from bcod_sim.collision.shapes import CollisionShape, bounding_radius
from bcod_sim.collision.shapes import SeabedSurface, TriangleMesh
from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.frames.transforms import normalize_quaternion


@dataclass(frozen=True)
class CollisionBody:
    env_id: int
    id: str
    shape: CollisionShape
    state: VesselState | None = None
    plant: Plant6 | None = None
    static_position_ned_m: tuple[float, float, float] | None = None
    static_q_to_ned: tuple[float, float, float, float] = (1, 0, 0, 0)
    kinematic_velocity_ned_mps: tuple[float, float, float] = (0, 0, 0)
    contact_material: Any | None = None

    def __post_init__(self) -> None:
        if self.env_id < 0 or not self.id:
            raise PhysicalValidationError("Collision body requires environment and stable ID")
        if (self.state is None) != (self.plant is None):
            raise PhysicalValidationError("Dynamic collision body requires both state and plant")
        if self.state is None and (self.static_position_ned_m is None or
                                   len(self.static_position_ned_m) != 3 or
                                   not all(math.isfinite(x) for x in self.static_position_ned_m)):
            raise PhysicalValidationError("Static collision body requires finite NED position")
        if len(self.kinematic_velocity_ned_mps) != 3 or not all(math.isfinite(x) for x in self.kinematic_velocity_ned_mps):
            raise PhysicalValidationError("Kinematic velocity must be finite NED vector")
        if self.state is not None and self.state.nu_body.dtype != self.plant.total_mass.dtype:
            raise PhysicalValidationError("Collision state and plant dtype must agree")
        if self.state is not None and self.state.nu_body.device != self.plant.total_mass.device:
            raise PhysicalValidationError("Collision state and plant device must agree")
        if self.state is not None and isinstance(self.shape, (TriangleMesh, SeabedSurface)):
            raise PhysicalValidationError("Terrain collision body must be static")
        q = normalize_quaternion(self.static_q_to_ned)
        if any(abs(a-b) > 1e-8 for a, b in zip(q, self.static_q_to_ned)):
            raise PhysicalValidationError("Static collision orientation must be normalized")

    @property
    def dynamic(self) -> bool:
        return self.state is not None

    def position(self) -> torch.Tensor:
        if self.state is not None:
            return self.state.position_ned
        return torch.tensor(self.static_position_ned_m, dtype=torch.float64)

    def orientation(self) -> torch.Tensor:
        if self.state is not None:
            return self.state.q_body_to_ned
        return torch.tensor(self.static_q_to_ned, dtype=torch.float64)


def broadphase_pairs(bodies: tuple[CollisionBody, ...]) -> tuple[tuple[CollisionBody, CollisionBody], ...]:
    ids = [(body.env_id, body.id) for body in bodies]
    if len(ids) != len(set(ids)):
        raise DuplicateIdentityError("Duplicate collision body identity")
    entries = []
    for body in bodies:
        radius = bounding_radius(body.shape)
        pos = body.position()
        entries.append((pos[0].item() - radius, pos[0].item() + radius, body.env_id, body.id, body,
                        pos.tolist(), radius))
    entries.sort(key=lambda row: (row[0], row[2], row[3]))
    active = []
    pairs = []
    for low, high, env, body_id, body, position, radius in entries:
        active = [entry for entry in active if entry[1] >= low]
        for other in active:
            _, _, other_env, other_id, other_body, other_position, other_radius = other
            if env != other_env or (not body.dynamic and not other_body.dynamic):
                continue
            if sum((a-b)**2 for a, b in zip(position, other_position)) > (radius + other_radius)**2:
                continue
            a, b = sorted((body, other_body), key=lambda item: item.id)
            pairs.append((a, b))
        active.append((low, high, env, body_id, body, position, radius))
    return tuple(sorted(pairs, key=lambda pair: (pair[0].env_id, pair[0].id, pair[1].id)))
