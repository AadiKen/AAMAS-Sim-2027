"""Geometry-bearing world entities; collision response belongs to Phase 6."""

from dataclasses import dataclass
import math

from bcod_sim.config.models import BoxShape, ScriptedEntity, SphereShape, StaticEntity
from bcod_sim.core.errors import PhysicalValidationError


@dataclass(frozen=True)
class EntitySnapshot:
    id: str
    position_ned_m: tuple[float, float, float]
    shape: SphereShape | BoxShape
    collision_enabled: bool
    scripted: bool
    velocity_ned_mps: tuple[float, float, float]


def snapshot_static(entity: StaticEntity) -> EntitySnapshot:
    return EntitySnapshot(entity.id, entity.position_ned_m, entity.shape, entity.collision_enabled,
                          False, (0.0, 0.0, 0.0))


def snapshot_scripted(entity: ScriptedEntity, sim_time_s: float) -> EntitySnapshot | None:
    if sim_time_s < entity.start_time_s:
        return None
    elapsed = sim_time_s - entity.start_time_s
    position = tuple(p + elapsed * velocity for p, velocity in zip(entity.position_ned_m, entity.velocity_ned_mps))
    if not all(math.isfinite(value) for value in position):
        raise PhysicalValidationError(f"Scripted entity {entity.id} produced nonfinite position")
    return EntitySnapshot(entity.id, position, entity.shape, entity.collision_enabled,
                          True, entity.velocity_ned_mps)
