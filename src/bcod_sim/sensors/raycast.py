"""Deterministic analytic rays against Phase 4 sphere/box entities and bottom planes."""

import math

import torch

from bcod_sim.config.models import BoxShape
from bcod_sim.core.errors import ExternalDataCoverageError
from bcod_sim.world.entities import EntitySnapshot
from bcod_sim.world.world import ParametricWorld

Vec3 = tuple[float, float, float]


def _sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: float) -> float | None:
    delta = tuple(o - c for o, c in zip(origin, center))
    b = sum(a * d for a, d in zip(delta, direction))
    c = sum(a * a for a in delta) - radius * radius
    discriminant = b * b - c
    if discriminant < 0:
        return None
    root = math.sqrt(discriminant)
    near, far = -b - root, -b + root
    return near if near >= 0 else far if far >= 0 else None


def _box(origin: Vec3, direction: Vec3, center: Vec3, half_extents: Vec3) -> float | None:
    near, far = -math.inf, math.inf
    for o, d, c, h in zip(origin, direction, center, half_extents):
        low, high = c - h, c + h
        if abs(d) < 1e-14:
            if o < low or o > high:
                return None
            continue
        a, b = (low - o) / d, (high - o) / d
        near, far = max(near, min(a, b)), min(far, max(a, b))
        if near > far:
            return None
    return near if near >= 0 else far if far >= 0 else None


def entity_distance(origin: Vec3, direction: Vec3, entity: EntitySnapshot) -> float | None:
    if isinstance(entity.shape, BoxShape):
        from bcod_sim.frames.tensor import rotate_world_to_body
        q = torch.tensor(entity.orientation_q_to_ned, dtype=torch.float64)
        local_origin = rotate_world_to_body(torch.tensor(origin, dtype=torch.float64) -
                                            torch.tensor(entity.position_ned_m, dtype=torch.float64), q)
        local_direction = rotate_world_to_body(torch.tensor(direction, dtype=torch.float64), q)
        return _box(tuple(local_origin.tolist()), tuple(local_direction.tolist()),
                    (0.0, 0.0, 0.0), entity.shape.half_extents_m)
    return _sphere(origin, direction, entity.position_ned_m, entity.shape.radius_m)


def bottom_distance(origin: Vec3, direction: Vec3, world: ParametricWorld) -> float | None:
    bathymetry = getattr(world, "bathymetry", None)
    if bathymetry is None:
        return None
    # Bracket and bisect z(ray)-bottom(x,y) through the canonical surface API.
    def residual(distance: float) -> float:
        n = origin[0] + distance*direction[0]
        e = origin[1] + distance*direction[1]
        z = origin[2] + distance*direction[2]
        return z - bathymetry.depth_at(n, e)
    previous_distance, previous = 0.0, residual(0.0)
    if previous >= 0:
        return 0.0
    distance = 0.25
    for _ in range(80):
        try:
            value = residual(distance)
        except ExternalDataCoverageError:
            return None
        if value >= 0:
            low, high = previous_distance, distance
            for _ in range(50):
                middle = (low+high)/2
                if residual(middle) >= 0: high = middle
                else: low = middle
            return (low+high)/2
        previous_distance, previous = distance, value
        distance *= 1.6
    return None


def environment_distance(origin: Vec3, direction: Vec3, world: ParametricWorld, *, sim_time_s: float,
                         include_entities: bool = True, include_bathymetry: bool = True):
    candidates: list[tuple[float, str]] = []
    if include_entities:
        for entity in world.entities(sim_time_s=sim_time_s, env_id=world.env_id):
            distance = entity_distance(origin, direction, entity)
            if distance is not None: candidates.append((distance, entity.id))
    if include_bathymetry:
        distance = bottom_distance(origin, direction, world)
        if distance is not None: candidates.append((distance, "bathymetry"))
    return min(candidates, default=(None, None), key=lambda row: float("inf") if row[0] is None else row[0])
