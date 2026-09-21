"""Deterministic analytic rays against Phase 4 sphere/box entities and bottom planes."""

import math

from bcod_sim.config.models import BoxShape, FlatBathymetry, SlopedBathymetry
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
        return _box(origin, direction, entity.position_ned_m, entity.shape.half_extents_m)
    return _sphere(origin, direction, entity.position_ned_m, entity.shape.radius_m)


def bottom_distance(origin: Vec3, direction: Vec3, world: ParametricWorld) -> float | None:
    bathymetry = world.spec.bathymetry
    if bathymetry is None:
        return None
    if isinstance(bathymetry, FlatBathymetry):
        numerator, denominator = bathymetry.bottom_ned_z_m - origin[2], direction[2]
    else:
        base = (bathymetry.bottom_at_origin_ned_z_m +
                bathymetry.north_slope * (origin[0] - bathymetry.origin_ned_m[0]) +
                bathymetry.east_slope * (origin[1] - bathymetry.origin_ned_m[1]))
        numerator = base - origin[2]
        denominator = direction[2] - bathymetry.north_slope * direction[0] - bathymetry.east_slope * direction[1]
    if abs(denominator) < 1e-14:
        return None
    distance = numerator / denominator
    return distance if distance >= 0 else None
