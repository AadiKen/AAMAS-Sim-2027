"""Canonical NED/FRD/SI transforms; quaternions are scalar-first."""

import math
from typing import Literal

from bcod_sim.core.errors import FrameConversionError

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Wrench = tuple[Vec3, Vec3]


def _vec3(v: Vec3) -> Vec3:
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise FrameConversionError("Expected finite 3-vector")
    return tuple(float(x) for x in v)  # type: ignore[return-value]


def ned_to_enu(v: Vec3) -> Vec3:
    n, e, d = _vec3(v)
    return e, n, -d


def enu_to_ned(v: Vec3) -> Vec3:
    e, n, u = _vec3(v)
    return n, e, -u


def frd_to_flu(v: Vec3) -> Vec3:
    f, r, d = _vec3(v)
    return f, -r, -d


def flu_to_frd(v: Vec3) -> Vec3:
    f, l, u = _vec3(v)
    return f, -l, -u


def normalize_quaternion(q: Quat) -> Quat:
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise FrameConversionError("Expected finite scalar-first quaternion")
    magnitude = math.sqrt(sum(x * x for x in q))
    if magnitude <= 1e-15:
        raise FrameConversionError("Zero quaternion")
    return tuple(float(x / magnitude) for x in q)  # type: ignore[return-value]


def quaternion_conjugate(q: Quat) -> Quat:
    w, x, y, z = normalize_quaternion(q)
    return w, -x, -y, -z


def rpy_to_quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Quat:
    if not all(math.isfinite(x) for x in (roll_rad, pitch_rad, yaw_rad)):
        raise FrameConversionError("Euler angles must be finite radians")
    cr, sr = math.cos(roll_rad / 2), math.sin(roll_rad / 2)
    cp, sp = math.cos(pitch_rad / 2), math.sin(pitch_rad / 2)
    cy, sy = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
    return normalize_quaternion((cr * cp * cy + sr * sp * sy,
                                 sr * cp * cy - cr * sp * sy,
                                 cr * sp * cy + sr * cp * sy,
                                 cr * cp * sy - sr * sp * cy))


def body_to_world(v: Vec3, orientation: Quat) -> Vec3:
    x, y, z = _vec3(v)
    w, qx, qy, qz = normalize_quaternion(orientation)
    # Rodrigues form of q * (0,v) * conjugate(q).
    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)
    return (x + w * tx + qy * tz - qz * ty,
            y + w * ty + qz * tx - qx * tz,
            z + w * tz + qx * ty - qy * tx)


def world_to_body(v: Vec3, orientation: Quat) -> Vec3:
    return body_to_world(v, quaternion_conjugate(orientation))


def rotate_wrench(wrench: Wrench, orientation: Quat, *, to_world: bool = True) -> Wrench:
    transform = body_to_world if to_world else world_to_body
    return transform(wrench[0], orientation), transform(wrench[1], orientation)


def moment_from_force(offset_frd_m: Vec3, force_frd_n: Vec3) -> Vec3:
    x, y, z = _vec3(offset_frd_m)
    fx, fy, fz = _vec3(force_frd_n)
    return y * fz - z * fy, z * fx - x * fz, x * fy - y * fx


def bathymetry_to_ned_z(value_m: float, *, convention: Literal["elevation_positive_up", "depth_positive_down"], datum: str) -> float:
    if not datum or not math.isfinite(value_m):
        raise FrameConversionError("Explicit vertical datum and finite value required")
    if convention == "elevation_positive_up":
        return -value_m
    if convention == "depth_positive_down":
        return value_m
    raise FrameConversionError(f"Unknown vertical convention: {convention}")


def ned_to_web(v: Vec3) -> Vec3:
    """Display axes: X east, Y up, Z north."""
    n, e, d = _vec3(v)
    return e, -d, n


def web_to_ned(v: Vec3) -> Vec3:
    e, u, n = _vec3(v)
    return n, e, -u
