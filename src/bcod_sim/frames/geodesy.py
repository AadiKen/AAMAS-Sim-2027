"""WGS84 geodetic coordinates and local NED via ECEF."""

import math

from bcod_sim.core.errors import FrameConversionError
from bcod_sim.frames.transforms import Vec3, _vec3

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def geodetic_to_ecef(lat_rad: float, lon_rad: float, altitude_m: float) -> Vec3:
    if not all(map(math.isfinite, (lat_rad, lon_rad, altitude_m))) or abs(lat_rad) > math.pi / 2:
        raise FrameConversionError("Invalid geodetic coordinate")
    sin_lat = math.sin(lat_rad)
    radius = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    return ((radius + altitude_m) * math.cos(lat_rad) * math.cos(lon_rad),
            (radius + altitude_m) * math.cos(lat_rad) * math.sin(lon_rad),
            (radius * (1 - WGS84_E2) + altitude_m) * sin_lat)


def ecef_to_geodetic(ecef_m: Vec3) -> Vec3:
    x, y, z = _vec3(ecef_m)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-10 and abs(z) < 1e-10:
        raise FrameConversionError("ECEF origin has no geodetic coordinate")
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(12):
        radius = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
        altitude = p / max(abs(math.cos(lat)), 1e-15) - radius if p > 1e-10 else abs(z) - radius * (1 - WGS84_E2)
        next_lat = math.atan2(z, p * (1 - WGS84_E2 * radius / (radius + altitude))) if p > 1e-10 else math.copysign(math.pi / 2, z)
        if abs(next_lat - lat) < 1e-14:
            lat = next_lat
            break
        lat = next_lat
    radius = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    altitude = p / math.cos(lat) - radius if p > 1e-10 else abs(z) - radius * (1 - WGS84_E2)
    return lat, lon, altitude


def geodetic_to_ned(lat_rad: float, lon_rad: float, altitude_m: float, origin: Vec3) -> Vec3:
    lat0, lon0, alt0 = _vec3(origin)
    x, y, z = geodetic_to_ecef(lat_rad, lon_rad, altitude_m)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    sl, cl, so, co = math.sin(lat0), math.cos(lat0), math.sin(lon0), math.cos(lon0)
    return (-sl * co * dx - sl * so * dy + cl * dz,
            -so * dx + co * dy,
            -cl * co * dx - cl * so * dy - sl * dz)


def ned_to_geodetic(ned_m: Vec3, origin: Vec3) -> Vec3:
    n, e, d = _vec3(ned_m)
    lat0, lon0, alt0 = _vec3(origin)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    sl, cl, so, co = math.sin(lat0), math.cos(lat0), math.sin(lon0), math.cos(lon0)
    return ecef_to_geodetic((x0 - sl * co * n - so * e - cl * co * d,
                             y0 - sl * so * n + co * e - cl * so * d,
                             z0 + cl * n - sl * d))
