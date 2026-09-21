"""Mounted WGS84 position and NED velocity sensor."""

import math

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.frames.geodesy import geodetic_to_ecef, ned_to_geodetic
from bcod_sim.sensors.base import SensorConfig, SensorContext, SensorPacket, packet, stable_noise
from bcod_sim.sensors.kinematics import mount_kinematics


class GPS:
    kind = "physical"

    def __init__(self, config: SensorConfig, *, origin_wgs84_rad_m: tuple[float, float, float]) -> None:
        if config.sensor_type != "gps" or len(origin_wgs84_rad_m) != 3 or not all(math.isfinite(x) for x in origin_wgs84_rad_m):
            raise PhysicalValidationError("GPS requires type gps and explicit WGS84 origin")
        geodetic_to_ecef(*origin_wgs84_rad_m)
        self.config = config
        self.origin = origin_wgs84_rad_m

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket:
        mount = mount_kinematics(self.config, context)
        position = mount.position_ned_m + stable_noise(self.config, sample_step, (3,),
                          dtype=mount.position_ned_m.dtype, device=mount.position_ned_m.device, stream="position")
        lat, lon, altitude = ned_to_geodetic(tuple(position.tolist()), self.origin)
        velocity = mount.velocity_ned_mps + stable_noise(self.config, sample_step, (3,),
                          dtype=position.dtype, device=position.device, stream="velocity")
        return packet(self.config, self.kind, sample_step,
                      {"wgs84_lat_lon_alt": position.new_tensor((lat, lon, altitude)), "velocity_ned_mps": velocity},
                      {"wgs84_lat_lon_alt": "rad,rad,m", "velocity_ned_mps": "m/s"},
                      {"wgs84_lat_lon_alt": "WGS84", "velocity_ned_mps": "NED"})
