"""Downward fan sonar against canonical NED bathymetry."""

import math

import torch

from bcod_sim.core.errors import ExternalDataUnavailableError, PhysicalValidationError
from bcod_sim.frames.tensor import rotate_body_to_world
from bcod_sim.sensors.base import SensorConfig, SensorContext, SensorPacket, packet, stable_noise
from bcod_sim.sensors.kinematics import mount_kinematics
from bcod_sim.sensors.raycast import bottom_distance


class Sonar:
    kind = "physical"

    def __init__(self, config: SensorConfig, *, min_range_m: float, max_range_m: float,
                 fov_rad: float, beam_count: int) -> None:
        if config.sensor_type != "sonar" or not (0 <= min_range_m < max_range_m and math.isfinite(max_range_m)):
            raise PhysicalValidationError("Invalid sonar type or range")
        if not math.isfinite(fov_rad) or not (0 <= fov_rad < math.pi) or beam_count < 1:
            raise PhysicalValidationError("Invalid sonar FOV or beam count")
        if (beam_count == 1 and fov_rad != 0) or (beam_count > 1 and fov_rad <= 0):
            raise PhysicalValidationError("Sonar FOV and resolution disagree")
        self.config, self.min_range_m, self.max_range_m = config, min_range_m, max_range_m
        self.fov_rad, self.beam_count = fov_rad, beam_count

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket:
        if context.world.bathymetry is None:
            raise ExternalDataUnavailableError("Sonar requires bathymetry")
        mount = mount_kinematics(self.config, context)
        origin = tuple(mount.position_ned_m.tolist())
        ranges, hit = [], []
        for index in range(self.beam_count):
            angle = -self.fov_rad / 2 + (self.fov_rad * index / (self.beam_count - 1) if self.beam_count > 1 else 0)
            local = mount.position_ned_m.new_tensor((math.sin(angle), 0.0, math.cos(angle)))
            direction = tuple(rotate_body_to_world(local, mount.q_mount_to_ned).tolist())
            distance = bottom_distance(origin, direction, context.world)
            valid = distance is not None and self.min_range_m <= distance <= self.max_range_m
            ranges.append(distance if valid else self.max_range_m)
            hit.append(valid)
        ranges_tensor = mount.position_ned_m.new_tensor(ranges)
        mask = torch.tensor(hit, dtype=torch.bool, device=ranges_tensor.device)
        noise = stable_noise(self.config, sample_step, (self.beam_count,), dtype=ranges_tensor.dtype,
                             device=ranges_tensor.device, stream="ranges")
        noisy = ranges_tensor + noise
        mask = mask & (noisy >= self.min_range_m) & (noisy <= self.max_range_m)
        ranges_tensor = torch.where(mask, noisy, torch.full_like(noisy, self.max_range_m))
        return packet(self.config, self.kind, sample_step, {"range_m": ranges_tensor, "hit": mask},
                      {"range_m": "m", "hit": "bool"}, {"range_m": "mount", "hit": "mount"})
