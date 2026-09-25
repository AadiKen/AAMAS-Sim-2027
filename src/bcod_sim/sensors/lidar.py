"""Mounted planar LiDAR with enforced fan, range, beam count, and no-return mask."""

import math

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.frames.tensor import rotate_body_to_world
from bcod_sim.sensors.base import SensorConfig, SensorContext, SensorPacket, packet, stable_noise
from bcod_sim.sensors.kinematics import mount_kinematics


class LiDAR:
    kind = "physical"
    requirements = frozenset({"geometry.raycast", "weather.visibility"})

    def __init__(self, config: SensorConfig, *, min_range_m: float, max_range_m: float,
                 fov_rad: float, ray_count: int) -> None:
        if config.sensor_type != "lidar" or not (0 <= min_range_m < max_range_m and math.isfinite(max_range_m)):
            raise PhysicalValidationError("Invalid LiDAR type or range")
        if not (0 <= fov_rad <= 2 * math.pi) or not math.isfinite(fov_rad) or ray_count < 1:
            raise PhysicalValidationError("Invalid LiDAR FOV or ray count")
        if (ray_count == 1 and fov_rad != 0) or (ray_count > 1 and fov_rad <= 0):
            raise PhysicalValidationError("LiDAR FOV and resolution disagree")
        self.config, self.min_range_m, self.max_range_m = config, min_range_m, max_range_m
        self.fov_rad, self.ray_count = fov_rad, ray_count

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket:
        mount = mount_kinematics(self.config, context)
        origin = tuple(mount.position_ned_m.tolist())
        environment = context.environment
        environment.require(self.requirements)
        visibility = environment.sample("weather.visibility", mount.position_ned_m,
                                        context.sim_time_s).value[0].item()
        effective_range = min(self.max_range_m, visibility)
        ranges, hit = [], []
        for index in range(self.ray_count):
            angle = -self.fov_rad / 2 + (self.fov_rad * index / (self.ray_count - 1) if self.ray_count > 1 else 0)
            local = mount.position_ned_m.new_tensor((math.cos(angle), math.sin(angle), 0.0))
            direction = tuple(rotate_body_to_world(local, mount.q_mount_to_ned).tolist())
            result = environment.sample("geometry.raycast", mount.position_ned_m,
                context.sim_time_s, direction_ned=mount.position_ned_m.new_tensor(direction),
                max_range_m=effective_range, include_bathymetry=False).value
            nearest = result["distance_m"] if result["hit"] else math.inf
            valid = self.min_range_m <= nearest <= effective_range
            ranges.append(nearest if valid else self.max_range_m)
            hit.append(valid)
        ranges_tensor = mount.position_ned_m.new_tensor(ranges)
        mask = torch.tensor(hit, dtype=torch.bool, device=ranges_tensor.device)
        noise = stable_noise(self.config, sample_step, (self.ray_count,), dtype=ranges_tensor.dtype,
                             device=ranges_tensor.device, stream="ranges")
        noisy = ranges_tensor + noise
        mask = mask & (noisy >= self.min_range_m) & (noisy <= self.max_range_m)
        ranges_tensor = torch.where(mask, noisy, torch.full_like(noisy, self.max_range_m))
        return packet(self.config, self.kind, sample_step, {"range_m": ranges_tensor, "hit": mask},
                      {"range_m": "m", "hit": "bool"}, {"range_m": "mount", "hit": "mount"},
                      sample_time_s=context.sim_time_s)
