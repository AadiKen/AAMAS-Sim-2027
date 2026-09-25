"""Range/FOV-constrained entity observations derived from visible world state."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.frames.tensor import rotate_world_to_body
from bcod_sim.sensors.base import SensorConfig, SensorContext, SensorPacket, packet, stable_noise
from bcod_sim.sensors.kinematics import mount_kinematics


@dataclass(frozen=True)
class EntityDetection:
    entity_id: str
    relative_mount_m: torch.Tensor
    range_m: float


class AbstractEntitySensor:
    kind = "abstract"
    requirements = frozenset({"geometry.entities"})

    def __init__(self, config: SensorConfig, *, max_range_m: float, horizontal_fov_rad: float) -> None:
        if config.sensor_type != "abstract_entities" or not math.isfinite(max_range_m) or max_range_m <= 0:
            raise PhysicalValidationError("Invalid abstract sensor type or range")
        if not math.isfinite(horizontal_fov_rad) or not (0 < horizontal_fov_rad <= 2 * math.pi):
            raise PhysicalValidationError("Invalid abstract sensor FOV")
        self.config = config
        self.max_range_m = max_range_m
        self.horizontal_fov_rad = horizontal_fov_rad

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket:
        mount = mount_kinematics(self.config, context)
        detections: list[EntityDetection] = []
        for entity in context.world.entities(sim_time_s=context.sim_time_s, env_id=self.config.env_id):
            target = mount.position_ned_m.new_tensor(entity.position_ned_m)
            relative = rotate_world_to_body(target - mount.position_ned_m, mount.q_mount_to_ned)
            range_m = torch.linalg.vector_norm(relative).item()
            bearing = math.atan2(relative[1].item(), relative[0].item())
            if range_m > self.max_range_m or abs(bearing) > self.horizontal_fov_rad / 2:
                continue
            noisy = relative + stable_noise(self.config, sample_step, (3,), dtype=relative.dtype,
                device=relative.device, stream=f"entity:{entity.id}")
            detections.append(EntityDetection(entity.id, noisy, torch.linalg.vector_norm(noisy).item()))
        return packet(self.config, self.kind, sample_step, {"detections": tuple(detections)},
                      {"detections": "m"}, {"detections": "mount"})
