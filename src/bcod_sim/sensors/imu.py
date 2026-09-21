"""Mounted specific force and angular rate in the sensor's local frame."""

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.frames.tensor import rotate_world_to_body
from bcod_sim.sensors.base import SensorConfig, SensorContext, SensorPacket, packet, stable_noise
from bcod_sim.sensors.kinematics import mount_kinematics


class IMU:
    kind = "physical"

    def __init__(self, config: SensorConfig) -> None:
        if config.sensor_type != "imu":
            raise PhysicalValidationError("IMU requires type imu")
        self.config = config

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket:
        mount = mount_kinematics(self.config, context)
        if mount.acceleration_body_mps2 is None:
            raise PhysicalValidationError("IMU requires current body linear and angular acceleration")
        state = context.state
        gravity_ned = state.position_ned.new_tensor((0, 0, 9.80665))
        gravity_body = rotate_world_to_body(gravity_ned, state.q_body_to_ned)
        q_mount = state.q_body_to_ned.new_tensor(self.config.mount_q_to_frd)
        specific_force = rotate_world_to_body(mount.acceleration_body_mps2 - gravity_body, q_mount)
        angular_rate = rotate_world_to_body(state.nu_body[3:], q_mount)
        specific_force = specific_force + stable_noise(self.config, sample_step, (3,),
            dtype=specific_force.dtype, device=specific_force.device, stream="specific_force")
        angular_rate = angular_rate + stable_noise(self.config, sample_step, (3,),
            dtype=angular_rate.dtype, device=angular_rate.device, stream="angular_rate")
        return packet(self.config, self.kind, sample_step,
                      {"specific_force_mount_mps2": specific_force, "angular_rate_mount_radps": angular_rate},
                      {"specific_force_mount_mps2": "m/s^2", "angular_rate_mount_radps": "rad/s"},
                      {"specific_force_mount_mps2": "mount", "angular_rate_mount_radps": "mount"})
