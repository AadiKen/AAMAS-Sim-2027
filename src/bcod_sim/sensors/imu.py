"""Mounted specific force and angular rate in the sensor's local frame."""

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.frames.tensor import rotate_world_to_body
from bcod_sim.sensors.base import SensorConfig, SensorContext, SensorPacket, packet, stable_noise
from bcod_sim.sensors.kinematics import mount_kinematics
from bcod_sim.sensors.configs import IMUErrorModel


class IMU:
    kind = "physical"
    requirements = frozenset({"vehicle.acceleration", "vehicle.angular_velocity"})

    def __init__(self, config: SensorConfig, *, error_model: IMUErrorModel | None = None) -> None:
        if config.sensor_type != "imu":
            raise PhysicalValidationError("IMU requires type imu")
        self.config = config
        self.error_model = error_model or IMUErrorModel(config.noise_std, config.noise_std)
        self.reset()

    def reset(self) -> None:
        self._accel_random_walk: torch.Tensor | None = None
        self._gyro_random_walk: torch.Tensor | None = None
        self._last_sample_time_s: float | None = None

    def snapshot(self) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None]:
        return (None if self._accel_random_walk is None else self._accel_random_walk.clone(),
                None if self._gyro_random_walk is None else self._gyro_random_walk.clone(),
                self._last_sample_time_s)

    def restore(self, snapshot: tuple[torch.Tensor | None, torch.Tensor | None, float | None]) -> None:
        accel, gyro, sample_time = snapshot
        self._accel_random_walk = None if accel is None else accel.clone()
        self._gyro_random_walk = None if gyro is None else gyro.clone()
        self._last_sample_time_s = sample_time

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
        model = self.error_model
        if self._accel_random_walk is None:
            self._accel_random_walk = specific_force.new_zeros(3)
            self._gyro_random_walk = angular_rate.new_zeros(3)
        sample_dt_s = (context.sim_time_s - self._last_sample_time_s
                       if self._last_sample_time_s is not None and context.sim_time_s > self._last_sample_time_s
                       else 1.0 / self.config.rate_hz)
        self._accel_random_walk = self._accel_random_walk + stable_noise(
            self.config, sample_step, (3,), dtype=specific_force.dtype, device=specific_force.device,
            stream="accel_random_walk_increment", std=model.accel_random_walk_mps2 * sample_dt_s**0.5)
        self._gyro_random_walk = self._gyro_random_walk + stable_noise(
            self.config, sample_step, (3,), dtype=angular_rate.dtype, device=angular_rate.device,
            stream="gyro_random_walk_increment", std=model.gyro_random_walk_radps * sample_dt_s**0.5)
        self._last_sample_time_s = context.sim_time_s
        specific_force = specific_force + specific_force.new_tensor(model.accel_bias_mps2)
        angular_rate = angular_rate + angular_rate.new_tensor(model.gyro_bias_radps)
        specific_force = specific_force + stable_noise(self.config, sample_step, (3,),
            dtype=specific_force.dtype, device=specific_force.device, stream="specific_force",
            std=model.accel_noise_std_mps2)
        angular_rate = angular_rate + stable_noise(self.config, sample_step, (3,),
            dtype=angular_rate.dtype, device=angular_rate.device, stream="angular_rate",
            std=model.gyro_noise_std_radps)
        specific_force = specific_force + self._accel_random_walk
        angular_rate = angular_rate + self._gyro_random_walk
        if model.accel_saturation_mps2 is not None:
            specific_force = specific_force.clamp(-model.accel_saturation_mps2, model.accel_saturation_mps2)
        if model.gyro_saturation_radps is not None:
            angular_rate = angular_rate.clamp(-model.gyro_saturation_radps, model.gyro_saturation_radps)
        return packet(self.config, self.kind, sample_step,
                      {"specific_force_mount_mps2": specific_force, "angular_rate_mount_radps": angular_rate},
                      {"specific_force_mount_mps2": "m/s^2", "angular_rate_mount_radps": "rad/s"},
                      {"specific_force_mount_mps2": "mount", "angular_rate_mount_radps": "mount"})
