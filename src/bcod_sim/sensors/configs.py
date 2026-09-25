"""Typed built-in sensor-specific error and geometry configuration."""
from dataclasses import dataclass
import math

from bcod_sim.core.errors import PhysicalValidationError


def _nonnegative(values):
    return all(math.isfinite(value) and value >= 0 for value in values)


@dataclass(frozen=True)
class GPSErrorModel:
    position_noise_std_m: float = 0.0
    velocity_noise_std_mps: float = 0.0
    position_bias_ned_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_bias_ned_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not _nonnegative((self.position_noise_std_m, self.velocity_noise_std_mps)):
            raise PhysicalValidationError("GPS noise must be finite and nonnegative")


@dataclass(frozen=True)
class IMUErrorModel:
    accel_noise_std_mps2: float = 0.0
    gyro_noise_std_radps: float = 0.0
    accel_bias_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_radps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_random_walk_mps2: float = 0.0
    gyro_random_walk_radps: float = 0.0
    accel_saturation_mps2: float | None = None
    gyro_saturation_radps: float | None = None

    def __post_init__(self):
        values = (self.accel_noise_std_mps2, self.gyro_noise_std_radps,
                  self.accel_random_walk_mps2, self.gyro_random_walk_radps)
        if not _nonnegative(values) or any(value is not None and (not math.isfinite(value) or value <= 0)
                                            for value in (self.accel_saturation_mps2, self.gyro_saturation_radps)):
            raise PhysicalValidationError("Invalid IMU error model")
