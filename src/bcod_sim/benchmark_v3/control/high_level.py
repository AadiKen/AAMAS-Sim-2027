"""Common East/North heading controller; positive yaw is counterclockwise."""
from dataclasses import dataclass
import math


CONTROLLER_VERSION = "heading-p-v1"


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class LowLevelCommand:
    desired_speed_mps: float
    desired_yaw_rate_radps: float
    heading_error_rad: float | None = None
    requested_yaw_rate_radps: float | None = None
    saturated: bool = False


@dataclass(frozen=True)
class HeadingController:
    max_yaw_rps: float = 0.25
    k_heading: float = 0.5

    def command(self, desired_speed_mps: float, desired_heading_rad: float,
                measured_heading_rad: float, measured_yaw_rate_radps: float,
                dt_s: float) -> LowLevelCommand:
        if dt_s <= 0 or not all(map(math.isfinite, (desired_speed_mps, desired_heading_rad,
                measured_heading_rad, measured_yaw_rate_radps))):
            raise ValueError("Invalid heading controller input")
        error = wrap_angle(desired_heading_rad - measured_heading_rad)
        requested = self.k_heading * error
        yaw = max(-self.max_yaw_rps, min(self.max_yaw_rps, requested))
        return LowLevelCommand(desired_speed_mps, yaw, error, requested,
                               abs(requested) > self.max_yaw_rps)
