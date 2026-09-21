"""Simple vessel-specific speed/heading adapter to fixed-thruster allocation."""

from dataclasses import dataclass
import math

from bcod_sim.actuators.allocation import allocate_fixed_thrusters
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.core.errors import PhysicalValidationError


@dataclass(frozen=True)
class HighLevelCommand:
    desired_speed_mps: float
    desired_heading_rad: float


@dataclass(frozen=True)
class HeadingSpeedAutopilot:
    speed_gain_n_per_mps: float
    heading_gain_nm_per_rad: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.speed_gain_n_per_mps) and self.speed_gain_n_per_mps > 0 and
                math.isfinite(self.heading_gain_nm_per_rad) and self.heading_gain_nm_per_rad > 0):
            raise PhysicalValidationError("Autopilot gains must be finite and positive")

    def commands(self, request: HighLevelCommand, *, current_speed_mps: float, current_heading_rad: float,
                 thrusters: tuple[FixedThruster, ...]) -> dict[str, ThrustCommand]:
        if not all(math.isfinite(x) for x in (request.desired_speed_mps, request.desired_heading_rad,
                                               current_speed_mps, current_heading_rad)):
            raise PhysicalValidationError("High-level command and state must be finite")
        surge = self.speed_gain_n_per_mps * (request.desired_speed_mps - current_speed_mps)
        heading_error = math.atan2(math.sin(request.desired_heading_rad - current_heading_rad),
                                   math.cos(request.desired_heading_rad - current_heading_rad))
        yaw = self.heading_gain_nm_per_rad * heading_error
        return allocate_fixed_thrusters(thrusters, surge_n=surge, yaw_nm=yaw)
