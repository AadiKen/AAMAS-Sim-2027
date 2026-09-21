"""Steerable azimuth pod with explicit steering bounds and slew."""

from dataclasses import dataclass
import math

from bcod_sim.actuators.base import (
    ActuatorConfig, ActuatorResult, ActuatorState, Bounds, advance_thrust,
    declared_power, enforce_bounds, mounted_wrench,
)
from bcod_sim.core.errors import PhysicalValidationError


@dataclass(frozen=True)
class AzipodCommand:
    thrust_n: float
    azimuth_rad: float


class Azipod:
    def __init__(self, config: ActuatorConfig, azimuth_bounds_rad: Bounds,
                 azimuth_rate_limit_radps: float | None = None) -> None:
        if azimuth_rate_limit_radps is not None and (not math.isfinite(azimuth_rate_limit_radps) or azimuth_rate_limit_radps <= 0):
            raise PhysicalValidationError("Invalid azimuth slew rate")
        self.config = config
        self.azimuth_bounds_rad = azimuth_bounds_rad
        self.azimuth_rate_limit_radps = azimuth_rate_limit_radps

    def step(self, command: AzipodCommand, state: ActuatorState, dt_s: float) -> ActuatorResult:
        if not isinstance(command, AzipodCommand):
            raise TypeError("Azipod requires AzipodCommand")
        thrust_target, thrust_events = enforce_bounds(command.thrust_n, self.config.thrust_bounds_n,
                                                      self.config.command_bounds_policy, self.config, "thrust_n")
        angle_target, angle_events = enforce_bounds(command.azimuth_rad, self.azimuth_bounds_rad,
                                                    self.config.command_bounds_policy, self.config, "azimuth_rad")
        thrust = advance_thrust(thrust_target, state.thrust_n, dt_s, self.config)
        angle = angle_target
        if self.azimuth_rate_limit_radps is not None:
            delta = self.azimuth_rate_limit_radps * dt_s
            angle = min(max(angle, state.steering_rad - delta), state.steering_rad + delta)
        force = (thrust * math.cos(angle), thrust * math.sin(angle), 0.0)
        return ActuatorResult(ActuatorState(thrust, angle), mounted_wrench(self.config, force),
                              declared_power(self.config, thrust), thrust_events + angle_events)
