"""Propeller thrust plus a configured rudder side-force approximation."""

from dataclasses import dataclass
import math

from bcod_sim.actuators.base import (
    ActuatorConfig, ActuatorResult, ActuatorState, Bounds, advance_thrust,
    declared_power, enforce_bounds, mounted_wrench,
)
from bcod_sim.core.errors import PhysicalValidationError


@dataclass(frozen=True)
class PropellerRudderCommand:
    thrust_n: float
    rudder_rad: float


class PropellerRudder:
    """Rudder side force = gain * abs(propeller thrust) * sin(angle)."""

    def __init__(self, config: ActuatorConfig, rudder_bounds_rad: Bounds, side_force_gain: float,
                 rudder_rate_limit_radps: float | None = None) -> None:
        if not math.isfinite(side_force_gain) or side_force_gain < 0:
            raise PhysicalValidationError("Rudder side-force gain must be nonnegative")
        if rudder_rate_limit_radps is not None and (not math.isfinite(rudder_rate_limit_radps) or rudder_rate_limit_radps <= 0):
            raise PhysicalValidationError("Invalid rudder slew rate")
        self.config = config
        self.rudder_bounds_rad = rudder_bounds_rad
        self.side_force_gain = side_force_gain
        self.rudder_rate_limit_radps = rudder_rate_limit_radps

    def step(self, command: PropellerRudderCommand, state: ActuatorState, dt_s: float) -> ActuatorResult:
        if not isinstance(command, PropellerRudderCommand):
            raise TypeError("PropellerRudder requires PropellerRudderCommand")
        target, thrust_events = enforce_bounds(command.thrust_n, self.config.thrust_bounds_n,
                                               self.config.command_bounds_policy, self.config, "thrust_n")
        angle, rudder_events = enforce_bounds(command.rudder_rad, self.rudder_bounds_rad,
                                              self.config.command_bounds_policy, self.config, "rudder_rad")
        thrust = advance_thrust(target, state.thrust_n, dt_s, self.config)
        if self.rudder_rate_limit_radps is not None:
            delta = self.rudder_rate_limit_radps * dt_s
            angle = min(max(angle, state.steering_rad - delta), state.steering_rad + delta)
        force = (thrust, self.side_force_gain * abs(thrust) * math.sin(angle), 0.0)
        return ActuatorResult(ActuatorState(thrust, angle), mounted_wrench(self.config, force),
                              declared_power(self.config, thrust), thrust_events + rudder_events)
