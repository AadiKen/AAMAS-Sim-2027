"""Fixed-axis thruster, local +X in its mounted frame."""

from dataclasses import dataclass

from bcod_sim.actuators.base import (
    ActuatorConfig, ActuatorResult, ActuatorState, advance_thrust,
    declared_power, enforce_bounds, mounted_wrench,
)


@dataclass(frozen=True)
class ThrustCommand:
    thrust_n: float


class FixedThruster:
    def __init__(self, config: ActuatorConfig) -> None:
        self.config = config

    def step(self, command: ThrustCommand, state: ActuatorState, dt_s: float) -> ActuatorResult:
        if not isinstance(command, ThrustCommand):
            raise TypeError("FixedThruster requires ThrustCommand")
        target, events = enforce_bounds(command.thrust_n, self.config.thrust_bounds_n,
                                         self.config.command_bounds_policy, self.config, "thrust_n")
        thrust = advance_thrust(target, state.thrust_n, dt_s, self.config)
        return ActuatorResult(ActuatorState(thrust), mounted_wrench(self.config, (thrust, 0.0, 0.0)),
                              declared_power(self.config, thrust), events)
