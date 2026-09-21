"""Minimal scalar/vector wrench SDK extensions with explicit schemas."""

from dataclasses import dataclass
import math

from bcod_sim.actuators.base import ActuatorConfig, ActuatorResult, ActuatorState, Bounds, Wrench6, enforce_bounds, mounted_wrench
from bcod_sim.core.errors import PhysicalValidationError


@dataclass(frozen=True)
class ScalarCommand:
    value: float


@dataclass(frozen=True)
class VectorCommand:
    values: tuple[float, ...]


class ScalarWrenchActuator:
    def __init__(self, config: ActuatorConfig, bounds: Bounds, wrench_per_unit: Wrench6) -> None:
        if len(wrench_per_unit) != 6 or not all(math.isfinite(x) for x in wrench_per_unit):
            raise PhysicalValidationError("Scalar actuator requires finite 6-wrench mapping")
        self.config, self.bounds, self.wrench_per_unit = config, bounds, wrench_per_unit

    def step(self, command: ScalarCommand, state: ActuatorState, dt_s: float) -> ActuatorResult:
        if not isinstance(command, ScalarCommand) or not math.isfinite(dt_s) or dt_s <= 0:
            raise PhysicalValidationError("Invalid scalar actuator command or timestep")
        value, events = enforce_bounds(command.value, self.bounds, self.config.command_bounds_policy, self.config, "value")
        force = tuple(value * x for x in self.wrench_per_unit[:3])
        moment = tuple(value * x for x in self.wrench_per_unit[3:])
        return ActuatorResult(ActuatorState(value), mounted_wrench(self.config, force, moment), None, events)


class VectorWrenchActuator:
    def __init__(self, config: ActuatorConfig, bounds: tuple[Bounds, ...], wrench_columns: tuple[Wrench6, ...]) -> None:
        if len(bounds) == 0 or len(bounds) != len(wrench_columns) or any(len(w) != 6 or not all(math.isfinite(x) for x in w) for w in wrench_columns):
            raise PhysicalValidationError("Vector actuator requires matching bounds and finite wrench columns")
        self.config, self.bounds, self.wrench_columns = config, bounds, wrench_columns

    def step(self, command: VectorCommand, state: ActuatorState, dt_s: float) -> ActuatorResult:
        if not isinstance(command, VectorCommand) or len(command.values) != len(self.bounds) or not math.isfinite(dt_s) or dt_s <= 0:
            raise PhysicalValidationError("Invalid vector actuator command or timestep")
        values, events = [], ()
        for i, (value, bounds) in enumerate(zip(command.values, self.bounds)):
            applied, new_events = enforce_bounds(value, bounds, self.config.command_bounds_policy, self.config, f"value_{i}")
            values.append(applied)
            events += new_events
        wrench = tuple(math.fsum(value * column[j] for value, column in zip(values, self.wrench_columns)) for j in range(6))
        return ActuatorResult(state, mounted_wrench(self.config, wrench[:3], wrench[3:]), None, events)
