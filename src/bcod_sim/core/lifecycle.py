"""Episode, agent, and termination state contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from bcod_sim.actuators.azipod import AzipodCommand
from bcod_sim.actuators.rudder import PropellerRudderCommand
from bcod_sim.actuators.thruster import ThrustCommand
from bcod_sim.actuators.generic import ScalarCommand, VectorCommand
from bcod_sim.core.errors import PhysicalValidationError


class TerminationReason(StrEnum):
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    AGENT_DISABLED = "agent_disabled"
    PHYSICS_FAILURE = "physics_failure"
    CONTRACT_FAILURE = "runtime_contract_failure"
    EXTERNAL_DATA_FAILURE = "external_data_failure"
    EXTERNAL_STOP = "external_stop"
    TIME_LIMIT = "time_limit"


CommandValue = ThrustCommand | AzipodCommand | PropellerRudderCommand | ScalarCommand | VectorCommand


@dataclass(frozen=True)
class DirectAction:
    commands: tuple[tuple[str, CommandValue], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.commands, tuple) or any(not isinstance(item, tuple) or len(item) != 2 or
            not isinstance(item[0], str) or not isinstance(item[1],
            (ThrustCommand, AzipodCommand, PropellerRudderCommand, ScalarCommand, VectorCommand))
            for item in self.commands):
            raise PhysicalValidationError("Direct actions must contain immutable typed command values")
        if any(isinstance(command, VectorCommand) and not isinstance(command.values, tuple)
               for _, command in self.commands):
            raise PhysicalValidationError("Vector action values must be immutable tuples")
        names = [name for name, _ in self.commands]
        if len(names) != len(set(names)):
            raise PhysicalValidationError("Duplicate actuator command in direct action")


@dataclass(frozen=True)
class AgentStatus:
    rl_active: bool
    physical_active: bool
    controller: Literal["policy_direct", "policy_high_level", "scripted"]
    disabled_reason: str | None = None
