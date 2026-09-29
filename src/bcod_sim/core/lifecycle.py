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
class PhysicalAction:
    """Direct normalized numeric commands for an ActuatorPipeline vessel."""
    commands: tuple[tuple[str, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        import math
        if not isinstance(self.commands, tuple) or any(not isinstance(k, str) or not isinstance(v, tuple) or
                not v or not all(math.isfinite(float(x)) for x in v) for k, v in self.commands):
            raise PhysicalValidationError("Physical action commands must be immutable finite numeric tuples")
        if len({k for k, _ in self.commands}) != len(self.commands):
            raise PhysicalValidationError("Duplicate physical actuator command")


def common_differential_to_thruster_commands(thrust_percent: float, difference_percent: float,
                                             *, port_id: str = "port", starboard_id: str = "starboard",
                                             command_limit_percent: float = 100.) -> PhysicalAction:
    """Map logged common/differential percentages to normalized direct commands.

    Mixing is performed in the recorded command domain first, then each thruster
    command is normalized and saturated independently to [-1, 1].
    """
    import math
    if (not math.isfinite(thrust_percent) or not math.isfinite(difference_percent) or
            not math.isfinite(command_limit_percent) or command_limit_percent <= 0 or
            not port_id or not starboard_id or port_id == starboard_id):
        raise PhysicalValidationError("Invalid common/differential actuator command")
    port = max(-1., min(1., (thrust_percent + difference_percent) / command_limit_percent))
    starboard = max(-1., min(1., (thrust_percent - difference_percent) / command_limit_percent))
    return PhysicalAction(((port_id, (port,)), (starboard_id, (starboard,))))


@dataclass(frozen=True)
class AgentStatus:
    rl_active: bool
    physical_active: bool
    controller: Literal["policy_direct", "policy_high_level", "scripted"]
    disabled_reason: str | None = None
