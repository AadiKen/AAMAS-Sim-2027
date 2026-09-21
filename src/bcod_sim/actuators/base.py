"""Shared actuator identity, bounds, dynamics, and body-frame wrench contracts."""

from dataclasses import dataclass
import math
from typing import Literal, Protocol

from bcod_sim.core.errors import CommandBoundsError, PhysicalValidationError
from bcod_sim.frames.transforms import body_to_world, moment_from_force, normalize_quaternion

Vec3 = tuple[float, float, float]
Wrench6 = tuple[float, float, float, float, float, float]
CommandBoundsPolicy = Literal["error", "clamp_with_event"]


@dataclass(frozen=True)
class Bounds:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum) and self.minimum < self.maximum):
            raise PhysicalValidationError("Actuator bounds must be finite and ordered")


@dataclass(frozen=True)
class ActuatorConfig:
    instance_id: str
    env_id: int
    owner_vessel_id: int
    mount_frd_m: Vec3
    mount_q_to_frd: tuple[float, float, float, float]
    thrust_bounds_n: Bounds
    command_bounds_policy: CommandBoundsPolicy = "error"
    thrust_rate_limit_nps: float | None = None
    thrust_time_constant_s: float | None = None
    deadband_n: float = 0.0
    power_coefficient_w_per_n: float | None = None

    def __post_init__(self) -> None:
        if not self.instance_id or self.env_id < 0 or self.owner_vessel_id < 0:
            raise PhysicalValidationError("Actuator requires stable ID and nonnegative owner IDs")
        if len(self.mount_frd_m) != 3 or not all(math.isfinite(v) for v in self.mount_frd_m):
            raise PhysicalValidationError("Actuator mount position must be finite FRD vector")
        q = normalize_quaternion(self.mount_q_to_frd)
        if any(abs(a-b) > 1e-8 for a, b in zip(q, self.mount_q_to_frd)):
            raise PhysicalValidationError("Actuator mount orientation must be normalized")
        if self.command_bounds_policy not in ("error", "clamp_with_event"):
            raise PhysicalValidationError("Unknown command bounds policy")
        for name, value, zero_ok in (("thrust_rate_limit_nps", self.thrust_rate_limit_nps, False),
                                      ("thrust_time_constant_s", self.thrust_time_constant_s, False),
                                      ("power_coefficient_w_per_n", self.power_coefficient_w_per_n, True)):
            if value is not None and (not math.isfinite(value) or value < 0 or (value == 0 and not zero_ok)):
                raise PhysicalValidationError(f"Invalid {name}")
        if not math.isfinite(self.deadband_n) or self.deadband_n < 0:
            raise PhysicalValidationError("Invalid actuator deadband")


@dataclass(frozen=True)
class ActuatorState:
    thrust_n: float = 0.0
    steering_rad: float = 0.0
    rpm: float = 0.0
    rudder_rad: float = 0.0


@dataclass(frozen=True)
class ActuatorClampEvent:
    kind: str
    env_id: int
    owner_vessel_id: int
    actuator_id: str
    field: str
    requested: float
    applied: float


@dataclass(frozen=True)
class ActuatorResult:
    state: ActuatorState
    wrench_frd: Wrench6
    power_w: float | None
    events: tuple[ActuatorClampEvent, ...]


def enforce_bounds(value: float, bounds: Bounds, policy: CommandBoundsPolicy,
                   config: ActuatorConfig, field: str) -> tuple[float, tuple[ActuatorClampEvent, ...]]:
    if not math.isfinite(value):
        raise CommandBoundsError(f"{config.instance_id}.{field} must be finite")
    if bounds.minimum <= value <= bounds.maximum:
        return value, ()
    if policy == "error":
        raise CommandBoundsError(f"{config.instance_id}.{field} outside [{bounds.minimum}, {bounds.maximum}]")
    applied = min(max(value, bounds.minimum), bounds.maximum)
    return applied, (ActuatorClampEvent("actuator_clamp", config.env_id, config.owner_vessel_id,
                                        config.instance_id, field, value, applied),)


def advance_thrust(target_n: float, previous_n: float, dt_s: float, config: ActuatorConfig) -> float:
    if not math.isfinite(dt_s) or dt_s <= 0 or not math.isfinite(previous_n):
        raise PhysicalValidationError("Actuator step and state must be finite; dt must be positive")
    target = 0.0 if abs(target_n) <= config.deadband_n else target_n
    if config.thrust_time_constant_s is not None:
        target = previous_n + (target - previous_n) * (1 - math.exp(-dt_s / config.thrust_time_constant_s))
    if config.thrust_rate_limit_nps is not None:
        delta = config.thrust_rate_limit_nps * dt_s
        target = min(max(target, previous_n - delta), previous_n + delta)
    return target


def mounted_wrench(config: ActuatorConfig, force_mount_n: Vec3, moment_mount_nm: Vec3 = (0.0, 0.0, 0.0)) -> Wrench6:
    force = body_to_world(force_mount_n, config.mount_q_to_frd)
    moment = body_to_world(moment_mount_nm, config.mount_q_to_frd)
    lever = moment_from_force(config.mount_frd_m, force)
    return (*force, *(a+b for a, b in zip(moment, lever)))


def declared_power(config: ActuatorConfig, thrust_n: float) -> float | None:
    if config.power_coefficient_w_per_n is None:
        return None
    return config.power_coefficient_w_per_n * abs(thrust_n)


class Actuator(Protocol):
    config: ActuatorConfig

    def step(self, command: object, state: ActuatorState, dt_s: float) -> ActuatorResult: ...
