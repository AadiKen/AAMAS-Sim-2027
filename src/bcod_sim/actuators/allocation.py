"""Deterministic actuator dispatch and per-owner wrench reduction."""

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from bcod_sim.actuators.base import Actuator, ActuatorResult, ActuatorState, Wrench6
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError
from bcod_sim.frames.transforms import body_to_world, moment_from_force


@dataclass(frozen=True)
class WrenchContribution:
    env_id: int
    owner_vessel_id: int
    actuator_id: str
    wrench_frd: Wrench6


@dataclass(frozen=True)
class BankResult:
    states: Mapping[tuple[int, int, str], ActuatorState]
    results: Mapping[tuple[int, int, str], ActuatorResult]
    wrenches_by_owner: Mapping[tuple[int, int], Wrench6]


def reduce_wrenches(contributions: tuple[WrenchContribution, ...]) -> dict[tuple[int, int], Wrench6]:
    seen: set[tuple[int, int, str]] = set()
    groups: dict[tuple[int, int], list[Wrench6]] = {}
    for item in sorted(contributions, key=lambda c: (c.env_id, c.owner_vessel_id, c.actuator_id)):
        key = (item.env_id, item.owner_vessel_id, item.actuator_id)
        if key in seen:
            raise DuplicateIdentityError(f"Duplicate actuator contribution: {key}")
        seen.add(key)
        if len(item.wrench_frd) != 6 or not all(math.isfinite(x) for x in item.wrench_frd):
            raise PhysicalValidationError("Invalid actuator wrench contribution")
        groups.setdefault(key[:2], []).append(item.wrench_frd)
    return {owner: tuple(math.fsum(row[i] for row in rows) for i in range(6)) for owner, rows in groups.items()}


class ActuatorBank:
    def __init__(self, actuators: tuple[Actuator, ...]) -> None:
        ids = [(a.config.env_id, a.config.owner_vessel_id, a.config.instance_id) for a in actuators]
        if len(ids) != len(set(ids)):
            raise DuplicateIdentityError("Actuator instance IDs must be unique per owner")
        self.actuators = {key: actuator for key, actuator in zip(ids, actuators)}
        groups: dict[type, list[tuple[int, int, str]]] = {}
        for key, actuator in self.actuators.items():
            groups.setdefault(type(actuator), []).append(key)
        self.schema_groups = {kind: tuple(sorted(keys)) for kind, keys in groups.items()}

    def step(self, commands: Mapping[tuple[int, int, str], object],
             states: Mapping[tuple[int, int, str], ActuatorState], dt_s: float) -> BankResult:
        expected = set(self.actuators)
        if set(commands) != expected or set(states) != expected:
            raise PhysicalValidationError("Every actuator needs exactly one command and state; unknown IDs are errors")
        results = {}
        for kind in sorted(self.schema_groups, key=lambda cls: cls.__name__):
            for name in self.schema_groups[kind]:
                results[name] = self.actuators[name].step(commands[name], states[name], dt_s)
        contributions = tuple(WrenchContribution(name[0], name[1], name[2], results[name].wrench_frd)
                              for name in sorted(expected))
        return BankResult({name: result.state for name, result in results.items()},
                          results, reduce_wrenches(contributions))


def allocate_fixed_thrusters(thrusters: tuple[FixedThruster, ...], *, surge_n: float, yaw_nm: float) -> dict[str, ThrustCommand]:
    """Least-norm allocation; reject unattainable requests instead of hiding residual."""
    if not thrusters or not math.isfinite(surge_n) or not math.isfinite(yaw_nm):
        raise PhysicalValidationError("Invalid high-level force request")
    owner = (thrusters[0].config.env_id, thrusters[0].config.owner_vessel_id)
    if any((t.config.env_id, t.config.owner_vessel_id) != owner for t in thrusters):
        raise PhysicalValidationError("One allocation cannot mix owners")
    columns = []
    for t in thrusters:
        force = body_to_world((1, 0, 0), t.config.mount_q_to_frd)
        moment = moment_from_force(t.config.mount_frd_m, force)
        if abs(force[1]) > 1e-10 or abs(force[2]) > 1e-10:
            raise PhysicalValidationError("High-level fixed-thruster allocator requires longitudinal axes")
        columns.append((force[0], moment[2]))
    matrix = torch.tensor(columns, dtype=torch.float64).T.contiguous()
    target = torch.tensor((surge_n, yaw_nm), dtype=torch.float64)
    solution = torch.linalg.lstsq(matrix, target).solution
    if not torch.allclose(matrix @ solution, target, atol=1e-8, rtol=1e-8):
        raise PhysicalValidationError("Requested surge/yaw wrench is unattainable")
    return {thruster.config.instance_id: ThrustCommand(solution[i].item()) for i, thruster in enumerate(thrusters)}
