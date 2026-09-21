"""Waypoint, team waypoint, formation, and coverage primitives."""

import math
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.tasks.base import StatefulTask, TaskEvaluation


class StrictTaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WaypointParams(StrictTaskParams):
    kind: Literal["waypoint"]
    agent_id: str = Field(min_length=1)
    target_ned_m: tuple[float, float, float]
    radius_m: float = Field(gt=0, allow_inf_nan=False)
    success_bonus: float = Field(default=10.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def finite_target(self):
        if not all(math.isfinite(x) for x in self.target_ned_m):
            raise ValueError("Waypoint target must be finite")
        return self


class TeamWaypointParams(StrictTaskParams):
    kind: Literal["waypoint_team"]
    target_ned_m: tuple[float, float, float]
    radius_m: float = Field(gt=0, allow_inf_nan=False)
    success_bonus: float = Field(default=10.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def finite_target(self):
        if not all(math.isfinite(x) for x in self.target_ned_m):
            raise ValueError("Team waypoint target must be finite")
        return self


class FormationParams(StrictTaskParams):
    kind: Literal["formation"]
    leader_id: str = Field(min_length=1)
    offsets_ned_m: dict[str, tuple[float, float, float]]
    tolerance_m: float = Field(gt=0, allow_inf_nan=False)
    success_bonus: float = Field(default=10.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def followers(self):
        if (not self.offsets_ned_m or self.leader_id in self.offsets_ned_m or
            any(not name or not all(math.isfinite(x) for x in offset)
                for name, offset in self.offsets_ned_m.items())):
            raise ValueError("Formation requires follower offsets distinct from leader")
        return self


class CoverageParams(StrictTaskParams):
    kind: Literal["coverage"]
    min_north_m: float = Field(allow_inf_nan=False)
    max_north_m: float = Field(allow_inf_nan=False)
    min_east_m: float = Field(allow_inf_nan=False)
    max_east_m: float = Field(allow_inf_nan=False)
    cell_size_m: float = Field(gt=0, allow_inf_nan=False)
    required_fraction: float = Field(gt=0, le=1, allow_inf_nan=False)
    success_bonus: float = Field(default=10.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def region(self):
        if self.min_north_m >= self.max_north_m or self.min_east_m >= self.max_east_m:
            raise ValueError("Coverage region must have positive extent")
        return self


def _distance(a: VesselState, target: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a.position_ned[i].item() - target[i])**2 for i in range(3)))


class WaypointTask(StatefulTask):
    def __init__(self, params: WaypointParams) -> None:
        self.params = params
        self.previous_distance: float | None = None
        self.completed = False

    def reset(self, states: Mapping[str, VesselState]) -> None:
        if self.params.agent_id not in states:
            raise PhysicalValidationError("Waypoint task refers to missing vessel")
        self.previous_distance = _distance(states[self.params.agent_id], self.params.target_ned_m)
        self.completed = False

    def evaluate(self, states: Mapping[str, VesselState], active_agent_ids: tuple[str, ...]) -> TaskEvaluation:
        agent = self.params.agent_id
        if agent not in active_agent_ids:
            return TaskEvaluation({}, {})
        distance = _distance(states[agent], self.params.target_ned_m)
        assert self.previous_distance is not None
        progress = self.previous_distance - distance
        self.previous_distance = distance
        success = distance <= self.params.radius_m and not self.completed
        self.completed |= success
        return TaskEvaluation({agent: {"waypoint_progress": progress,
                                       "waypoint_success": self.params.success_bonus if success else 0.0}},
                              {}, success=success)


class TeamWaypointTask(StatefulTask):
    def __init__(self, params: TeamWaypointParams) -> None:
        self.params = params
        self.previous: dict[str, float] = {}
        self.completed = False

    def reset(self, states: Mapping[str, VesselState]) -> None:
        self.previous = {name: _distance(state, self.params.target_ned_m) for name, state in states.items()}
        self.completed = False

    def evaluate(self, states: Mapping[str, VesselState], active_agent_ids: tuple[str, ...]) -> TaskEvaluation:
        if not active_agent_ids:
            return TaskEvaluation({}, {})
        individual = {}
        distances = []
        for name in active_agent_ids:
            distance = _distance(states[name], self.params.target_ned_m)
            individual[name] = {"waypoint_progress": self.previous[name] - distance}
            self.previous[name] = distance
            distances.append(distance)
        team_progress = sum(values["waypoint_progress"] for values in individual.values()) / len(individual)
        success = all(distance <= self.params.radius_m for distance in distances) and not self.completed
        self.completed |= success
        return TaskEvaluation(individual, {"team_progress": team_progress,
                                           "team_success": self.params.success_bonus if success else 0.0},
                              success=success)


class FormationTask(StatefulTask):
    def __init__(self, params: FormationParams) -> None:
        self.params = params
        self.previous: dict[str, float] = {}
        self.completed = False

    def _errors(self, states: Mapping[str, VesselState]) -> dict[str, float]:
        if self.params.leader_id not in states or any(name not in states for name in self.params.offsets_ned_m):
            raise PhysicalValidationError("Formation task refers to missing vessel")
        leader = states[self.params.leader_id].position_ned
        return {name: math.sqrt(sum((states[name].position_ned[i].item() -
                    leader[i].item() - offset[i])**2 for i in range(3)))
                for name, offset in self.params.offsets_ned_m.items()}

    def reset(self, states: Mapping[str, VesselState]) -> None:
        self.previous = self._errors(states)
        self.completed = False

    def evaluate(self, states: Mapping[str, VesselState], active_agent_ids: tuple[str, ...]) -> TaskEvaluation:
        errors = self._errors(states)
        individual = {name: {"formation_progress": self.previous[name] - error}
                      for name, error in errors.items() if name in active_agent_ids}
        self.previous = errors
        success = (self.params.leader_id in active_agent_ids and
                   all(name in active_agent_ids and errors[name] <= self.params.tolerance_m
                       for name in self.params.offsets_ned_m) and not self.completed)
        self.completed |= success
        return TaskEvaluation(individual, {"formation_success": self.params.success_bonus if success else 0.0},
                              success=success)


class CoverageTask(StatefulTask):
    def __init__(self, params: CoverageParams) -> None:
        self.params = params
        self.visited: set[tuple[int, int]] = set()
        self.completed = False
        self.north_cells = math.ceil((params.max_north_m-params.min_north_m)/params.cell_size_m)
        self.east_cells = math.ceil((params.max_east_m-params.min_east_m)/params.cell_size_m)

    def reset(self, states: Mapping[str, VesselState]) -> None:
        self.visited.clear()
        self.completed = False

    def evaluate(self, states: Mapping[str, VesselState], active_agent_ids: tuple[str, ...]) -> TaskEvaluation:
        newly_visited = 0
        for name in active_agent_ids:
            north, east = states[name].position_ned[:2].tolist()
            if not (self.params.min_north_m <= north < self.params.max_north_m and
                    self.params.min_east_m <= east < self.params.max_east_m):
                continue
            cell = (min(self.north_cells-1, int((north-self.params.min_north_m)/self.params.cell_size_m)),
                    min(self.east_cells-1, int((east-self.params.min_east_m)/self.params.cell_size_m)))
            if cell not in self.visited:
                self.visited.add(cell)
                newly_visited += 1
        fraction = len(self.visited)/(self.north_cells*self.east_cells)
        success = fraction >= self.params.required_fraction and not self.completed
        self.completed |= success
        return TaskEvaluation({}, {"new_cells": float(newly_visited),
                                   "coverage_success": self.params.success_bonus if success else 0.0},
                              success=success)
