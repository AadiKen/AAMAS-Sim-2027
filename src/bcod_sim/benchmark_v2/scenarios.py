"""Versioned scenario geometry and deterministic straight-goal banks."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable


@dataclass(frozen=True)
class VesselPose:
    x_m: float
    y_m: float
    heading_rad: float


@dataclass(frozen=True)
class Obstacle:
    x_m: float
    y_m: float
    radius_m: float


@dataclass(frozen=True)
class Scenario:
    split: str
    seed: int
    starts: tuple[VesselPose, ...]
    goals: tuple[tuple[float, float], ...]
    obstacles: tuple[Obstacle, ...] = ()

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(f"vessel_{i}" for i in range(len(self.starts)))

    def geometry_hash(self) -> str:
        geometry = {"starts": [asdict(s) for s in self.starts],
                    "goals": self.goals, "obstacles": [asdict(o) for o in self.obstacles]}
        return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def case_a_reference(seed: int = 11) -> Scenario:
    return Scenario("case_a_reference", seed, (VesselPose(-20., 0., 0.),), ((20., 0.),))


def case_a_validation(seed: int = 11) -> tuple[Scenario, ...]:
    # Fixed geometries, not ten identical scenarios with different seeds.
    cases = []
    for i in range(10):
        sx = -24. + 1.7 * i
        sy = -12. + 2.4 * (i % 5)
        distance = 23. + 2.2 * i
        angle = (-0.30 + 0.065 * i)
        goal = (sx + distance * math.cos(angle), sy + distance * math.sin(angle))
        cases.append(Scenario("case_a_validation", seed + i,
                              (VesselPose(sx, sy, angle),), (goal,)))
    if len({c.geometry_hash() for c in cases}) != len(cases):
        raise AssertionError("Duplicate validation geometry")
    return tuple(cases)


def validate_scenario(scenario: Scenario, config) -> None:
    if len(scenario.starts) != config.agent_count or len(scenario.goals) != config.agent_count:
        raise ValueError("Scenario agent count differs from task configuration")
    for pose, goal in zip(scenario.starts, scenario.goals):
        values = (pose.x_m, pose.y_m, pose.heading_rad, *goal)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite scenario geometry")
        if max(abs(pose.x_m), abs(pose.y_m), abs(goal[0]), abs(goal[1])) > config.half_width_m - config.vessel_radius_m:
            raise ValueError("Scenario geometry lies outside the arena")
    for obstacle in scenario.obstacles:
        if obstacle.radius_m <= 0 or not all(math.isfinite(x) for x in (obstacle.x_m, obstacle.y_m, obstacle.radius_m)):
            raise ValueError("Invalid obstacle")
