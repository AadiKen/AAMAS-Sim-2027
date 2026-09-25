"""Portable geometry, policy observation, reward, and episode accounting."""

from dataclasses import dataclass, asdict
import json
import math
import random
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

NAMES = ("vessel_0", "vessel_1", "vessel_2", "vessel_3")


@dataclass(frozen=True)
class BenchmarkConfig:
    half_width_m: float = 50.0
    goal_radius_m: float = 2.0
    vessel_radius_m: float = 1.0
    max_steps: int = 600
    dt_s: float = 0.2
    max_surge_mps: float = 2.0
    max_yaw_rps: float = 0.5
    visibility_m: float = 30.0
    progress_weight: float = 1.0
    goal_bonus: float = 20.0
    collision_penalty: float = 50.0
    step_penalty: float = 0.01


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
    obstacles: tuple[Obstacle, ...]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        data = json.loads(Path(path).read_text())
        return cls(data["split"], data["seed"],
                   tuple(VesselPose(**v) for v in data["starts"]),
                   tuple(tuple(g) for g in data["goals"]),
                   tuple(Obstacle(**o) for o in data["obstacles"]))


def generate_scenario(seed: int, split: str = "train", *, stress: bool = False,
                      config: BenchmarkConfig = BenchmarkConfig()) -> Scenario:
    if split not in {"train", "nominal", "stress"} or seed < 0:
        raise ValueError("Invalid scenario split or seed")
    rng = random.Random(f"benchmark-v1:{split}:{seed}")
    extent = config.half_width_m
    # Opposite-side assignments create crossing routes without forcing an exact center crossing.
    angles = [i * math.pi / 2 + rng.uniform(-0.16, 0.16) for i in range(4)]
    starts = []
    goals = []
    for angle in angles:
        sx = 0.72 * extent * math.cos(angle) + rng.uniform(-2, 2)
        sy = 0.72 * extent * math.sin(angle) + rng.uniform(-2, 2)
        gx = -sx + rng.uniform(-5, 5)
        gy = -sy + rng.uniform(-5, 5)
        starts.append(VesselPose(sx, sy, math.atan2(gy - sy, gx - sx)))
        goals.append((gx, gy))
    obstacles = []
    target = rng.randint(8, 10) if stress else rng.randint(6, 8)
    for _ in range(1000):
        if len(obstacles) == target:
            break
        candidate = Obstacle(rng.uniform(-0.52 * extent, 0.52 * extent),
                             rng.uniform(-0.52 * extent, 0.52 * extent), rng.uniform(1.2, 2.2))
        points = [(s.x_m, s.y_m) for s in starts] + goals
        if all(math.dist((candidate.x_m, candidate.y_m), p) > candidate.radius_m + 6 for p in points) and all(
                math.dist((candidate.x_m, candidate.y_m), (o.x_m, o.y_m)) > candidate.radius_m + o.radius_m + 2
                for o in obstacles):
            obstacles.append(candidate)
    if len(obstacles) != target:
        raise RuntimeError("Could not place obstacle geometry")
    return Scenario(split, seed, tuple(starts), tuple(goals), tuple(obstacles))


@dataclass(frozen=True)
class VesselReading:
    """Agent-visible sensor estimate in common east/north planar coordinates."""
    x_m: float
    y_m: float
    heading_rad: float
    surge_mps: float
    yaw_rps: float
    nearby_agents: tuple[tuple[float, float], ...] = ()
    nearby_obstacles: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class Truth:
    x_m: float
    y_m: float
    heading_rad: float


class SimulatorAdapter(Protocol):
    def reset(self, scenario: Scenario, seed: int) -> tuple[Mapping[str, VesselReading], Mapping[str, Truth]]: ...
    def step(self, actions: Mapping[str, tuple[float, float]]) -> tuple[Mapping[str, VesselReading], Mapping[str, Truth]]: ...
    def close(self) -> None: ...


def observation(reading: VesselReading, goal: tuple[float, float], config: BenchmarkConfig) -> np.ndarray:
    scale = config.half_width_m
    vector = [reading.x_m / scale, reading.y_m / scale,
              math.sin(reading.heading_rad), math.cos(reading.heading_rad),
              reading.surge_mps / config.max_surge_mps, reading.yaw_rps / config.max_yaw_rps,
              (goal[0] - reading.x_m) / (2 * scale), (goal[1] - reading.y_m) / (2 * scale)]
    for detected in (sorted(reading.nearby_agents, key=lambda p: math.dist(p, (reading.x_m, reading.y_m)))[:3],
                     sorted(reading.nearby_obstacles, key=lambda p: math.dist(p[:2], (reading.x_m, reading.y_m)))[:10]):
        count = 3 if len(vector) == 8 else 10
        for item in list(detected) + [None] * (count - len(detected)):
            if item is None or math.dist(item[:2], (reading.x_m, reading.y_m)) > config.visibility_m:
                vector.extend((0.0, 0.0, 0.0))
            else:
                dx, dy = item[0] - reading.x_m, item[1] - reading.y_m
                bearing = math.atan2(dy, dx) - reading.heading_rad
                vector.extend((math.hypot(dx, dy) / config.visibility_m, math.sin(bearing), math.cos(bearing)))
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (47,) or not np.isfinite(result).all():
        raise ValueError("Invalid benchmark observation")
    return np.clip(result, -1.0, 1.0)


class Benchmark:
    def __init__(self, adapter: SimulatorAdapter, config: BenchmarkConfig = BenchmarkConfig()):
        self.adapter, self.config = adapter, config

    def reset(self, scenario: Scenario):
        if len(scenario.starts) != 4 or len(scenario.goals) != 4 or not 6 <= len(scenario.obstacles) <= 10:
            raise ValueError("Scenario must contain four vessels and 6–10 obstacles")
        self.scenario = scenario
        readings, self.truth = self.adapter.reset(scenario, scenario.seed)
        self._validate(readings, self.truth)
        self.initial_distances = [math.dist((s.x_m, s.y_m), g) for s, g in zip(scenario.starts, scenario.goals)]
        self.reached = [False] * 4
        self.collided = False
        self.collision_count = 0
        self.steps = 0
        self.path_lengths = [0.0] * 4
        return self._observations(readings)

    def _validate(self, readings, truth):
        if set(readings) != set(NAMES) or set(truth) != set(NAMES):
            raise ValueError("Adapter must supply exactly four vessel readings and truth states")

    def _observations(self, readings):
        return {name: observation(readings[name], self.scenario.goals[i], self.config)
                for i, name in enumerate(NAMES)}

    def step(self, actions: Mapping[str, tuple[float, float]]):
        if self.steps >= self.config.max_steps or self.collided or all(self.reached):
            raise RuntimeError("Episode has ended")
        if set(actions) != set(NAMES):
            raise ValueError("Exactly four actions required")
        for action in actions.values():
            if len(action) != 2 or any(not math.isfinite(x) or abs(x) > 1 for x in action):
                raise ValueError("Actions must be finite normalized (surge, yaw) pairs")
        readings, truth = self.adapter.step(actions)
        self._validate(readings, truth)
        old_truth = self.truth
        self.truth = truth
        self.steps += 1
        collisions = set()
        def segment_distance(a, b, center):
            vx, vy = b[0] - a[0], b[1] - a[1]
            denominator = vx * vx + vy * vy
            fraction = 0.0 if denominator == 0 else max(0.0, min(1.0,
                ((center[0] - a[0]) * vx + (center[1] - a[1]) * vy) / denominator))
            return math.dist((a[0] + fraction * vx, a[1] + fraction * vy), center)
        for i, name in enumerate(NAMES):
            p = truth[name]
            old = old_truth[name]
            self.path_lengths[i] += math.dist((p.x_m, p.y_m), (old_truth[name].x_m, old_truth[name].y_m))
            if abs(p.x_m) + self.config.vessel_radius_m > self.config.half_width_m or abs(p.y_m) + self.config.vessel_radius_m > self.config.half_width_m:
                collisions.add(name)
            if any(segment_distance((old.x_m, old.y_m), (p.x_m, p.y_m), (o.x_m, o.y_m))
                   <= self.config.vessel_radius_m + o.radius_m for o in self.scenario.obstacles):
                collisions.add(name)
            for other in NAMES[i + 1:]:
                q = truth[other]
                old_q = old_truth[other]
                if (segment_distance((old.x_m - old_q.x_m, old.y_m - old_q.y_m),
                                     (p.x_m - q.x_m, p.y_m - q.y_m), (0., 0.))
                        <= 2 * self.config.vessel_radius_m):
                    collisions.update((name, other))
        rewards = {}
        for i, name in enumerate(NAMES):
            goal = self.scenario.goals[i]
            before = math.dist((old_truth[name].x_m, old_truth[name].y_m), goal)
            after = math.dist((truth[name].x_m, truth[name].y_m), goal)
            newly_reached = not self.reached[i] and after <= self.config.goal_radius_m
            rewards[name] = (self.config.progress_weight * (before - after)
                             + self.config.goal_bonus * newly_reached
                             - self.config.collision_penalty * (name in collisions)
                             - self.config.step_penalty)
            self.reached[i] |= newly_reached
        self.collided |= bool(collisions)
        self.collision_count += len(collisions)
        terminated = self.collided or all(self.reached)
        truncated = self.steps >= self.config.max_steps and not terminated
        info = {"fleet_success": all(self.reached) and not self.collided,
                "per_agent_success": dict(zip(NAMES, self.reached)),
                "collision_count": self.collision_count, "steps": self.steps,
                "completion_time_s": self.steps * self.config.dt_s,
                "path_length_m": dict(zip(NAMES, self.path_lengths)),
                "path_efficiency": {name: self.initial_distances[i] / max(self.path_lengths[i], 1e-9)
                                    if self.reached[i] else 0.0 for i, name in enumerate(NAMES)}}
        return self._observations(readings), rewards, terminated, truncated, info

    def close(self):
        self.adapter.close()
