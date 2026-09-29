"""Versioned two-vessel M0 crossing and yielding scenario distribution."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math

import numpy as np

from bcod_sim.benchmark_v2.scenarios import Scenario, VesselPose, validate_scenario
from ..config import TaskConfig


M0_SCENARIO_VERSION = "m0-two-vessel-crossing-v1"
M0_CURRICULUM_VERSION = "m0-highlevel-curriculum-v1"


def sample_m0(rng: np.random.Generator, *, split="M0-train", layout=None):
    if layout is None:
        layout = int(rng.integers(0, 5))
    if layout not in range(5):
        raise ValueError("Unsupported M0 layout")
    seed = int(rng.integers(0, 2**31 - 1))
    d0, d1 = rng.uniform(13., 20., size=2)
    offset = float(rng.uniform(-3., 3.))
    if layout == 0:  # opposing courses, staggered by a small lateral offset
        starts = [(-d0, offset), (d1, -offset)]
        goals = [(d0, offset), (-d1, -offset)]
    elif layout == 1:  # perpendicular crossing
        starts = [(-d0, offset), (offset, -d1)]
        goals = [(d0, offset), (offset, d1)]
    elif layout == 2:  # oblique crossing with distinct goal locations
        starts = [(-d0, -5. + offset), (-d1, 7. - offset)]
        goals = [(d0, 6. + offset), (d1, -6. - offset)]
    elif layout == 3:  # converging lanes, separated final goals
        starts = [(-d0, -7. + offset), (-d1, 7. - offset)]
        goals = [(d0, -2. + offset), (d1, 2. - offset)]
    else:  # offset arrival timing at a perpendicular crossing
        starts = [(-d0, offset), (offset, -d1)]
        goals = [(d0, offset), (offset, d1)]
        starts[1] = (starts[1][0], starts[1][1] - float(rng.uniform(4., 8.)))
    angle = float(rng.uniform(-math.pi, math.pi))
    sign = -1. if int(rng.integers(0, 2)) else 1.
    tx, ty = rng.uniform(-4., 4., size=2)
    def transform(p):
        x, y = p[0], sign * p[1]
        return (float(tx + x * math.cos(angle) - y * math.sin(angle)),
                float(ty + x * math.sin(angle) + y * math.cos(angle)))
    start_world, goal_world = tuple(map(transform, starts)), tuple(map(transform, goals))
    poses = tuple(VesselPose(*start, math.atan2(goal[1] - start[1], goal[0] - start[0]))
                  for start, goal in zip(start_world, goal_world))
    scenario = Scenario(split, seed, poses, goal_world)
    validate_scenario(scenario, TaskConfig(agent_count=2, action_mode="high_level"))
    return scenario


def _bank(count, seed, split):
    rng = np.random.default_rng(seed)
    cases, hashes = [], set()
    while len(cases) < count:
        case = sample_m0(rng, split=split, layout=len(cases) % 5)
        digest = case.geometry_hash()
        if digest not in hashes:
            cases.append(case)
            hashes.add(digest)
    return tuple(cases)


@lru_cache(maxsize=1)
def m0_dev_bank():
    return _bank(50, 6101, "M0-dev")


@lru_cache(maxsize=1)
def m0_test_bank():
    bank = _bank(100, 6201, "M0-test")
    assert not ({c.geometry_hash() for c in bank} &
                {c.geometry_hash() for c in m0_dev_bank()})
    return bank


@lru_cache(maxsize=1)
def reserved_m0_hashes():
    return frozenset(c.geometry_hash() for c in (*m0_dev_bank(), *m0_test_bank()))


def sample_m0_train(rng):
    reserved = reserved_m0_hashes()
    while True:
        case = sample_m0(rng)
        if case.geometry_hash() not in reserved:
            return case


def bank_hash(bank):
    return hashlib.sha256(json.dumps([case.geometry_hash() for case in bank],
                                     separators=(",", ":")).encode()).hexdigest()


def sample_m0_curriculum(rng: np.random.Generator, stage: str, *, split="M0-train"):
    """Sample a varied geometry at the requested interaction difficulty."""
    if stage == "C":
        return sample_m0(rng, split=split)
    if stage == "B":
        # Avoid the strongest head-on conflicts until the final stage.
        return sample_m0(rng, split=split, layout=int(rng.integers(1, 5)))
    if stage != "A":
        raise ValueError(f"Unknown M0 curriculum stage {stage}")
    seed = int(rng.integers(0, 2**31 - 1))
    layout = int(rng.integers(0, 4))
    distance = rng.uniform(13., 19., size=2)
    offset = float(rng.uniform(-1.5, 1.5))
    # Parallel, distinct goal lanes, offset travel, and a mild merge.
    lane0 = -5. + offset
    lane1 = 5. + offset
    starts = [(-distance[0], lane0), (-distance[1] + (3. if layout == 2 else 0.), lane1)]
    goals = [(distance[0], lane0 + (1. if layout == 3 else 0.)),
             (distance[1], lane1 - (1. if layout == 3 else 0.))]
    if layout == 1:
        goals[0] = (distance[0], lane0 - 1.)
        goals[1] = (distance[1], lane1 + 1.)
    angle = float(rng.uniform(-math.pi, math.pi))
    sign = -1. if int(rng.integers(0, 2)) else 1.
    tx, ty = rng.uniform(-4., 4., size=2)
    def transform(point):
        x, y = point[0], sign * point[1]
        return (float(tx + x * math.cos(angle) - y * math.sin(angle)),
                float(ty + x * math.sin(angle) + y * math.cos(angle)))
    starts, goals = tuple(map(transform, starts)), tuple(map(transform, goals))
    poses = tuple(VesselPose(*start, math.atan2(goal[1]-start[1], goal[0]-start[0]))
                  for start, goal in zip(starts, goals))
    scenario = Scenario(split, seed, poses, goals)
    validate_scenario(scenario, TaskConfig(agent_count=2, action_mode="high_level"))
    return scenario


@lru_cache(maxsize=None)
def m0_curriculum_bank(stage: str, split: str):
    if stage not in ("A", "B", "C") or split not in ("dev", "test"):
        raise ValueError("Expected M0 stage A/B/C and dev/test split")
    count = 50 if split == "dev" else 100
    stage_index = "ABC".index(stage)
    rng = np.random.default_rng((7101 if split == "dev" else 7201) + stage_index)
    cases, hashes = [], set()
    while len(cases) < count:
        case = sample_m0_curriculum(rng, stage, split=f"M0-{stage}-{split}")
        digest = case.geometry_hash()
        if digest not in hashes:
            cases.append(case)
            hashes.add(digest)
    return tuple(cases)


@lru_cache(maxsize=1)
def reserved_m0_curriculum_hashes():
    banks = [m0_curriculum_bank(stage, split) for stage in "ABC"
             for split in ("dev", "test")]
    hashes = [case.geometry_hash() for bank in banks for case in bank]
    if len(hashes) != len(set(hashes)):
        raise AssertionError("Curriculum dev/test geometry overlap")
    return frozenset(hashes)


def sample_m0_curriculum_train(rng, stage):
    reserved = reserved_m0_curriculum_hashes()
    while True:
        case = sample_m0_curriculum(rng, stage)
        if case.geometry_hash() not in reserved:
            return case
