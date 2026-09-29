"""Scenario generation is independent of any simulator backend."""
import math
from functools import lru_cache
import numpy as np

from bcod_sim.benchmark_v2.scenarios import (Obstacle, Scenario, VesselPose,
                                             case_a_reference, validate_scenario)

S1_GENERATOR_VERSION = "s1-obstructed-v1"


def sample_s0(rng: np.random.Generator, *, seed: int | None = None, split: str = "S0-train") -> Scenario:
    """Random turn-required, obstacle-free single-vessel navigation case."""
    if seed is None:
        seed = int(rng.integers(0, 2**31 - 1))
    while True:
        start_x, start_y = rng.uniform(-24, 24, size=2)
        distance = float(rng.uniform(18, 38))
        goal_direction = float(rng.uniform(-math.pi, math.pi))
        goal = (float(start_x + distance * math.cos(goal_direction)),
                float(start_y + distance * math.sin(goal_direction)))
        if max(abs(goal[0]), abs(goal[1])) < 48:
            break
    turn = float(rng.uniform(0.45, 2.45)) * (-1 if rng.integers(0, 2) else 1)
    heading = math.atan2(math.sin(goal_direction + turn), math.cos(goal_direction + turn))
    return Scenario(split, seed, (VesselPose(float(start_x), float(start_y), heading),), (goal,))


def s0_validation_bank(count: int = 50, *, seed: int = 2027) -> tuple[Scenario, ...]:
    if count < 50:
        raise ValueError("S0 validation bank must contain at least 50 cases")
    rng = np.random.default_rng(seed)
    cases = []
    hashes = set()
    while len(cases) < count:
        case = sample_s0(rng, split="S0-validation")
        digest = case.geometry_hash()
        if digest not in hashes:
            hashes.add(digest)
            cases.append(case)
    return tuple(cases)


def sample_s1(rng: np.random.Generator, *, seed: int | None = None,
              split: str = "S1-train", layout: int | None = None) -> Scenario:
    """Feasible, obstructed one-vessel routes with 1–4 static circles.

    A clear corridor on the designated detour side is reserved by construction.
    Every layout blocks the direct start-to-goal segment by at least one circle.
    """
    if seed is None:
        seed = int(rng.integers(0, 2**31 - 1))
    if layout is None:
        layout = int(rng.integers(0, 5))
    side = 1 if int(rng.integers(0, 2)) else -1
    direction = float(rng.uniform(-math.pi, math.pi))
    ux, uy = math.cos(direction), math.sin(direction)
    nx, ny = -uy, ux
    distance = float(rng.uniform(32, 40))
    cx, cy = rng.uniform(-6, 6, size=2)
    def point(along, across):
        return (float(cx + along * ux + across * nx),
                float(cy + along * uy + across * ny))
    start = point(-distance / 2, 0)
    goal = point(distance / 2, 0)
    heading = direction + float(rng.uniform(-0.8, 0.8))
    radii = rng.uniform(2.0, 3.0, size=4)
    # The direct path is blocked at along=0; additional circles create offset,
    # gap, or multi-obstacle routes while leaving the side*8 m corridor open.
    offsets = {
        0: [(0., 0.)],
        1: [(0., 0.), (8., -side * 4.5)],
        2: [(-4., -side * 2.6), (4., side * 2.6)],
        3: [(-8., 0.), (0., -side * 3.8), (8., 0.)],
        4: [(-9., 0.), (-3., -side * 4.5), (4., 0.), (10., -side * 4.5)],
    }[layout]
    obstacles = tuple(Obstacle(*point(a, b), float(radii[i]))
                      for i, (a, b) in enumerate(offsets))
    case = Scenario(split, seed, (VesselPose(*start, heading),), (goal,), obstacles)
    validate_s1_scenario(case)
    return case


def validate_s1_scenario(case: Scenario) -> None:
    """Reject invalid geometry, unobstructed direct routes, or blocked detour corridors."""
    from .config import TaskConfig
    config = TaskConfig()
    validate_scenario(case, config)
    if len(case.starts) != 1 or not 1 <= len(case.obstacles) <= 4:
        raise ValueError("S1 requires one vessel and 1–4 static obstacles")
    start = (case.starts[0].x_m, case.starts[0].y_m)
    goal = case.goals[0]
    distance = math.dist(start, goal)
    ux, uy = (goal[0]-start[0])/distance, (goal[1]-start[1])/distance
    nx, ny = -uy, ux
    def clearance(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length2 = dx*dx + dy*dy
        return min(math.dist((o.x_m, o.y_m),
            (a[0] + max(0., min(1., ((o.x_m-a[0])*dx + (o.y_m-a[1])*dy)/length2))*dx,
             a[1] + max(0., min(1., ((o.x_m-a[0])*dx + (o.y_m-a[1])*dy)/length2))*dy))
            - o.radius_m - config.vessel_radius_m for o in case.obstacles)
    if clearance(start, goal) >= 0:
        raise ValueError("S1 direct path is not obstructed")
    feasible = False
    for side in (-1, 1):
        path = [start,
                (start[0] + 4*ux + side*8*nx, start[1] + 4*uy + side*8*ny),
                (goal[0] - 4*ux + side*8*nx, goal[1] - 4*uy + side*8*ny), goal]
        if all(max(abs(x), abs(y)) < config.half_width_m-config.vessel_radius_m
               for x, y in path) and all(clearance(a, b) > 0.25 for a, b in zip(path, path[1:])):
            feasible = True
            break
    if not feasible:
        raise ValueError("S1 has no verified detour corridor")


def s1_validation_bank(count: int = 50, *, seed: int = 3027) -> tuple[Scenario, ...]:
    if count < 50:
        raise ValueError("S1 validation bank must contain at least 50 cases")
    rng = np.random.default_rng(seed)
    cases = []
    hashes = set()
    while len(cases) < count:
        case = sample_s1(rng, split="S1-validation", layout=len(cases) % 5)
        digest = case.geometry_hash()
        if digest not in hashes:
            hashes.add(digest)
            cases.append(case)
    return tuple(cases)


def s1_dev_bank() -> tuple[Scenario, ...]:
    """The previously inspected 50-case bank, now explicitly development data."""
    return tuple(Scenario("S1-dev", c.seed, c.starts, c.goals, c.obstacles)
                 for c in s1_validation_bank())


@lru_cache(maxsize=1)
def s1_test_bank() -> tuple[Scenario, ...]:
    cases = s1_validation_bank(100, seed=4027)
    test = tuple(Scenario("S1-test", c.seed, c.starts, c.goals, c.obstacles)
                 for c in cases)
    assert not ({c.geometry_hash() for c in test} &
                {c.geometry_hash() for c in s1_dev_bank()})
    return test


@lru_cache(maxsize=1)
def s1_final_test_bank() -> tuple[Scenario, ...]:
    """Fresh paper-candidate test geometries; never a checkpoint-selection bank."""
    cases = s1_validation_bank(100, seed=5027)
    test = tuple(Scenario("S1-final-test", c.seed, c.starts, c.goals, c.obstacles)
                 for c in cases)
    previous = {c.geometry_hash() for c in (*s1_dev_bank(), *s1_test_bank())}
    assert not ({c.geometry_hash() for c in test} & previous)
    return test


@lru_cache(maxsize=1)
def _reserved_s1_hashes() -> frozenset[str]:
    return frozenset(c.geometry_hash() for c in
                     (*s1_dev_bank(), *s1_test_bank(), *s1_final_test_bank()))


def sample_s1_train(rng: np.random.Generator) -> Scenario:
    """Same S1 distribution, rejecting any exact dev/test geometry overlap."""
    reserved = _reserved_s1_hashes()
    while True:
        case = sample_s1(rng)
        if case.geometry_hash() not in reserved:
            return case
