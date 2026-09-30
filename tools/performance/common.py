"""Shared fixed Stage5C-light workload helpers for CPU performance experiments."""
from __future__ import annotations

import os

# Must be set before importing torch/numpy in child processes.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/validation"))
sys.path.insert(0, str(ROOT / "tools"))

SEED = 53
VESSELS = 4
DT_S = 0.1


def names_for(count):
    return tuple(f"agent_{i:03d}" for i in range(count))


def make_env():
    from test_stage5a_marl import command, make_world
    names = names_for(VESSELS)
    # Exact `light` branch of stage5c_scaling_validation.make_case: uniform zero
    # current/wind, calm waves, no runtime loads, no sensors or obstacles.
    sensors = {name: ("none", 10, 0) for name in names}
    env = make_world(names, max_steps=10**9, goal=(1e9, 0., 0.),
        sensor_by_name=sensors,
        spawn_by_name={name: (20. * i, 0., 0.) for i, name in enumerate(names)},
        mass_by_name={name: 10 + (i % 4) * 2 for i, name in enumerate(names)},
        environment_config={"current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "waves": {"kind": "calm"}, "visibility_m": 1000},
        environment_loads_by_name={}, static_entities=(), bottom=2.)
    env.reset(seed=SEED)
    actions = {name: command(0) for name in names_for(VESSELS)}
    return env, actions
