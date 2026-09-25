"""Real simulator qualification tests; Pyquaticus needs its isolated interpreter."""

import math
import json
import os
import random
from dataclasses import replace

import numpy as np
import pytest

from bcod_sim.benchmark.bcod_adapter import BCODAdapter
from bcod_sim.benchmark.core import (Benchmark, BenchmarkConfig, NAMES, Obstacle,
                                     Scenario, VesselPose, generate_scenario)
from bcod_sim.benchmark.pyquaticus_adapter import PyquaticusAdapter


def _adapter(kind, config):
    if kind == "bcod":
        return BCODAdapter(config)
    executable = os.environ.get("PYQUATICUS_BENCHMARK_PYTHON")
    if not executable:
        pytest.skip("Set PYQUATICUS_BENCHMARK_PYTHON to the isolated Python 3.10 interpreter")
    return PyquaticusAdapter(executable, config)


def _easy_scenario():
    rows = (-30., -10., 10., 30.)
    return Scenario("nominal", 123, tuple(VesselPose(-15., y, 0.) for y in rows),
                    tuple((15., y) for y in rows),
                    tuple(Obstacle(x, y, 1.5) for x, y in
                          ((-35., -35.), (-35., 0.), (-35., 35.),
                           (35., -35.), (35., 0.), (35., 35.))))


@pytest.mark.parametrize("kind", ("bcod", "pyquaticus"))
def test_adapter_seeded_reset_and_short_soak(kind):
    config = BenchmarkConfig(max_steps=40)
    env = Benchmark(_adapter(kind, config), config)
    rng = random.Random(93)
    try:
        scenario = generate_scenario(9)
        first = env.reset(scenario)
        initial = dict(env.truth)
        second = env.reset(scenario)
        assert initial == env.truth
        assert all(np.array_equal(first[n], second[n]) for n in NAMES)
        for _ in range(40):
            actions = {name: (rng.uniform(0, 1), rng.uniform(-1, 1)) for name in NAMES}
            observations, rewards, done, truncated, info = env.step(actions)
            assert all(v.shape == (47,) and np.isfinite(v).all() for v in observations.values())
            assert all(math.isfinite(v) for v in rewards.values())
            if done or truncated:
                assert set(info["per_agent_success"]) == set(NAMES)
                break
        assert info["steps"] > 0
    finally:
        env.close()


@pytest.mark.parametrize("kind", ("bcod", "pyquaticus"))
def test_simple_scripted_navigation(kind):
    config = BenchmarkConfig(max_steps=250)
    env = Benchmark(_adapter(kind, config), config)
    try:
        env.reset(_easy_scenario())
        for _ in range(config.max_steps):
            _, _, done, truncated, info = env.step({name: (0.7, 0.) for name in NAMES})
            if done or truncated:
                break
        assert info["fleet_success"] and info["collision_count"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("kind", ("bcod", "pyquaticus"))
def test_real_collision_and_timeout_metrics(kind):
    config = BenchmarkConfig(max_steps=2)
    env = Benchmark(_adapter(kind, config), config)
    try:
        scenario = generate_scenario(5)
        starts = (scenario.starts[0],
                  VesselPose(scenario.starts[0].x_m + 1.5, scenario.starts[0].y_m,
                             scenario.starts[0].heading_rad), *scenario.starts[2:])
        env.reset(replace(scenario, starts=starts))
        _, reward, done, truncated, info = env.step({n: (0., 0.) for n in NAMES})
        assert done and not truncated and info["collision_count"] >= 2
        assert reward[NAMES[0]] < -0.8 * config.collision_penalty
        json.dumps(info)
        env.reset(scenario)
        env.step({n: (0., 0.) for n in NAMES})
        _, _, done, truncated, info = env.step({n: (0., 0.) for n in NAMES})
        assert truncated and not done and info["steps"] == 2
    finally:
        env.close()


@pytest.mark.parametrize("kind", ("bcod", "pyquaticus"))
def test_positive_yaw_action_turns_counterclockwise(kind):
    config = BenchmarkConfig(max_steps=20)
    env = Benchmark(_adapter(kind, config), config)
    try:
        env.reset(generate_scenario(33))
        initial = env.truth[NAMES[0]].heading_rad
        for _ in range(5):
            env.step({name: (0.8, 1.) for name in NAMES})
        final = env.truth[NAMES[0]].heading_rad
        delta = math.atan2(math.sin(final - initial), math.cos(final - initial))
        assert delta > 0
    finally:
        env.close()


def test_bcod_uses_pinned_otter_and_sensor_packets():
    env = Benchmark(BCODAdapter())
    try:
        env.reset(generate_scenario(51))
        engine = env.adapter.engine
        assert all(v.plant.mass.mass_kg == 55.0 for v in engine.vessels.values())
        assert all(v.plant.crossflow.model_name == "strip_theory" for v in engine.vessels.values())
        assert all({sensor.config.instance_id for sensor in v.sensors} == {"gps", "imu", "entities"}
                   for v in engine.vessels.values())
        assert all((i, "entities") in engine.latest_packets for i in range(1, 5))
    finally:
        env.close()
