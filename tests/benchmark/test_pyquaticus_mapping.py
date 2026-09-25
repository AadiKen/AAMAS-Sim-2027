"""Quantified native Heron control and state-to-sensor mapping checks."""
import math
import os

import pytest

from bcod_sim.benchmark.core import BenchmarkConfig, Obstacle, Scenario, VesselPose
from bcod_sim.benchmark.pyquaticus_adapter import PyquaticusAdapter


@pytest.fixture
def py_env():
    executable = os.environ.get("PYQUATICUS_BENCHMARK_PYTHON")
    if not executable:
        pytest.skip("isolated Pyquaticus interpreter unavailable")
    env = PyquaticusAdapter(executable, BenchmarkConfig(max_steps=100))
    yield env
    env.close()


def _scenario(obstacles=None):
    if obstacles is None:
        obstacles = (Obstacle(-35., -35., 1.5), Obstacle(35., 35., 1.5))
    ys = (-30., -10., 10., 30.)
    return Scenario("nominal", 123, tuple(VesselPose(-15., y, 0.) for y in ys),
                    tuple((15., y) for y in ys), tuple(obstacles))


def test_pyquaticus_sensor_range_and_frame(py_env):
    s = _scenario((Obstacle(14., -30., 1.), Obstacle(16., -30., 1.)))
    readings, truth = py_env.reset(s, 123)
    first = readings["vessel_0"]
    assert (first.x_m, first.y_m, first.heading_rad) == pytest.approx((-15., -30., 0.))
    assert first.nearby_obstacles == [[14., -30., 1.]]  # 29 m seen; 31 m hidden
    assert (first.surge_mps, first.yaw_rps) == (0., 0.)
    assert len(first.nearby_agents) == 1  # 20 m agent visible; 40/60 m hidden
    assert truth["vessel_0"].x_m == first.x_m


@pytest.mark.parametrize("yaw,sign", [(0.5, 1), (-0.5, -1)])
def test_pyquaticus_heron_yaw_rate_mapping(py_env, yaw, sign):
    py_env.reset(_scenario(), 123)
    rates = []
    for _ in range(20):
        readings, _ = py_env.step({f"vessel_{i}": (0.8, yaw) for i in range(4)})
        rates.append(readings["vessel_0"].yaw_rps)
    steady = sum(rates[-5:]) / 5
    assert steady * sign == pytest.approx(0.25, abs=0.06)
    assert all(math.isfinite(rate) for rate in rates)


def test_pyquaticus_heron_surge_command_response(py_env):
    observed = []
    for throttle in (0.5, 1.0):
        py_env.reset(_scenario(), 123)
        speeds = []
        for _ in range(20):
            readings, _ = py_env.step({f"vessel_{i}": (throttle, 0.) for i in range(4)})
            speeds.append(readings["vessel_0"].surge_mps)
        observed.append(sum(speeds[-5:]) / 5)
        assert all(b >= a for a, b in zip(speeds, speeds[1:]))
    assert observed[0] == pytest.approx(0.356, abs=0.04)
    assert observed[1] == pytest.approx(0.711, abs=0.06)
    assert observed[1] > observed[0] * 1.8
