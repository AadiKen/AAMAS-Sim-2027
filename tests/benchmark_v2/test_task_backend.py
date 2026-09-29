import math
import numpy as np
import pytest

from bcod_sim.benchmark_v2.backend import BackendFrame, Truth, VesselReading
from bcod_sim.benchmark_v2.config import TaskConfig
from bcod_sim.benchmark_v2.gym_env import NavigationGymEnv, action_to_physical
from bcod_sim.benchmark_v2.scenarios import Scenario, VesselPose, case_a_reference, case_a_validation
from bcod_sim.benchmark_v2.task import NavigationTask


def frame(x, y=0.):
    return BackendFrame({"vessel_0": VesselReading(x, y, 0., 0., 0.)},
                        {"vessel_0": Truth(x, y, 0.)})


def scenario():
    return Scenario("test", 1, (VesselPose(0., 0., 0.),), ((10., 0.),))


def test_action_map_once_and_bounds():
    cfg = TaskConfig()
    assert action_to_physical([-1., -1.], cfg) == (0., -.25)
    assert action_to_physical([0., 0.], cfg) == (1., 0.)
    assert action_to_physical([1., 1.], cfg) == (2., .25)
    with pytest.raises(ValueError):
        action_to_physical([-1.01, 0.], cfg)


def test_observation_horizon_and_goal_lifecycle():
    task = NavigationTask(TaskConfig(deadline_steps=2))
    obs = task.reset(scenario(), frame(0.))["vessel_0"]
    assert obs.shape == (48,) and obs.dtype == np.float32 and obs[-1] == 1.
    obs, reward, term, trunc, info = task.step(frame(1.))
    assert not term and not trunc and obs["vessel_0"][-1] == .5
    assert reward["vessel_0"] == pytest.approx(sum(info["reward_components"]["vessel_0"].values()))
    _, _, term, trunc, info = task.step(frame(9.))
    assert term and not trunc and info["reason"] == "success"  # final permitted step
    assert info["reward_components"]["vessel_0"]["goal"] == 20.


def test_deadline_terminal_potential_and_shaping_identity():
    cfg = TaskConfig(deadline_steps=2)
    task = NavigationTask(cfg)
    task.reset(scenario(), frame(0.))
    _, _, term, _, i1 = task.step(frame(1.))
    _, _, term, trunc, i2 = task.step(frame(2.))
    assert term and not trunc and i2["reason"] == "deadline"
    f1 = i1["reward_components"]["vessel_0"]["potential_shaping"]
    f2 = i2["reward_components"]["vessel_0"]["potential_shaping"]
    assert f1 + cfg.gamma * f2 == pytest.approx(10.)


def test_collision_precedes_goal_and_native_contact():
    task = NavigationTask(TaskConfig())
    task.reset(scenario(), frame(0.))
    _, _, term, _, info = task.step(BackendFrame(frame(9.).readings, frame(9.).truth,
                                                 {"native_contacts": ["vessel_0 contact"]}))
    assert term and info["reason"] == "collision"
    assert info["reward_components"]["vessel_0"]["collision"] == -50.
    assert info["reward_components"]["vessel_0"]["goal"] == 0.


def test_scenario_geometry_distinct():
    bank = case_a_validation()
    assert len(bank) == len({s.geometry_hash() for s in bank}) == 10
    assert len({s.geometry_hash() for s in (case_a_reference(1), case_a_reference(2))}) == 1


def test_exactly_one_physical_vessel_and_repeated_close():
    env = NavigationGymEnv()
    obs, info = env.reset(options={"scenario": case_a_reference()})
    assert env.backend.engine is not None
    assert set(env.backend.engine.vessels) == {"vessel_0"}
    assert set(env.backend.engine.states) == {"vessel_0"}
    assert obs.shape == (48,)
    obs, reward, term, trunc, info = env.step(np.array([0., 0.], np.float32))
    assert info["physical_command"] == {"speed_mps": 1., "yaw_rate_radps": 0.}
    env.close()
    env.close()


def test_fixed_command_matches_qualified_bcod_adapter():
    from bcod_sim.benchmark.bcod_adapter import BCODAdapter
    from bcod_sim.benchmark.core import BenchmarkConfig as LegacyConfig
    from bcod_sim.benchmark.core import Scenario as LegacyScenario
    from bcod_sim.benchmark.core import VesselPose as LegacyPose
    from bcod_sim.benchmark_v2.adapters.bcod import BCODBackend

    seed = 17
    starts = (VesselPose(-20., 0., 0.), VesselPose(30., 30., 0.),
              VesselPose(-30., 30., 0.), VesselPose(30., -30., 0.))
    legacy = BCODAdapter(LegacyConfig(), reduced_fidelity=True)
    modern = BCODBackend(TaskConfig())
    old_case = LegacyScenario("test", seed, tuple(LegacyPose(p.x_m, p.y_m, p.heading_rad) for p in starts),
                              ((20., 0.), (0., 0.), (0., 0.), (0., 0.)), ())
    new_case = Scenario("test", seed, (starts[0],), ((20., 0.),))
    try:
        _, old_truth = legacy.reset(old_case, seed)
        new_frame = modern.reset(new_case, seed)
        assert new_frame.truth["vessel_0"].x_m == pytest.approx(old_truth["vessel_0"].x_m)
        for _ in range(3):
            _, old_truth = legacy.step({"vessel_0": (.5, .2), "vessel_1": (0., 0.),
                                        "vessel_2": (0., 0.), "vessel_3": (0., 0.)})
            new_frame = modern.step({"vessel_0": (1., .05)}, .2)
            assert new_frame.truth["vessel_0"].x_m == pytest.approx(old_truth["vessel_0"].x_m, abs=1e-8)
            assert new_frame.truth["vessel_0"].y_m == pytest.approx(old_truth["vessel_0"].y_m, abs=1e-8)
            assert new_frame.truth["vessel_0"].heading_rad == pytest.approx(old_truth["vessel_0"].heading_rad, abs=1e-8)
    finally:
        modern.close()
        legacy.close()


def test_four_agent_collision_is_fleet_penalty():
    names = tuple(f"vessel_{i}" for i in range(4))
    starts = tuple(VesselPose(-20. + 10 * i, -20., 0.) for i in range(4))
    goals = tuple((20., -20. + 4 * i) for i in range(4))
    case = Scenario("fleet", 3, starts, goals)
    def fleet_frame(x0):
        truth = {name: Truth(x0 if i == 0 else starts[i].x_m, starts[i].y_m, 0.)
                 for i, name in enumerate(names)}
        readings = {name: VesselReading(t.x_m, t.y_m, 0., 0., 0.) for name, t in truth.items()}
        return BackendFrame(readings, truth)
    task = NavigationTask(TaskConfig(agent_count=4))
    task.reset(case, fleet_frame(-20.))
    _, rewards, term, trunc, info = task.step(fleet_frame(-50.))
    assert term and not trunc and info["reason"] == "collision"
    assert info["physical_colliders"] == ["vessel_0"]
    assert all(info["reward_components"][name]["collision"] == -50. for name in names)
