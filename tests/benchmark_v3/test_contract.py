import math
import numpy as np
import pytest

from bcod_sim.benchmark_v2.backend import VesselReading
from bcod_sim.benchmark_v2.scenarios import Scenario, VesselPose
from bcod_sim.benchmark_v3 import ACTION_VERSION, OBSERVATION_VERSION, TASK_VERSION
from bcod_sim.benchmark_v3.adapters.kinematic import KinematicBackend
from bcod_sim.benchmark_v3.config import TaskConfig
from bcod_sim.benchmark_v3.gym_env import NavigationGymEnv, action_to_physical
from bcod_sim.benchmark_v3.observations import FIELDS, SCHEMA_HASH, build_observation
from bcod_sim.benchmark_v3.scenarios import case_a_reference, s0_validation_bank


def test_versions_action_boundaries_and_sign():
    cfg = TaskConfig()
    assert (TASK_VERSION, OBSERVATION_VERSION, ACTION_VERSION) == (
        "navigation-v3", "ego-relative-v2", "desired-speed-yawrate-v2")
    assert action_to_physical([-1, -1], cfg) == (0., -.25)
    assert action_to_physical([0, 0], cfg) == (1., 0.)
    assert action_to_physical([1, 1], cfg) == (2., .25)
    with pytest.raises(ValueError):
        action_to_physical([1.01, 0], cfg)
    with pytest.raises(ValueError):
        action_to_physical([np.nan, 0], cfg)


def test_kinematic_turn_golden_and_measured_frame():
    cfg = TaskConfig()
    scenario = Scenario("golden", 11, (VesselPose(0., 0., 0.),), ((20., 0.),))
    backend = KinematicBackend(cfg)
    backend.reset(scenario, 11)
    frame = backend.step({"vessel_0": (1., .25)}, .2)
    truth = frame.truth["vessel_0"]
    assert truth.heading_rad == pytest.approx(.05)
    assert truth.x_m == pytest.approx(.2 * math.cos(.05))
    assert truth.y_m == pytest.approx(.2 * math.sin(.05))
    assert frame.readings["vessel_0"].x_m == truth.x_m
    backend.close()
    backend.close()


def test_s0_bank_has_turns_and_unique_geometries():
    bank = s0_validation_bank()
    assert len(bank) == 50
    assert len({s.geometry_hash() for s in bank}) == 50
    assert all(s.split == "S0-validation" and not s.obstacles for s in bank)
    for scenario in bank:
        start, goal = scenario.starts[0], scenario.goals[0]
        distance = math.dist((start.x_m, start.y_m), goal)
        angle = math.atan2(goal[1] - start.y_m, goal[0] - start.x_m) - start.heading_rad
        turn = abs(math.atan2(math.sin(angle), math.cos(angle)))
        assert 18 <= distance <= 38
        assert .45 - 1e-12 <= turn <= 2.45 + 1e-12


def test_relative_velocity_distinguishes_approach_and_recede():
    cfg = TaskConfig(agent_count=2)
    ego = VesselReading(0., 0., 0., 0., 0.)
    other = VesselReading(5., 0., 0., 0., 0.)
    now = {"vessel_0": ego, "vessel_1": other}
    approaching = {"vessel_0": ego, "vessel_1": VesselReading(5.2, 0., 0., 0., 0.)}
    receding = {"vessel_0": ego, "vessel_1": VesselReading(4.8, 0., 0., 0., 0.)}
    a = build_observation("vessel_0", ego, (20., 0.), now, approaching, 1, cfg)
    b = build_observation("vessel_0", ego, (20., 0.), now, receding, 1, cfg)
    assert a.shape == b.shape == (78,)
    np.testing.assert_array_equal(a[:12], b[:12])
    assert a[12] < 0 < b[12]
    assert a[14] == b[14] == 1
    assert len(FIELDS) == 78 and len(SCHEMA_HASH) == 64


def test_env_seed_schema_and_case_a_control():
    env = NavigationGymEnv(TaskConfig())
    try:
        a, ia = env.reset(seed=123)
        b, ib = env.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        assert ia["scenario_hash"] == ib["scenario_hash"]
        assert a.dtype == np.float32 and env.observation_space.contains(a)
        assert a[-1] == 1
        env.reset(seed=11, options={"scenario": case_a_reference(11)})
        for step in range(600):
            obs, reward, terminated, truncated, info = env.step(np.array([0., 0.], dtype=np.float32))
            assert truncated is False
            if terminated:
                break
        assert info["terminal_reason"] == "success" and step + 1 == 191
        assert obs[-1] == pytest.approx((600 - 191) / 600)
    finally:
        env.close()
        env.close()


def test_discounted_shaping_identity_and_trajectory_order():
    cfg = TaskConfig()
    gamma = cfg.gamma
    # Fully specified terminal trajectories with Phi(terminal)=0.
    def score(length, terminal_bonus):
        phi = [-40.] + [-40. + 40. * i / length for i in range(1, length)] + [0.]
        shaping = [gamma * phi[i + 1] - phi[i] for i in range(length)]
        discounted_shape = sum(gamma**i * reward for i, reward in enumerate(shaping))
        assert discounted_shape == pytest.approx(-phi[0])
        return discounted_shape + gamma**(length - 1) * terminal_bonus \
            - cfg.step_penalty * sum(gamma**i for i in range(length))
    assert score(190, 20.) > score(600, 0.) > score(300, -50.)


def test_deadline_is_task_termination_not_truncation():
    cfg = TaskConfig(deadline_steps=2)
    env = NavigationGymEnv(cfg, scenario=case_a_reference(11))
    try:
        env.reset(seed=11)
        first = env.step([-1., 0.])
        second = env.step([-1., 0.])
        assert first[2:4] == (False, False)
        assert second[2:4] == (True, False)
        assert second[4]["terminal_reason"] == "deadline"
        assert second[0][-1] == 0.
        assert second[4]["episode_returns"]["vessel_0"] == pytest.approx(first[1] + second[1])
        assert sum(second[4]["episode_reward_components"]["vessel_0"].values()) \
            == pytest.approx(second[4]["episode_returns"]["vessel_0"])
    finally:
        env.close()


def test_fleet_collision_penalty_preserves_physical_collider():
    from bcod_sim.benchmark_v3.task import NavigationTask

    cfg = TaskConfig(agent_count=4)
    starts = (VesselPose(48.8, 0., 0.), VesselPose(-20., -20., 0.),
              VesselPose(-20., 20., 0.), VesselPose(20., 20., 0.))
    goals = ((0., 0.), (20., -20.), (20., 20.), (-20., 20.))
    scenario = Scenario("fleet-test", 5, starts, goals)
    backend = KinematicBackend(cfg)
    task = NavigationTask(cfg)
    task.reset(scenario, backend.reset(scenario, 5))
    commands = {f"vessel_{i}": (2., 0.) if i == 0 else (0., 0.) for i in range(4)}
    _, rewards, terminated, truncated, info = task.step(backend.step(commands, .2), commands)
    assert terminated and not truncated
    assert info["physical_colliders"] == ["vessel_0"]
    assert all(info["reward_components"][name]["collision"] == -50. for name in commands)
    assert len(rewards) == 4
    backend.close()


def test_observation_only_s0_scripted_controller_solves_bank():
    from bcod_sim.benchmark_v3.evaluate import evaluate_bank
    from bcod_sim.benchmark_v3.scripted import GeometricController

    report = evaluate_bank(GeometricController(), TaskConfig(), s0_validation_bank(),
                           checkpoint_step=0)
    assert report["success_count"] == 50
    assert report["collision_count"] == report["deadline_count"] == 0


def test_evaluation_tracking_and_trajectory_fields(tmp_path):
    import json
    from bcod_sim.benchmark_v3.evaluate import evaluate_bank
    from bcod_sim.benchmark_v3.scripted import GeometricController

    report = evaluate_bank(GeometricController(), TaskConfig(), [case_a_reference(11)],
                           checkpoint_step=0, trajectory_dir=tmp_path)
    assert report["mean_commanded_speed_mps"] > 0
    assert report["mean_achieved_surge_mps"] > 0
    assert report["completion_steps_median"] is not None
    trace = json.loads((tmp_path / "step-0-case-0.json").read_text())
    row = trace["trajectory"][0]
    for field in ("time_s", "truth", "reading", "physical_6dof", "actuator_saturated",
                  "termination_reason", "goal_bearing_rad", "commanded_speed_mps",
                  "achieved_surge_mps", "commanded_yaw_rate_radps", "achieved_yaw_rate_radps",
                  "distance_to_goal_m"):
        assert field in row
