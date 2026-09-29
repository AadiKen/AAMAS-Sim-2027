import pytest

pytest.importorskip("benchmarl")
pytest.importorskip("torchrl")

from torchrl.envs.utils import check_env_specs

from bcod_sim.benchmark_v3.centralized_state import STATE_SCHEMA_HASH
from bcod_sim.benchmark_v3.marl.scenarios import (bank_hash, m0_dev_bank,
                                                   m0_test_bank, sample_m0_train)
from bcod_sim.benchmark_v3.marl.task import V3M0Task
from bcod_sim.benchmark_v2.scenarios import Scenario, VesselPose
from bcod_sim.benchmark_v3.config import TaskConfig
from bcod_sim.benchmark_v3.parallel_env import NavigationParallelEnv
import numpy as np


def test_m0_banks_are_unique_disjoint_and_two_vessel_obstacle_free():
    dev, test = m0_dev_bank(), m0_test_bank()
    assert len(dev) == 50 and len(test) == 100
    assert bank_hash(dev) == "d73df5d79047a5ad369462597343b58567ea6ceffb5ac4b07d8a459a0004af31"
    assert bank_hash(test) == "9d96a60bf9985c569e5414efbf7b3b254a8d6b02ba8701cfcbc61cd090e5819e"
    reserved = {c.geometry_hash() for c in (*dev, *test)}
    assert len(reserved) == 150
    assert all(len(c.agent_ids) == 2 and not c.obstacles for c in (*dev, *test))
    import numpy as np
    rng = np.random.default_rng(99)
    assert all(sample_m0_train(rng).geometry_hash() not in reserved for _ in range(50))


def test_benchmarl_task_uses_local_actor_and_global_critic_specs():
    task = V3M0Task(deadline_steps=40)
    env = task.get_env_fun(1, True, 11, "cpu")()
    try:
        check_env_specs(env)
        actor = task.observation_spec(env)
        critic = task.state_spec(env)
        assert actor["agents", "observation"].shape == (2, 78)
        assert critic["state"].shape == (89,)
        assert "state" not in actor.keys()
        assert task.group_map(env) == {"agents": ["vessel_0", "vessel_1"]}
        assert STATE_SCHEMA_HASH
    finally:
        env.close()


def test_action_alignment_and_fleet_collision_reward_termination():
    config = TaskConfig(agent_count=2, dynamics="kinematic", action_mode="high_level")
    safe = Scenario("alignment", 3,
                    (VesselPose(-12., -10., 0.), VesselPose(12., 10., np.pi)),
                    ((12., -10.), (-12., 10.)))
    env = NavigationParallelEnv(config, scenario=safe)
    try:
        env.reset()
        before = {k: env._frame.truth[k] for k in env.agents}
        env.step({"vessel_0": np.array([1., 0.], dtype=np.float32),
                  "vessel_1": np.array([-1., 0.], dtype=np.float32)})
        assert env._frame.truth["vessel_0"].x_m > before["vessel_0"].x_m
        assert env._frame.truth["vessel_1"].x_m == before["vessel_1"].x_m
    finally:
        env.close()


def test_reached_vessel_policy_action_is_held_out_of_control():
    config = TaskConfig(agent_count=2, dynamics="kinematic", action_mode="high_level")
    scenario = Scenario("reached-hold", 5,
                        (VesselPose(-12., -8., 0.), VesselPose(12., 8., np.pi)),
                        ((12., -8.), (-12., 8.)))
    env = NavigationParallelEnv(config, scenario=scenario)
    try:
        env.reset()
        env.task.reached["vessel_0"] = True
        before = env._frame.truth["vessel_0"]
        _, _, _, _, infos = env.step({name: np.array([1., 1.], dtype=np.float32)
                                      for name in env.agents})
        after = env._frame.truth["vessel_0"]
        assert (after.x_m, after.y_m) == (before.x_m, before.y_m)
        assert infos["vessel_0"]["physical_command"] == (0., 0.)
        assert infos["vessel_0"]["policy_action_ignored_after_goal"]
        assert not infos["vessel_1"]["policy_action_ignored_after_goal"]
    finally:
        env.close()


def test_real_goal_reach_persists_while_other_vessel_continues():
    config = TaskConfig(agent_count=2, dynamics="kinematic", action_mode="high_level")
    scenario = Scenario("asymmetric-arrival", 6,
                        (VesselPose(-15., -10., 0.), VesselPose(-15., 10., 0.)),
                        ((-5., -10.), (20., 10.)))
    env = NavigationParallelEnv(config, scenario=scenario)
    try:
        env.reset()
        actions = {name: np.array([1., 0.], dtype=np.float32) for name in env.agents}
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(actions)
            if env.task.reached["vessel_0"]:
                break
            assert not any(terminated.values()) and not any(truncated.values())
        assert env.task.reached["vessel_0"] and not env.task.reached["vessel_1"]
        before = env._frame.truth["vessel_0"]
        _, _, _, _, infos = env.step(actions)
        after = env._frame.truth["vessel_0"]
        assert env.task.reached["vessel_0"]
        assert (after.x_m, after.y_m) == (before.x_m, before.y_m)
        assert infos["vessel_0"]["policy_action_ignored_after_goal"]
    finally:
        env.close()

    collision = Scenario("collision", 4,
                         (VesselPose(-1.6, 0., 0.), VesselPose(1.6, 0., np.pi)),
                         ((15., 0.), (-15., 0.)))
    env = NavigationParallelEnv(config, scenario=collision)
    try:
        env.reset()
        action = {name: np.array([1., 0.], dtype=np.float32) for name in env.agents}
        for _ in range(5):
            _, rewards, terminated, truncated, infos = env.step(action)
            if all(terminated.values()):
                break
        assert all(terminated.values()) and not any(truncated.values())
        assert not env.agents
        assert all(infos[name]["terminal_reason"] == "collision" for name in action)
        assert all(infos[name]["reward_components"]["collision"] == -50.
                   for name in action)
        assert all(np.isfinite(list(rewards.values())))
    finally:
        env.close()
