import math

import numpy as np
import pytest

from bcod_sim.benchmark_v3.deadline_harness.marl import (
    ACTION_SPACE, OBS_FIELDS, GeometricCoordinator, ResidualCoordinatorEnv,
    dcpa_tcpa, make_four_vessel_scenario,
)
from bcod_sim.benchmark_v3.deadline_harness.sarl import (
    SARLTrackingEnv, common_differential_to_thrusters, tracking_observation,
)
from bcod_sim.benchmark_v2.backend import VesselReading
from bcod_sim.benchmark_v3.deadline_harness.algorithms import MAPPO, TD3Actor, TwinCritic
from bcod_sim.benchmark_v3.deadline_harness.evaluation import evaluate_sarl
import torch


def test_residual_action_contract_and_body_relative_observation():
    scenario = make_four_vessel_scenario(np.random.default_rng(2), "easy")
    from bcod_sim.benchmark_v3.config import TaskConfig
    env = ResidualCoordinatorEnv(config=TaskConfig(agent_count=4, dynamics="kinematic",
        action_mode="high_level", dt_s=.5), scenario=scenario)
    try:
        obs, _ = env.reset(seed=scenario.seed)
        assert all(value.shape == (len(OBS_FIELDS),) for value in obs.values())
        assert env.action_space("vessel_0").shape == (2,)
        assert np.array_equal(env.action_space("vessel_0").low, ACTION_SPACE.low)
        assert np.isfinite(env.state()).all()
        teacher = GeometricCoordinator().actions(env)
        _, rewards, _, _, info = env.step(teacher)
        assert len(set(rewards.values())) == 1
        assert all("shared_fleet_reward_components" in item for item in info.values())
    finally:
        env.close()


def test_dcpa_tcpa_closing_and_parallel_geometry():
    distance, time = dcpa_tcpa(10., 0., -2., 0.)
    assert distance == pytest.approx(0.)
    assert time == pytest.approx(5.)
    distance, time = dcpa_tcpa(10., 0., 1., 0.)
    assert distance == pytest.approx(10.) and time == 0.


def test_common_differential_mapping_and_heading_wrap_observation():
    assert common_differential_to_thrusters(.5, .25) == pytest.approx((.25, .75))
    assert common_differential_to_thrusters(.9, .5) == pytest.approx((.4, 1.))
    reading = VesselReading(0., 0., -math.pi+.01, 0., 0.)
    obs = tracking_observation(desired_speed=0., desired_heading=math.pi-.01,
        reading=reading, speed_integral=9., heading_integral=-9.,
        previous_action=(0., 0.))
    assert obs.shape == (10,) and obs[1] == pytest.approx(math.sin(-.02), abs=1e-5)
    assert obs[6] == 1. and obs[7] == -1.


def test_sarl_direct_actuator_short_episode_smoke():
    env = SARLTrackingEnv(episode_steps=2)
    try:
        obs, _ = env.reset(seed=3, command_family="heading")
        assert obs.shape == (10,) and np.isfinite(obs).all()
        obs, reward, terminated, truncated, info = env.step(np.array([0., .5], np.float32))
        assert not terminated and not truncated and math.isfinite(reward)
        assert env.reading.yaw_rps > 0.  # Positive differential thrust turns CCW.
        assert "reward_components" in info and obs.shape == (10,)
    finally:
        env.close()


def test_shared_actor_central_critic_and_td3_shapes():
    marl = MAPPO()
    action, logp, _ = marl.act(__import__("torch").zeros((4, 36)))
    assert action.shape == (4, 2) and logp.shape == (4,)
    assert torch.all((action[:, 0] >= 0) & (action[:, 0] <= 1))
    assert torch.all((action[:, 1] >= -1) & (action[:, 1] <= 1))
    assert marl.value(torch.zeros((1, 89))).shape == (1,)
    actor = TD3Actor(); critic = TwinCritic()
    action = actor(torch.zeros((3, 10)))
    assert action.shape == (3, 2) and torch.all(action.abs() <= 1)
    q1, q2 = critic(torch.zeros((3, 10)), action)
    assert q1.shape == q2.shape == (3, 1)


def test_observation_normalizer_roundtrips_in_policy_state():
    policy=TD3Actor(); policy.set_observation_normalizer(np.arange(40,dtype=np.float32).reshape(4,10))
    restored=TD3Actor(); restored.load_state_dict(policy.state_dict())
    torch.testing.assert_close(policy.obs_mean,restored.obs_mean)
    torch.testing.assert_close(policy.obs_std,restored.obs_std)


def test_paired_sarl_evaluation_exports_all_families(tmp_path):
    actor=TD3Actor()
    checkpoint=tmp_path/"policy.pt"
    torch.save({"actor":actor.state_dict()},checkpoint)
    result=evaluate_sarl(checkpoint,tmp_path/"eval",episodes=6,steps=2,seed=101)
    assert result["paired_identical_seeds"] and result["episodes"]==6
    assert (tmp_path/"eval"/"per_episode.csv").is_file()
    assert (tmp_path/"eval"/"summary.json").is_file()
    assert len(list((tmp_path/"eval"/"timeseries").glob("episode-*.csv")))==6
