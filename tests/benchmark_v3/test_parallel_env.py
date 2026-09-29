import numpy as np
import pytest
pytest.importorskip("pettingzoo")
from dataclasses import replace
from pettingzoo.test import parallel_api_test, parallel_seed_test

from bcod_sim.benchmark_v2.scenarios import Scenario, VesselPose
from bcod_sim.benchmark_v3.centralized_state import (CENTRALIZED_STATE_VERSION,
                                                      FIELDS, STATE_SPACE,
                                                      encode_centralized_state)
from bcod_sim.benchmark_v3.config import TaskConfig
from bcod_sim.benchmark_v3.parallel_env import NavigationParallelEnv


def make_env():
    scenario = Scenario("M0-contract", 7,
                        (VesselPose(-16., -4., 0.),
                         VesselPose(4., -16., np.pi / 2)),
                        ((16., -4.), (4., 16.)))
    config = TaskConfig(agent_count=2, dynamics="kinematic", action_mode="high_level",
                        deadline_steps=40)
    return NavigationParallelEnv(config, scenario=scenario)


def test_parallel_api_and_seed_contract():
    parallel_api_test(make_env(), num_cycles=80)
    parallel_seed_test(make_env, num_cycles=80)


def test_centralized_state_is_versioned_training_only_and_tracks_motion():
    env = make_env()
    try:
        observations, infos = env.reset(seed=7)
        assert env.centralized_state_schema["version"] == CENTRALIZED_STATE_VERSION
        assert tuple(observations) == tuple(env.possible_agents)
        assert all(obs.shape == (78,) for obs in observations.values())
        assert all("state" not in info for info in infos.values())
        initial = env.state()
        assert len(FIELDS) == len(initial) == 89 and STATE_SPACE.contains(initial)
        assert initial[FIELDS.index("agent_2_present")] == 0.
        assert initial[FIELDS.index("agent_0_present")] == 1.
        actions = {agent: np.array([0., 0.], dtype=np.float32) for agent in env.agents}
        next_obs, _, _, _, next_info = env.step(actions)
        advanced = env.state()
        assert STATE_SPACE.contains(advanced)
        assert not np.array_equal(initial, advanced)
        assert advanced[FIELDS.index("agent_0_velocity_x")] != 0.
        reduced = replace(env.config, dynamics="bcod-reduced")
        full = replace(env.config, dynamics="bcod-full")
        for backend_config in (reduced, full):
            other = encode_centralized_state(env.scenario, env._frame,
                previous_readings=env.task._previous_readings,
                reached=env.task.reached, steps=env.task.steps, config=backend_config)
            np.testing.assert_array_equal(advanced, other)
        assert all(obs.shape == (78,) for obs in next_obs.values())
        assert all("state" not in info for info in next_info.values())
    finally:
        env.close()


def test_centralized_state_requires_reset():
    env = make_env()
    try:
        with pytest.raises(RuntimeError, match="Reset"):
            env.state()
    finally:
        env.close()
