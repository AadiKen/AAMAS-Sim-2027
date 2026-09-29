import pytest

from bcod_sim.benchmark_v3.config import TaskConfig
from bcod_sim.benchmark_v3.gym_env import NavigationGymEnv
from bcod_sim.benchmark_v3.scenarios import case_a_reference


@pytest.mark.parametrize("backend_name", ["kinematic", "bcod-reduced", "bcod-full"])
def test_same_v3_task_runs_on_each_backend(backend_name):
    config = TaskConfig(dynamics=backend_name)
    env = NavigationGymEnv(config, scenario=case_a_reference(11))
    try:
        obs, info = env.reset(seed=11)
        assert obs.shape == (78,)
        if backend_name.startswith("bcod"):
            assert len(env.backend.engine.vessels) == 1
            assert tuple(env.backend.engine.vessels) == ("vessel_0",)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step([0., 0.])
            assert obs.shape == (78,) and not terminated and not truncated
            assert info["physical_command"] == {"speed_mps": 1., "yaw_rate_radps": 0.}
            assert reward == pytest.approx(sum(info["reward_components"]["vessel_0"].values()))
            if backend_name.startswith("bcod"):
                physical = info["backend_diagnostics"]["physical_6dof"]["vessel_0"]
                assert all(abs(physical[key]) < 10 for key in ("roll_rad", "pitch_rad", "sway_frd_mps"))
                assert isinstance(info["backend_diagnostics"]["saturated"]["vessel_0"], bool)
    finally:
        env.close()
