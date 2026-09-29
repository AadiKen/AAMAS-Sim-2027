"""Cheap runner contract checks; the one-rollout smoke is a separate gate."""
from pathlib import Path

import pytest
import numpy as np
from stable_baselines3 import PPO

from bcod_sim.benchmark_v2.config import TaskConfig
from bcod_sim.benchmark_v2.runner import _read_config, _status, train
from bcod_sim.benchmark_v2.gym_env import NavigationGymEnv


def test_locked_case_a_configuration():
    settings, task = _read_config(Path("configs/benchmark_v2/case_a_sb3.yaml"))
    assert settings["algorithm"] == "sb3_ppo"
    assert settings["n_envs"] == task.agent_count == 1
    assert (settings["n_steps"], settings["batch_size"], settings["n_epochs"]) == (2048, 64, 10)
    assert task.gamma == pytest.approx(0.99 ** 0.2)


def test_status_distinguishes_requested_from_actual_steps():
    class Model:
        num_timesteps = 2048
        _n_updates = 10

    state = _status(Model(), {"n_steps": 2048}, TaskConfig(), state="interrupted", requested=51200)
    assert state["actual_environment_steps"] == 2048
    assert state["requested_environment_steps"] == 51200
    assert state["agent_transitions"] == 2048
    assert state["completed_rollouts"] == 1
    assert state["optimizer_epochs"] == 10


def test_invalid_step_count_does_not_create_run_directory(tmp_path):
    run_dir = tmp_path / "no-run"
    with pytest.raises(ValueError, match="multiple of n_steps"):
        train(Path("configs/benchmark_v2/case_a_sb3.yaml"), run_dir, 10)
    assert not run_dir.exists()


def test_runner_has_no_legacy_learner_import():
    source = Path("src/bcod_sim/benchmark_v2/runner.py").read_text()
    assert "from bcod_sim.benchmark.ppo" not in source
    assert "from bcod_sim.benchmark.runner" not in source


def test_sb3_policy_save_reload_matches_deterministic_action(tmp_path):
    env = NavigationGymEnv(TaskConfig())
    try:
        observation, _ = env.reset(seed=11)
        model = PPO("MlpPolicy", env, seed=11, n_steps=16, batch_size=8, n_epochs=1, device="cpu")
        before, _ = model.predict(observation, deterministic=True)
        path = tmp_path / "model.zip"
        model.save(path)
        restored = PPO.load(path, device="cpu")
        after, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_array_equal(before, after)
    finally:
        env.close()


def test_fresh_run_rejects_existing_directory(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    (run_dir / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError):
        train(Path("configs/benchmark_v2/case_a_sb3.yaml"), run_dir, 2048)
    assert (run_dir / "sentinel").read_text() == "preserve"
