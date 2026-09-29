from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from bcod_sim.benchmark_v3.config import TaskConfig
from bcod_sim.benchmark_v3.gym_env import NavigationGymEnv
from bcod_sim.benchmark_v3.runner import (DEFAULT_CONFIG, EpisodeLogger, load_policy, read_config,
                                            make_model, save_policy, status, train_one)
from bcod_sim.benchmark_v3.scenarios import (s1_validation_bank, s1_dev_bank,
                                              s1_test_bank, s1_final_test_bank, sample_s1_train,
                                              validate_s1_scenario)
from bcod_sim.benchmark_v3.evaluate import task_metric_key
from bcod_sim.benchmark_v3.qualification import (bank_hash, selection_reason,
                                                   initialize_policy_from_checkpoint)
from bcod_sim.benchmark_v3.imitation import DemonstrationSet, behavior_clone


def test_stock_ppo_config_and_version_guard(tmp_path):
    settings, task = read_config(DEFAULT_CONFIG)
    assert settings["n_steps"] == 2048 and settings["n_epochs"] == 10
    assert task.gamma == pytest.approx(.99 ** .2)
    env = NavigationGymEnv(task)
    try:
        observation, _ = env.reset(seed=11)
        model = PPO("MlpPolicy", env, seed=11, n_steps=16, batch_size=8,
                    n_epochs=1, device="cpu")
        path = tmp_path / "step-0.zip"
        save_policy(model, path, task)
        before, _ = model.predict(observation, deterministic=True)
        loaded = load_policy(path, task)
        after, _ = loaded.predict(observation, deterministic=True)
        np.testing.assert_array_equal(before, after)
        path.with_suffix(".schema.json").unlink()
        with pytest.raises(ValueError, match="migration required"):
            load_policy(path, task)
    finally:
        env.close()


def test_policy_warm_start_copies_weights_with_fresh_optimizer(tmp_path):
    settings, task = read_config(DEFAULT_CONFIG)
    source, source_env = make_model(settings, task)
    target, target_env = make_model({**settings, "seed": 73}, task)
    try:
        with __import__("torch").no_grad():
            source.policy.action_net.bias.add_(0.4)
        checkpoint = tmp_path / "source.zip"
        save_policy(source, checkpoint, task)
        initialize_policy_from_checkpoint(target, checkpoint, task)
        for name, value in source.policy.state_dict().items():
            assert __import__("torch").equal(value, target.policy.state_dict()[name])
        assert target.policy.optimizer.state == {}
        assert target.num_timesteps == 0
    finally:
        source_env.close()
        target_env.close()


def test_generic_behavior_cloning_fits_actions_without_critic_updates():
    settings, task = read_config(DEFAULT_CONFIG)
    model, env = make_model(settings, task)
    try:
        obs = np.zeros((64, 78), dtype=np.float32)
        actions = np.tile(np.array([0.25, -0.5], dtype=np.float32), (64, 1))
        critic_before = {name: value.clone() for name, value in
                         model.policy.value_net.state_dict().items()}
        losses = behavior_clone(model, DemonstrationSet(obs, actions, ("synthetic",)),
                                epochs=12, batch_size=32, learning_rate=1e-3)
        assert losses[-1] < losses[0] * .5
        for name, value in critic_before.items():
            assert __import__("torch").equal(value, model.policy.value_net.state_dict()[name])
    finally:
        env.close()


def test_fresh_run_does_not_merge_with_existing(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "old.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        train_one(DEFAULT_CONFIG, output, total_timesteps=2048)
    assert (output / "old.txt").read_text() == "preserve"


def test_v3_runner_does_not_import_custom_optimizers():
    text = Path("src/bcod_sim/benchmark_v3/runner.py").read_text()
    assert "from stable_baselines3 import PPO" in text
    assert "benchmark.ppo" not in text
    assert "benchmark.audit_learning" not in text
    assert "benchmark.runner" not in text


def test_episode_logger_records_accumulated_values(tmp_path):
    logger = EpisodeLogger(tmp_path / "episodes.jsonl")
    logger.locals = {"dones": [True], "infos": [{"terminal_reason": "deadline", "steps": 600,
        "final_goal_distance_m": {"vessel_0": 8.}, "episode_returns": {"vessel_0": 4.},
        "episode_reward_components": {"vessel_0": {"goal": 0., "step": -6.,
                                                        "potential_shaping": 10., "collision": 0.}}}]}
    logger.num_timesteps = 600
    assert logger._on_step()
    import json
    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert record["episode_return"] == 4.
    assert sum(record["reward_components"].values()) == 4.


def test_s1_bank_has_unique_feasible_obstructed_geometry():
    bank = s1_validation_bank()
    assert len(bank) == len({case.geometry_hash() for case in bank}) == 50
    assert {len(case.obstacles) for case in bank} == {1, 2, 3, 4}
    for case in bank:
        validate_s1_scenario(case)
        assert case.split == "S1-validation"


def test_s1_selector_orders_deadline_before_completion():
    common = {"success_rate": .8, "collision_rate": .1,
              "mean_success_completion_steps": 100., "mean_path_efficiency": .7}
    assert task_metric_key({**common, "deadline_rate": .1}, stage="S1") > task_metric_key(
        {**common, "deadline_rate": .2, "mean_success_completion_steps": 80.}, stage="S1")


def test_s1_dev_test_banks_are_fixed_disjoint_and_feasible():
    dev, test, final = s1_dev_bank(), s1_test_bank(), s1_final_test_bank()
    assert len(dev) == 50 and len(test) == len(final) == 100
    assert bank_hash(dev) == "3666670b1d6bddfe6144e7cd24b7f7777cc61dff06b88aac4f820e37d683f561"
    assert bank_hash(test) == "0aef5f73abdb32ccb3417719d4166f75d7ed906bebb510ce9e687ec383e7d705"
    assert bank_hash(final) == "9932b07c41f2701e394a86e3dfd9d7a03c45601149f1de5ac1daa55bb0a38e7f"
    assert not ({c.geometry_hash() for c in dev} & {c.geometry_hash() for c in (*test, *final)})
    assert not ({c.geometry_hash() for c in test} & {c.geometry_hash() for c in final})
    for case in (*test, *final):
        validate_s1_scenario(case)
    rng = np.random.default_rng(44)
    reserved = {c.geometry_hash() for c in (*dev, *test, *final)}
    assert all(sample_s1_train(rng).geometry_hash() not in reserved for _ in range(100))


def test_s1_best_dev_selector_is_strict_lexicographic_with_earlier_ties():
    base = {"success_rate": .8, "collision_rate": .1, "deadline_rate": .1,
            "mean_success_completion_steps": 150., "mean_path_efficiency": .7}
    assert selection_reason(base, dict(base)) is None
    assert selection_reason(base, {**base, "success_rate": .82, "collision_rate": .9})
    assert selection_reason(base, {**base, "collision_rate": .08, "deadline_rate": .9})
    assert selection_reason(base, {**base, "deadline_rate": .08,
                                   "mean_success_completion_steps": 200.})
    assert selection_reason(base, {**base, "mean_success_completion_steps": 140.})
    assert selection_reason(base, {**base, "mean_path_efficiency": .8})
    assert selection_reason(base, {**base, "success_rate": .79}) is None


def test_s1_eight_environment_accounting_and_independent_resets():
    settings, task = read_config(Path("configs/benchmark_v3/s1_ppo_stability_a.yaml"), seed=11)
    model, env = make_model(settings, task)
    try:
        assert model.n_envs == 8 and model.n_steps == 256
        assert len({id(item) for item in env.envs}) == 8
        env.reset()
        assert len({info["scenario_hash"] for info in env.reset_infos}) == 8
        assert [item.np_random_seed for item in env.envs] == list(range(11, 19))
        model.num_timesteps = 2048
        record = status(model, settings, 51200, "running")
        assert (record["actual_environment_steps"], record["vector_steps"],
                record["rollout_transitions"], record["completed_rollouts"]) == (2048, 256, 2048, 1)
    finally:
        env.close()
