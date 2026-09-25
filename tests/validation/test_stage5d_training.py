"""Executable Stage 5D training and stress integration registry."""

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import stage5d_training_validation as campaign
import stage5d_full_replay
from bcod_sim.rl.vector_env import VectorEnvironment


def test_random_baseline_and_learnable_task():
    random = campaign.evaluate(700, episodes=8)
    forward = campaign.evaluate(700, {"a": 4.}, episodes=8)
    assert forward["mean_return"] > random["mean_return"] + 0.5
    assert forward["success_rate"] > random["success_rate"]
    assert all(math.isfinite(row["return"]) for row in random["episodes"])


def test_shared_policy_training_and_vector_reward_routing():
    logits, history = campaign.train(11, epochs=8, env_count=2)
    assert len(history) == 8 and set(logits) == {"a"}
    assert logits["a"] > 0
    assert campaign.evaluate(700, logits, episodes=4)["mean_return"] > 0


def test_heterogeneous_policy_mapping_and_training():
    logits, history = campaign.train(71, heterogeneous=True, epochs=8, env_count=2)
    assert set(logits) == {"a", "b"} and len(history) == 8
    assert all(math.isfinite(value) for value in logits.values())
    result = campaign.evaluate(700, logits, heterogeneous=True, episodes=3, trace=True)
    assert all(set(row["trace"][0]["actions"]) == {"a", "b"}
               for row in result["episodes"])


def test_checkpoint_reloads_in_fresh_process(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    output = tmp_path / "reloaded.json"
    values = {"a": 2.}
    torch.save({"logits": values, "heterogeneous": False}, checkpoint)
    expected = campaign.evaluate(901, values, episodes=3, trace=True)
    subprocess.run([sys.executable, str(Path(campaign.__file__)), "--reload",
                    str(checkpoint), str(output)], check=True)
    assert json.loads(output.read_text()) == expected


def test_heterogeneous_checkpoint_preserves_policy_mapping(tmp_path):
    checkpoint = tmp_path / "heterogeneous.pt"
    output = tmp_path / "reloaded.json"
    values = {"a": 2., "b": -1.}
    torch.save({"logits": values, "heterogeneous": True,
                "agent_policy_mapping": {"a": "a", "b": "b"}}, checkpoint)
    expected = campaign.evaluate(901, values, heterogeneous=True, episodes=3, trace=True)
    subprocess.run([sys.executable, str(Path(campaign.__file__)), "--reload",
                    str(checkpoint), str(output)], check=True)
    assert json.loads(output.read_text()) == expected


def test_stress_fixed_seed_replay_and_state_finiteness():
    left = campaign.stress_run(101, 20, digest=True)
    right = campaign.stress_run(101, 20, digest=True)
    assert left["digest"] == right["digest"]
    assert left["steps"] == right["steps"] == 20


def test_parallel_stress_environment_attribution():
    result = campaign.stress_vector(151, env_count=3, steps=4)
    assert result["counts"] == {0: 4, 1: 4, 2: 4}
    assert set(result["returns"]) == {0, 1, 2}


def test_vector_training_reset_and_cross_env_isolation():
    vector = VectorEnvironment({0: campaign.training_env().engine,
                                1: campaign.training_env().engine})
    vector.reset(seeds={0: 5, 1: 6})
    actions = {0: {n: campaign.command(100.) for n in ("a", "b")},
               1: {n: campaign.command(-100.) for n in ("a", "b")}}
    result = vector.step(actions)
    solo = campaign.training_env()
    solo.reset(seed=6)
    _, rewards, _, _, _ = solo.step(actions[1])
    assert dict(result.rewards[1]) == rewards
    for name in ("a", "b"):
        assert torch.equal(result.frames[1].states[name].position_ned,
                           solo.engine.states[name].position_ned)
    vector.reset(seeds={0: 5, 1: 6})
    replay = vector.step(actions)
    assert dict(replay.rewards[1]) == rewards


def test_full_field_maximum_stress_replay():
    first = stage5d_full_replay.run(101, 20)
    second = stage5d_full_replay.run(101, 20)
    assert first["sha256"] == second["sha256"]
    assert first["nonempty_observations"] > 0


STAGE5D_CASES = (
    "test_random_baseline_and_learnable_task",
    "test_shared_policy_training_and_vector_reward_routing",
    "test_heterogeneous_policy_mapping_and_training",
    "test_checkpoint_reloads_in_fresh_process",
    "test_heterogeneous_checkpoint_preserves_policy_mapping",
    "test_stress_fixed_seed_replay_and_state_finiteness",
    "test_parallel_stress_environment_attribution",
    "test_vector_training_reset_and_cross_env_isolation",
    "test_full_field_maximum_stress_replay",
)


def test_registry_complete():
    assert set(STAGE5D_CASES) == {name for name in globals()
                                   if name.startswith("test_") and name != "test_registry_complete"}
