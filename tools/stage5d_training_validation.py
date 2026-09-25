"""Stage 5D training integration, routing, replay, and stress campaign."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/validation"))
from test_stage5a_marl import command, make_world

from bcod_sim.core.environment_loads import LinearEnvironmentLoads
from bcod_sim.rl.vector_env import VectorEnvironment
from bcod_sim.rl.training_control import TrainingControl

PROCESS = psutil.Process()
TRAIN_SEEDS = (11, 29, 47)
TASK = {"kind": "waypoint_team", "target_ned_m": (5., 0., 0.),
        "radius_m": 1., "success_bonus": 1.}


def training_env(*, heterogeneous=False, max_steps=10):
    names = ("a", "b")
    return make_world(names, max_steps=max_steps, task_payload=TASK,
        spawn_by_name={"a": (0., -0.6, 0.), "b": (0., 0.6, 0.)},
        sensor_by_name=({"a": ("truth", 10, 0.), "b": ("gps", 5, 0.)}
                        if heterogeneous else {}),
        mass_by_name=({"a": 10., "b": 16.} if heterogeneous else {}),
        thrust_bounds_by_name=({"a": (-100., 100.), "b": (-60., 60.)}
                               if heterogeneous else {}))


def policy_action(logit, generator, magnitude):
    probability = torch.sigmoid(logit)
    choice = torch.bernoulli(probability.detach(), generator=generator)
    log_probability = torch.where(choice > 0, torch.nn.functional.logsigmoid(logit),
                                  torch.nn.functional.logsigmoid(-logit))
    return command(magnitude if choice.item() else -magnitude), log_probability


def train(seed, *, heterogeneous=False, epochs=32, env_count=4, run_dir=None):
    """REINFORCE with per-agent returns and independent vector environments."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    logits = {"a": torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))}
    if heterogeneous:
        logits["b"] = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.Adam(logits.values(), lr=0.2)
    history = []
    control = None
    if run_dir is not None:
        run_path = Path(run_dir)
        def change_lr(value):
            for group in optimizer.param_groups:
                group['lr'] = value
        def save_checkpoint(step):
            path = run_path / 'checkpoints' / f'step_{step}.pt'
            temporary = path.with_suffix('.tmp')
            torch.save({'logits': {name: float(value.detach()) for name, value in logits.items()},
                        'heterogeneous': heterogeneous, 'seed': seed, 'task': TASK}, temporary)
            os.replace(temporary, path)
            return path
        def run_evaluation(episodes):
            values = {name: float(value.detach()) for name, value in logits.items()}
            result = evaluate(seed + 10000, values, heterogeneous=heterogeneous, episodes=episodes)
            return {key: value for key, value in result.items() if key != 'episodes'}
        control = TrainingControl(run_path, {'seed': seed, 'heterogeneous': heterogeneous,
            'epochs': epochs, 'env_count': env_count, 'learning_rate': 0.2},
            set_learning_rate=change_lr, evaluate=run_evaluation, checkpoint=save_checkpoint)
        change_lr(control.runtime_config['learning_rate'])
    for epoch in range(epochs):
        if control and not control.wait_if_paused(): break
        vector = VectorEnvironment({i: training_env(heterogeneous=heterogeneous).engine
                                    for i in range(env_count)})
        vector.reset(seeds={i: seed * 100000 + epoch * 100 + i for i in vector.env_ids})
        logs = {(i, name): [] for i in vector.env_ids for name in ("a", "b")}
        rewards = {(i, name): [] for i in vector.env_ids for name in ("a", "b")}
        for _ in range(10):
            actions = {}
            for i in vector.env_ids:
                actions[i] = {}
                for name in ("a", "b"):
                    key = name if heterogeneous else "a"
                    magnitude = 60. if heterogeneous and name == "b" else 100.
                    actions[i][name], logp = policy_action(logits[key], generator, magnitude)
                    logs[(i, name)].append(logp)
            result = vector.step(actions)
            for i in vector.env_ids:
                if set(result.rewards[i]) != {"a", "b"}:
                    raise AssertionError("Reward attribution changed during training")
                for name in ("a", "b"):
                    rewards[(i, name)].append(result.rewards[i][name])
        returns, scores = [], []
        for identity, values in rewards.items():
            future = 0.
            per_step = []
            for reward in reversed(values):
                future += reward
                per_step.append(future)
            returns.append(torch.tensor(list(reversed(per_step)), dtype=torch.float64))
            scores.append(sum(values))
        all_returns = torch.cat(returns)
        baseline = all_returns.mean()
        advantage_scale = all_returns.std().clamp_min(1e-4)
        loss = sum((-(torch.stack(logs[identity]) *
                      ((ret - baseline) / advantage_scale).detach()).sum())
                   for identity, ret in zip(logs, returns)) / len(all_returns)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history.append({"epoch": epoch, "mean_return": sum(scores)/len(scores),
                        "prob_a": torch.sigmoid(logits["a"]).item(),
                        "prob_b": torch.sigmoid(logits["b"]).item() if heterogeneous else None})
        if control:
            control.progress(step=epoch + 1, episode=(epoch + 1) * env_count,
                metrics={'episode_return': history[-1]['mean_return'], 'loss': float(loss.detach()),
                         'learning_rate': optimizer.param_groups[0]['lr']})
            if not control.wait_if_paused(): break
    if control: control.finish()
    return {name: float(value.detach()) for name, value in logits.items()}, history


def evaluate(seed, logits=None, *, heterogeneous=False, episodes=20, trace=False):
    generator = torch.Generator().manual_seed(seed)
    totals, outcomes = [], []
    for episode in range(episodes):
        env = training_env(heterogeneous=heterogeneous)
        env.reset(seed=seed + episode)
        reward = {"a": 0., "b": 0.}
        record = []
        for step in range(10):
            actions = {}
            for name in ("a", "b"):
                magnitude = 60. if heterogeneous and name == "b" else 100.
                if logits is None:
                    sign = 1 if torch.rand((), generator=generator).item() >= 0.5 else -1
                else:
                    key = name if heterogeneous else "a"
                    sign = 1 if logits[key] >= 0 else -1
                actions[name] = command(sign * magnitude)
            observation, rewards, terminations, truncations, infos = env.step(actions)
            for name in reward:
                reward[name] += rewards[name]
            if trace:
                record.append({"step": step, "actions": {n: actions[n].commands[0][1].thrust_n
                    for n in actions}, "rewards": rewards,
                    "position": {n: env.engine.states[n].position_ned.tolist() for n in reward},
                    "terminations": terminations, "truncations": truncations,
                    "sensor_age": {n: infos[n]["observation_freshness"] for n in reward}})
            if not env.agents:
                break
        totals.append(sum(reward.values()) / 2)
        outcomes.append({"episode": episode, "return": totals[-1],
                         "success": any(terminations.values()) if env.engine.termination_reason else False,
                         "steps": len(record) if trace else step+1, "trace": record if trace else None})
    return {"mean_return": float(np.mean(totals)),
            "success_rate": sum(x["success"] for x in outcomes) / episodes,
            "mean_length": float(np.mean([x["steps"] for x in outcomes])),
            "episodes": outcomes}


def checkpoint_reload(checkpoint, expected_path):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = evaluate(901, data["logits"], heterogeneous=data["heterogeneous"],
                      episodes=3, trace=True)
    Path(expected_path).write_text(json.dumps(result, sort_keys=True) + "\n")


def stress_env(*, max_steps=500, seed=0):
    names = ("a", "b", "c", "d")
    sensor = {"a": ("truth", 10, 0.), "b": ("gps", 5, 0.01, 0.1),
              "c": ("truth", 5, 0.), "d": ("gps", 10, 0.02, 0.15)}
    environment = {"current": {"kind": "uniform", "ned_mps": [0.08, 0.02, 0]},
                   "wind": {"kind": "uniform", "ned_mps": [0.04, -0.03, 0]},
                   "waves": {"kind": "irregular", "spectrum": "jonswap",
                             "significant_height_m": 0.02, "peak_period_s": 5.,
                             "direction_rad": 0.2, "component_count": 8, "seed": seed},
                   "visibility_m": 1000}
    obstacles = ({"id": "marker", "position_ned_m": (1.2, 0., 0.),
                  "shape": {"kind": "sphere", "radius_m": 0.3}},)
    return make_world(names, max_steps=max_steps, goal=(1e6, 0., 0.),
        sensor_by_name=sensor, sensor_latency_by_name={"b": 1, "d": 2},
        mass_by_name={"b": 14., "c": 18., "d": 12.},
        inertia_by_name={"b": (5., 6., 7.), "c": (6., 7., 8.)},
        thrust_bounds_by_name={"b": (-60., 60.), "c": (-80., 80.)},
        spawn_by_name={"a": (0., -0.5, 0.), "b": (0., 0.5, 0.),
                       "c": (-0.5, -0.5, 0.), "d": (-0.5, 0.5, 0.)},
        environment_config=environment, static_entities=obstacles,
        environment_loads_by_name={name: LinearEnvironmentLoads(0.01, 0.01, 0.01, 0.)
                                   for name in names},
        bottom=0.3, seabed_collision=True)


def stress_run(seed, steps, *, digest=False):
    env = stress_env(seed=seed)
    env.reset(seed=seed)
    hasher = hashlib.sha256()
    contacts = grounding = resets = invalid_sensor_packets = 0
    total_return = {name: 0. for name in ("a", "b", "c", "d")}
    rss = []
    start = time.perf_counter()
    for index in range(steps):
        if not env.agents:
            resets += 1
            env.reset(seed=seed, options={"episode_index": resets})
        actions = {name: command((20. if (index // 11 + ordinal) % 2 else -20.))
                   for ordinal, name in enumerate(env.agents)}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if set(rewards) != set(actions) or set(infos) != set(actions):
            raise AssertionError("Stress reward/identity corruption")
        for name in actions:
            total_return[name] += float(rewards[name])
            for packet in infos[name]["observation_freshness"].values():
                if packet["validity"] != "valid":
                    invalid_sensor_packets += 1
        for name, values in obs.items():
            for value in values.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                    raise AssertionError(f"Nonfinite stress observation: {name}")
        for name, state in env.engine.states.items():
            if not all(torch.isfinite(x).all().item() for x in
                       (state.position_ned, state.q_body_to_ned, state.nu_body)):
                raise AssertionError(f"Nonfinite stress state: {name}")
            if not math.isfinite(float(rewards[name])):
                raise AssertionError(f"Nonfinite stress reward: {name}")
            if digest:
                hasher.update(name.encode()); hasher.update(state.position_ned.numpy().tobytes())
                hasher.update(state.nu_body.numpy().tobytes())
                hasher.update(repr((rewards[name], terms[name], truncs[name],
                                    infos[name]["observation_freshness"])).encode())
        contacts += len({event.contact_id for info in infos.values()
                         for event in info["contact_events"]})
        grounding += sum(bool(info["grounding"] and info["grounding"]["contact_active"])
                         for info in infos.values())
        if (index + 1) % 5000 == 0:
            rss.append({"step": index+1, "rss_mb": PROCESS.memory_info().rss / 2**20,
                        "contacts": contacts, "grounding": grounding,
                        "return": dict(total_return),
                        "invalid_sensor_packets": invalid_sensor_packets})
    elapsed = time.perf_counter() - start
    return {"seed": seed, "steps": steps, "elapsed_s": elapsed,
            "steps_per_s": steps / elapsed, "contacts": contacts,
            "grounding_contacts": grounding, "resets": resets,
            "total_return": total_return,
            "invalid_sensor_packets": invalid_sensor_packets,
            "rss_samples": rss, "digest": hasher.hexdigest() if digest else None}


def stress_vector(seed, env_count=4, steps=100):
    vector = VectorEnvironment({i: stress_env(max_steps=steps+1, seed=seed+i).engine
                                for i in range(env_count)})
    vector.reset(seeds={i: seed+i for i in vector.env_ids})
    counts = {i: 0 for i in vector.env_ids}
    rewards = {i: 0. for i in vector.env_ids}
    for step in range(steps):
        actions = {i: {name: command(20. if (step+i) % 2 else -20.)
                       for name in ("a", "b", "c", "d")}
                   for i in vector.env_ids}
        result = vector.step(actions)
        if set(result.frames) != set(vector.env_ids):
            raise AssertionError("Parallel stress environment disappeared")
        for i in vector.env_ids:
            if set(result.frames[i].states) != {"a", "b", "c", "d"} or set(result.rewards[i]) != set(actions[i]):
                raise AssertionError("Parallel stress agent/reward attribution changed")
            counts[i] += 1
            rewards[i] += sum(result.rewards[i].values())
            for state in result.frames[i].states.values():
                if not torch.isfinite(state.nu_body).all().item():
                    raise AssertionError("Nonfinite parallel stress state")
    return {"env_count": env_count, "agent_count": 4, "steps_per_env": steps,
            "counts": counts, "returns": rewards}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", nargs=2, metavar=("CHECKPOINT", "OUTPUT"))
    parser.add_argument("--stress-steps", type=int, default=50000)
    args = parser.parse_args()
    if args.reload:
        checkpoint_reload(*args.reload)
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "stage5_results" / f"stage5d-{run_id}"
    for folder in ("training_configs", "random_baseline", "shared_policy",
                   "heterogeneous_policy", "checkpoints", "evaluation",
                   "max_stress", "seeds", "adversarial_starts", "long_rollout",
                   "reproduction", "plots", "failures"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", {"run_id": run_id,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "seeds": TRAIN_SEEDS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage3_coefficient_fidelity": "pending"})
    write_json(out / "training_configs" / "task.json", {"task": TASK,
        "algorithm": "REINFORCE", "epochs": 32, "vector_envs": 4,
        "episode_steps": 10, "agents": ["a", "b"]})
    try:
        baseline = evaluate(700, episodes=30)
        write_json(out / "random_baseline" / "evaluation.json", baseline)
        trained = []
        for seed in TRAIN_SEEDS:
            logits, history = train(seed)
            evaluation = evaluate(700, logits, episodes=30)
            trained.append({"seed": seed, "logits": logits,
                            "mean_return": evaluation["mean_return"]})
            write_json(out / "shared_policy" / f"seed_{seed}.json",
                       {"history": history, "evaluation": evaluation})
        best = max(trained, key=lambda row: row["mean_return"])
        if best["mean_return"] <= baseline["mean_return"] + 0.05:
            raise AssertionError("Shared policy did not improve over random baseline")
        hetero_logits, hetero_history = train(71, heterogeneous=True)
        hetero_eval = evaluate(700, hetero_logits, heterogeneous=True, episodes=30)
        write_json(out / "heterogeneous_policy" / "training.json",
                   {"logits": hetero_logits, "history": hetero_history,
                    "evaluation": hetero_eval})
        if set(hetero_logits) != {"a", "b"}:
            raise AssertionError("Heterogeneous policy mapping missing agent")
        checkpoint = out / "checkpoints" / "shared_policy.pt"
        torch.save({"logits": best["logits"], "heterogeneous": False,
                    "seed": best["seed"], "task": TASK}, checkpoint)
        expected = evaluate(901, best["logits"], episodes=3, trace=True)
        write_json(out / "evaluation" / "in_process.json", expected)
        subprocess.run([sys.executable, __file__, "--reload", str(checkpoint),
                        str(out / "evaluation" / "fresh_process.json")], check=True, cwd=ROOT)
        actual = json.loads((out / "evaluation" / "fresh_process.json").read_text())
        if actual != expected:
            raise AssertionError("Fresh-process checkpoint evaluation changed")
        replay_a = stress_run(101, 300, digest=True)
        replay_b = stress_run(101, 300, digest=True)
        if replay_a["digest"] != replay_b["digest"]:
            raise AssertionError("Maximum-stress fixed-seed replay diverged")
        write_json(out / "max_stress" / "replay.json", {"first": replay_a, "second": replay_b})
        seed_results = [stress_run(seed, 500, digest=True) for seed in (101, 102, 103, 104, 105)]
        write_json(out / "seeds" / "five_seeds.json", seed_results)
        vector_result = stress_vector(151)
        write_json(out / "max_stress" / "parallel_envs.json", vector_result)
        long_run = stress_run(113, args.stress_steps)
        write_json(out / "long_rollout" / "result.json", long_run)
        if len(long_run["rss_samples"]) >= 4:
            samples = long_run["rss_samples"]
            if samples[-1]["rss_mb"] - samples[len(samples)//2]["rss_mb"] > 50:
                raise AssertionError("Maximum-stress RSS grew over latter half")
        summary = {"status": "PASS", "baseline_mean_return": baseline["mean_return"],
                   "shared_policy": trained, "heterogeneous_mean_return": hetero_eval["mean_return"],
                   "checkpoint_fresh_process": "PASS", "stress_replay": "PASS",
                   "stress_seeds": len(seed_results), "parallel_stress": vector_result,
                   "long_run": long_run}
    except Exception as exc:
        (out / "failures" / "exception.txt").write_text(repr(exc) + "\n")
        summary = {"status": "FAIL", "error": repr(exc)}
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text("# Stage 5D training and maximum-stress validation\n\n"
        f"Status: **{summary['status']}**.\n\n"
        f"Random baseline: {summary.get('baseline_mean_return')}.\n\n"
        f"Training: {summary.get('shared_policy')}.\n\n"
        f"Stress: {summary.get('long_run')}.\n")
    (out / "reproduction" / "run.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"\n'
        '"${repo_root}/.venv/bin/python" "${repo_root}/tools/stage5d_training_validation.py"\n')
    print(out)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
