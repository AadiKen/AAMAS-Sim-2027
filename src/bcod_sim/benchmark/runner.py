"""One shared actor-critic rollout, checkpoint, and evaluation pipeline."""

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
try:
    import psutil
except ImportError:
    psutil = None

from .bcod_adapter import BCODAdapter
from .core import Benchmark, BenchmarkConfig, NAMES, Scenario, generate_scenario
from .holoocean_adapter import HoloOceanAdapter
from .pyquaticus_adapter import PyquaticusAdapter


class SharedActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(47, 128), torch.nn.Tanh(),
                                        torch.nn.Linear(128, 128), torch.nn.Tanh())
        self.actor = torch.nn.Linear(128, 2)
        self.critic = torch.nn.Linear(128, 1)
        self.log_std = torch.nn.Parameter(torch.full((2,), -0.7))

    def forward(self, obs):
        hidden = self.body(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def make_adapter(sim, config, *, pyquaticus_python=None):
    if sim == "bcod":
        return BCODAdapter(config)
    if sim == "bcod-reduced":
        return BCODAdapter(config, reduced_fidelity=True)
    if sim == "pyquaticus":
        if not pyquaticus_python:
            raise RuntimeError("Pyquaticus requires --pyquaticus-python pointing to isolated Python 3.10")
        return PyquaticusAdapter(pyquaticus_python, config)
    if sim == "holoocean":
        return HoloOceanAdapter(config)
    raise ValueError(f"Unknown simulator: {sim}")


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def _action(model, observations, device, deterministic=False):
    tensor = torch.as_tensor(np.stack([observations[n] for n in NAMES]), dtype=torch.float32, device=device)
    mean, value = model(tensor)
    distribution = torch.distributions.Normal(mean, model.log_std.exp())
    raw = mean if deterministic else distribution.sample()
    action = torch.tanh(raw)
    log_probability = distribution.log_prob(raw).sum(-1)
    return {name: tuple(float(x) for x in action[i].detach().cpu()) for i, name in enumerate(NAMES)}, log_probability, value


def _update(model, optimizer, records, bootstrap, gamma=0.99):
    future = {i: value.detach() for i, value in bootstrap.items()}
    losses = []
    for env_id, log_probability, value, reward, done in reversed(records):
        future[env_id] = reward + gamma * future[env_id] * (not done)
        advantage = future[env_id] - value
        losses.append(-(log_probability * advantage.detach()).mean() + 0.5 * advantage.square().mean())
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def _config(path):
    if not path:
        return BenchmarkConfig()
    data = json.loads(Path(path).read_text())
    allowed = {f.name for f in fields(BenchmarkConfig)}
    if set(data) - allowed:
        raise ValueError(f"Unknown benchmark configuration keys: {sorted(set(data) - allowed)}")
    return BenchmarkConfig(**data)


def train(args):
    start = time.perf_counter()
    config = _config(args.config)
    if args.sim == "holoocean":
        make_adapter(args.sim, config)
    if args.sim == "pyquaticus" and not args.pyquaticus_python:
        raise RuntimeError("Pyquaticus requires --pyquaticus-python")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model = SharedActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    output = Path(args.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"simulator": args.sim, "seed": args.seed, "benchmark": asdict(config),
                "algorithm": "shared_actor_critic_v1", "architecture": "47-128-128-tanh-gaussian-2",
                "optimizer": {"type": "Adam", "learning_rate": 3e-4}, "gamma": 0.99,
                "total_steps": args.total_steps, "num_envs": args.num_envs,
                "checkpoint_interval": args.checkpoint_interval, "device": args.device,
                "started_at": _stamp()}
    (output / "run-config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    envs = [Benchmark(make_adapter(args.sim, config, pyquaticus_python=args.pyquaticus_python), config)
            for _ in range(args.num_envs)]
    observations = [env.reset(generate_scenario(args.seed * 100000 + i)) for i, env in enumerate(envs)]
    episode_index = [0] * len(envs)
    episode_rewards = [0.] * len(envs)
    episode_count = 0
    successes = 0
    collisions = 0
    completed = []
    checkpoint_index = 0
    environment_seconds = 0.
    try:
        for joint_step in range(1, args.total_steps + 1):
            i = (joint_step - 1) % len(envs)
            actions, log_probability, value = _action(model, observations[i], device)
            tick_start = time.perf_counter()
            obs, reward, done, truncated, info = envs[i].step(actions)
            environment_seconds += time.perf_counter() - tick_start
            rewards = torch.tensor([reward[n] for n in NAMES], dtype=torch.float32, device=device)
            completed.append((i, log_probability, value, rewards, done or truncated))
            observations[i] = obs
            episode_rewards[i] += sum(reward.values())
            if done or truncated:
                episode_count += 1
                successes += int(info["fleet_success"])
                collisions += int(info["collision_count"] > 0)
                _append(output / "episodes.jsonl", {"simulator": args.sim, "seed": args.seed,
                    "environment_steps": joint_step, "episode": episode_count,
                    "cumulative_reward": episode_rewards[i], **info})
                episode_rewards[i] = 0.
                episode_index[i] += 1
                observations[i] = envs[i].reset(generate_scenario(args.seed * 100000 + i +
                                                        episode_index[i] * args.num_envs))
            checkpoint_due = joint_step % args.checkpoint_interval == 0 or joint_step == args.total_steps
            if checkpoint_due:
                with torch.no_grad():
                    bootstrap = {j: model(torch.as_tensor(np.stack([observations[j][n] for n in NAMES]),
                                    dtype=torch.float32, device=device))[1]
                                 for j in range(len(envs))}
                loss = _update(model, optimizer, completed, bootstrap)
                completed = []
                checkpoint_index += 1
                elapsed = time.perf_counter() - start
                checkpoint = output / f"checkpoint-{joint_step:09d}.pt"
                torch.save({"model": model.state_dict(), "manifest": manifest,
                            "environment_steps": joint_step, "checkpoint_index": checkpoint_index}, checkpoint)
                _append(output / "training.jsonl", {"simulator": args.sim, "seed": args.seed,
                    "environment_steps": joint_step, "episodes": episode_count,
                    "fleet_success_rate": successes / episode_count if episode_count else None,
                    "collision_rate": collisions / episode_count if episode_count else None,
                    "loss": loss, "environment_steps_per_second": joint_step / max(environment_seconds, 1e-9),
                    "trainer_steps_per_second": joint_step / max(elapsed, 1e-9),
                    "cpu_percent": (100 * sum(psutil.Process().cpu_times()[:2]) / max(elapsed, 1e-9)
                                    if psutil else None),
                    "ram_rss_bytes": psutil.Process().memory_info().rss if psutil else None,
                    "total_training_wall_clock_s": elapsed, "checkpoint_number": checkpoint_index,
                    "checkpoint_wall_clock_timestamp": _stamp(), "checkpoint": str(checkpoint)})
        return checkpoint
    finally:
        for env in envs:
            env.close()


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = BenchmarkConfig(**checkpoint["manifest"]["benchmark"])
    model = SharedActorCritic().to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.scenario_bank:
        files = sorted(Path(args.scenario_bank).glob("*.json"))
        scenarios = [Scenario.load(path) for path in files[:args.episodes]]
        if len(scenarios) < args.episodes:
            raise ValueError("Scenario bank contains fewer files than requested episodes")
        if any(s.split != args.mode for s in scenarios):
            raise ValueError("Scenario bank split does not match evaluation mode")
    else:
        scenarios = [generate_scenario(900000 + i, args.mode, stress=args.mode == "stress", config=config)
                     for i in range(args.episodes)]
    env = Benchmark(make_adapter(args.sim, config, pyquaticus_python=args.pyquaticus_python), config)
    rows = []
    started = time.perf_counter()
    try:
        for scenario in scenarios:
            observations = env.reset(scenario)
            cumulative = 0.
            while True:
                with torch.no_grad():
                    actions, _, _ = _action(model, observations, args.device, args.deterministic)
                observations, reward, done, truncated, info = env.step(actions)
                cumulative += sum(reward.values())
                if done or truncated:
                    rows.append({"scenario_seed": scenario.seed, "cumulative_reward": cumulative, **info})
                    break
    finally:
        env.close()
    result = {"simulator": args.sim, "checkpoint": str(args.checkpoint), "mode": args.mode,
              "deterministic": args.deterministic, "episodes": rows,
              "stress_factors_applied": (["unseen_geometry", "8_to_10_obstacles"]
                                         if args.mode == "stress" else []),
              "stress_factors_not_comparable": (["current", "waves", "sensor_noise_dropout",
                                                 "vessel_parameter_perturbation"]
                                                if args.mode == "stress" else []),
              "fleet_success_rate": sum(r["fleet_success"] for r in rows) / len(rows),
              "collision_rate": sum(r["collision_count"] > 0 for r in rows) / len(rows),
              "mean_reward": sum(r["cumulative_reward"] for r in rows) / len(rows),
              "environment_steps_per_second": sum(r["steps"] for r in rows) / max(time.perf_counter() - started, 1e-9)}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    evaluation = commands.add_parser("evaluate")
    for command in (training, evaluation):
        command.add_argument("--sim", choices=("bcod", "bcod-reduced", "pyquaticus", "holoocean"), required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--pyquaticus-python")
    training.add_argument("--seed", type=int, default=11)
    training.add_argument("--total-steps", type=int, required=True)
    training.add_argument("--num-envs", type=int, default=1)
    training.add_argument("--checkpoint-dir", required=True)
    training.add_argument("--checkpoint-interval", type=int, required=True)
    training.add_argument("--config")
    training.add_argument("--headless", action="store_true")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--scenario-bank")
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--mode", choices=("nominal", "stress"), default="nominal")
    evaluation.add_argument("--deterministic", action="store_true")
    evaluation.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.total_steps < 1 or args.num_envs < 1 or args.checkpoint_interval < 1:
            parser.error("Training counts must be positive")
        train(args)
    else:
        if args.episodes < 1:
            parser.error("Episode count must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
