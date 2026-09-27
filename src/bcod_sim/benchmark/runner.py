"""One shared actor-critic rollout, checkpoint, and evaluation pipeline."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
import uuid

import numpy as np
import torch
try:
    import psutil
except ImportError:
    psutil = None

from .bcod_adapter import BCODAdapter
from .core import Benchmark, BenchmarkConfig, DISCOUNT_PER_SECOND, NAMES, Scenario, generate_scenario
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


class _EtaBar:
    """Optional stderr progress display; never changes benchmark accounting."""

    def __init__(self, label, total, *, enabled):
        self.label = label
        self.total = total
        self.enabled = enabled
        self.started = time.perf_counter()
        self.last_drawn = float("-inf")
        self.count = 0
        if enabled:
            self.update(0)

    def update(self, count, *, force=False):
        if not self.enabled:
            return
        self.count = count
        now = time.perf_counter()
        if not force and count < self.total and now - self.last_drawn < 0.25:
            return
        elapsed = now - self.started
        remaining = elapsed * (self.total - count) / count if count else None
        eta = (f"{int(remaining // 3600):02d}:{int(remaining // 60) % 60:02d}:"
               f"{int(remaining) % 60:02d}" if remaining is not None else "--:--:--")
        filled = round(20 * count / self.total)
        sys.stderr.write(f"\r{self.label} |{'#' * filled}{'-' * (20 - filled)}| "
                         f"{count}/{self.total} ETA {eta}")
        sys.stderr.flush()
        self.last_drawn = now

    def close(self):
        if self.enabled:
            if self.count < self.total:
                self.update(self.count, force=True)
            sys.stderr.write("\n")
            sys.stderr.flush()


def _action(model, observations, device, deterministic=False, diagnostics=None, agent_names=NAMES):
    tensor = torch.as_tensor(np.stack([observations[n] for n in agent_names]), dtype=torch.float32, device=device)
    mean, value = model(tensor)
    distribution = torch.distributions.Normal(mean, model.log_std.exp())
    raw = mean if deterministic else distribution.sample()
    squashed = torch.tanh(raw)
    action = torch.stack(((squashed[..., 0] + 1) / 2, squashed[..., 1]), dim=-1)
    # Exact density of ((tanh(z_surge)+1)/2, tanh(z_yaw)). raw is a
    # detached sample, so its Jacobian is constant for score gradients.
    correction = 2 * (math.log(2.) - raw - torch.nn.functional.softplus(-2 * raw))
    log_probability = (distribution.log_prob(raw) - correction).sum(-1) - math.log(0.5)
    if diagnostics is not None:
        mean_squashed = torch.tanh(mean)
        mean_action = torch.stack(((mean_squashed[..., 0] + 1) / 2, mean_squashed[..., 1]), dim=-1)
        diagnostics.update(actor_mean=mean_action.detach().cpu().tolist(),
                           sampled=action.detach().cpu().tolist(),
                           log_std=model.log_std.detach().cpu().tolist(),
                           raw_std=model.log_std.exp().detach().cpu().tolist())
    return {name: tuple(float(x) for x in action[i].detach().cpu()) for i, name in enumerate(agent_names)}, log_probability, value


def _return_targets(records, bootstrap, gamma=None):
    """Detached, per-environment n-step targets in chronological record order."""
    if gamma is None:
        gamma = BenchmarkConfig().gamma
    future = {i: value.detach() for i, value in bootstrap.items()}
    targets = []
    for env_id, _, _, reward, boundary_value in reversed(records):
        if boundary_value is not None:
            future[env_id] = boundary_value.detach()
        future[env_id] = reward.detach() + gamma * future[env_id]
        targets.append(future[env_id])
    return list(reversed(targets))


def _update(model, optimizer, records, bootstrap, gamma=None, diagnostics=None):
    targets = torch.stack(_return_targets(records, bootstrap, gamma))
    values = torch.stack([r[2] for r in records])
    logp = torch.stack([r[1] for r in records])
    advantage = targets - values
    actor_loss = -(logp * advantage.detach()).mean()
    critic_loss = .5 * advantage.square().mean()
    loss = actor_loss + critic_loss
    optimizer.zero_grad()
    if diagnostics is not None:
        def norm(grads):
            return sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5
        body = list(model.body.parameters())
        actor_body = torch.autograd.grad(actor_loss, body, retain_graph=True, allow_unused=True)
        critic_body = torch.autograd.grad(critic_loss, body, retain_graph=True, allow_unused=True)
        # Each contribution includes that agent's full rollout, before clipping.
        actor_parameters = body + list(model.actor.parameters()) + [model.log_std]
        contributions = []
        for agent in range(logp.shape[-1]):
            agent_loss = -(logp[:, agent] * advantage[:, agent].detach()).mean() / logp.shape[-1]
            gradients = torch.autograd.grad(agent_loss, actor_parameters, retain_graph=True, allow_unused=True)
            contributions.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                                            for p, g in zip(actor_parameters, gradients)]))
        lengths = [float(g.norm()) for g in contributions]
        diagnostics['per_agent_actor_grad_norm'] = lengths
        diagnostics['actor_contribution_cosines'] = [
            [float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12)) for b in contributions]
            for a in contributions]
        diagnostics['actor_cancellation_ratio'] = float(torch.stack(contributions).sum(0).norm()) / max(sum(lengths),1e-12)
        variance = targets.var(unbiased=False)
        uses_bootstrap = dict.fromkeys(bootstrap, True)
        bootstrap_records = 0
        for env_id, _, _, _, boundary in reversed(records):
            if boundary is not None:
                uses_bootstrap[env_id] = bool(torch.any(boundary != 0))
            bootstrap_records += uses_bootstrap[env_id]
        diagnostics.update(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
            total_loss=float(loss.detach()), entropy=float(-logp.detach().mean()),
            entropy_estimator='Monte Carlo squashed Gaussian differential entropy',
            mean_advantage=float(advantage.detach().mean()), std_advantage=float(advantage.detach().std(unbiased=False)),
            min_advantage=float(advantage.detach().min()), max_advantage=float(advantage.detach().max()),
            mean_return_target=float(targets.mean()), std_return_target=float(targets.std(unbiased=False)),
            mean_predicted_value=float(values.detach().mean()),
            explained_variance=float(1-(targets-values.detach()).var(unbiased=False)/variance) if variance>1e-12 else None,
            actor_body_grad_norm=norm(actor_body), critic_body_grad_norm=norm(critic_body),
            rollout_records=len(records), episode_boundaries=sum(r[4] is not None for r in records),
            bootstrap_fraction=bootstrap_records/len(records))
    loss.backward()
    if diagnostics is not None:
        diagnostics['gradient_norms_before_clip'] = {
            'shared_body': norm(p.grad for p in model.body.parameters()),
            'actor_head': norm(p.grad for p in model.actor.parameters()),
            'critic_head': norm(p.grad for p in model.critic.parameters()),
            'log_std': norm([model.log_std.grad])}
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if diagnostics is not None:
        diagnostics['total_grad_norm_before_clip'] = float(total_norm)
        diagnostics['clip_scale'] = min(1., 1./(float(total_norm)+1e-6))
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
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise ValueError("Run directory is not empty; choose a new directory or --overwrite (archives old run)")
        output.rename(output.with_name(output.name + ".archive-" + uuid.uuid4().hex))
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    manifest = {"run_id": run_id, "simulator": args.sim, "seed": args.seed, "benchmark": asdict(config),
                "algorithm": "shared_actor_critic_v2", "architecture": "47-128-128-tanh-gaussian-2",
                "optimizer": {"type": "Adam", "learning_rate": 3e-4}, "gamma": config.gamma,
                "gamma_per_second": DISCOUNT_PER_SECOND,
                "total_steps": args.total_steps, "num_envs": args.num_envs,
                "checkpoint_interval": args.checkpoint_interval, "update_interval": args.update_interval,
                "timeout_semantics": "bootstrap_final_observation", "device": args.device,
                "eta_bar": args.eta_bar,
                "started_at": _stamp()}
    (output / "run-config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "manifest": manifest,
                "environment_steps": 0, "optimizer_updates": 0},
               output / "checkpoint-000000000.pt")
    envs, observations = [], []
    cleanup = ExitStack()
    from .rl_diagnostics import EpisodeDiagnostics
    episode_diagnostics = [EpisodeDiagnostics(config.gamma) for _ in range(args.num_envs)]
    episode_index = [0] * args.num_envs
    episode_rewards = [0.] * args.num_envs
    episode_count = 0
    successes = 0
    collisions = 0
    completed = []
    action_diagnostics = []
    checkpoint_index = 0
    optimizer_updates = 0
    loss = None
    checkpoint = None
    environment_seconds = 0.
    progress = _EtaBar(f"{args.sim} train", args.total_steps, enabled=args.eta_bar)
    try:
        for i in range(args.num_envs):
            env = Benchmark(make_adapter(args.sim, config, pyquaticus_python=args.pyquaticus_python), config)
            cleanup.callback(env.close)
            envs.append(env)
            observations.append(env.reset(generate_scenario(args.seed * 100000 + i, config=config)))
        for joint_step in range(1, args.total_steps + 1):
            i = (joint_step - 1) % len(envs)
            action_row = {}
            actions, log_probability, value = _action(model, observations[i], device, diagnostics=action_row)
            action_diagnostics.append(action_row)
            tick_start = time.perf_counter()
            obs, reward, done, truncated, info = envs[i].step(actions)
            environment_seconds += time.perf_counter() - tick_start
            rewards = torch.tensor([reward[n] for n in NAMES], dtype=torch.float32, device=device)
            boundary_value = None
            if done:
                boundary_value = torch.zeros_like(value)
            elif truncated:
                with torch.no_grad():
                    boundary_value = model(torch.as_tensor(np.stack([obs[n] for n in NAMES]),
                                           dtype=torch.float32, device=device))[1]
            completed.append((i, log_probability, value, rewards, boundary_value))
            observations[i] = obs
            episode_rewards[i] += sum(reward.values())
            episode_diagnostics[i].add(reward, info, envs[i])
            if done or truncated:
                episode_count += 1
                successes += int(info["fleet_success"])
                collisions += int(info["collision_count"] > 0)
                _append(output / "episodes.jsonl", {"run_id": run_id, "simulator": args.sim, "seed": args.seed,
                    "optimizer_updates": optimizer_updates, "environment_steps": joint_step, "episode": episode_count,
                    "cumulative_reward": episode_rewards[i], **episode_diagnostics[i].result(info)})
                episode_diagnostics[i] = EpisodeDiagnostics(config.gamma)
                episode_rewards[i] = 0.
                episode_index[i] += 1
                observations[i] = envs[i].reset(generate_scenario(args.seed * 100000 + i +
                                                        episode_index[i] * args.num_envs, config=config))
            checkpoint_due = joint_step % args.checkpoint_interval == 0 or joint_step == args.total_steps
            update_due = joint_step % args.update_interval == 0 or joint_step == args.total_steps
            if update_due:
                with torch.no_grad():
                    bootstrap = {j: model(torch.as_tensor(np.stack([observations[j][n] for n in NAMES]),
                                    dtype=torch.float32, device=device))[1]
                                 for j in range(len(envs))}
                update_diagnostics = {}
                loss = _update(model, optimizer, completed, bootstrap, config.gamma, diagnostics=update_diagnostics)
                from .rl_diagnostics import action_summary
                _append(output / "updates.jsonl", {"run_id": run_id, "environment_steps": joint_step,
                    "optimizer_updates": optimizer_updates + 1, **update_diagnostics,
                    **action_summary(action_diagnostics)})
                action_diagnostics = []
                completed = []
                optimizer_updates += 1
            if checkpoint_due:
                checkpoint_index += 1
                elapsed = time.perf_counter() - start
                checkpoint = output / f"checkpoint-{joint_step:09d}.pt"
                torch.save({"model": model.state_dict(), "manifest": manifest,
                            "environment_steps": joint_step, "optimizer_updates": optimizer_updates,
                            "optimizer": optimizer.state_dict(), "checkpoint_index": checkpoint_index,
                            "rng_states": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                                           "python": random.getstate()},
                            "resumable": False}, checkpoint)
            if update_due or checkpoint_due:
                elapsed = time.perf_counter() - start
                _append(output / "training.jsonl", {"run_id": run_id, "simulator": args.sim, "seed": args.seed,
                    "optimizer_updates": optimizer_updates, "environment_steps": joint_step, "episodes": episode_count,
                    "fleet_success_rate": successes / episode_count if episode_count else None,
                    "collision_rate": collisions / episode_count if episode_count else None,
                    "loss": loss, "environment_steps_per_second": joint_step / max(environment_seconds, 1e-9),
                    "trainer_steps_per_second": joint_step / max(elapsed, 1e-9),
                    "cpu_percent": (100 * sum(psutil.Process().cpu_times()[:2]) / max(elapsed, 1e-9)
                                    if psutil else None),
                    "ram_rss_bytes": psutil.Process().memory_info().rss if psutil else None,
                    "total_training_wall_clock_s": elapsed, "checkpoint_number": checkpoint_index,
                    "checkpoint_wall_clock_timestamp": _stamp() if checkpoint_due else None,
                    "checkpoint": str(checkpoint) if checkpoint_due else None})
            progress.update(joint_step)
        return checkpoint
    finally:
        progress.close()
        cleanup.close()


def evaluate(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
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
    from .rl_diagnostics import EpisodeDiagnostics, action_summary
    action_rows = []
    rows = []
    started = time.perf_counter()
    progress = _EtaBar(f"{args.sim} evaluate", len(scenarios), enabled=args.eta_bar)
    try:
        for scenario in scenarios:
            observations = env.reset(scenario)
            cumulative = 0.
            episode = EpisodeDiagnostics(config.gamma)
            while True:
                with torch.no_grad():
                    action_row = {}
                    actions, _, _ = _action(model, observations, args.device, args.deterministic, diagnostics=action_row)
                    action_rows.append(action_row)
                observations, reward, done, truncated, info = env.step(actions)
                cumulative += sum(reward.values())
                episode.add(reward, info, env)
                if done or truncated:
                    rows.append({"scenario_seed": scenario.seed, "cumulative_reward": cumulative, **episode.result(info)})
                    progress.update(len(rows))
                    break
    finally:
        progress.close()
        env.close()
    result = {"simulator": args.sim, "checkpoint": str(args.checkpoint), "mode": args.mode,
              "source_run_id": checkpoint["manifest"].get("run_id"), "seed": args.seed,
              "deterministic": args.deterministic, "episodes": rows,
              "action_statistics": action_summary(action_rows),
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
        command.add_argument("--eta-bar", action="store_true", help="show progress and estimated time remaining on stderr")
    training.add_argument("--seed", type=int, default=11)
    training.add_argument("--total-steps", type=int, required=True)
    training.add_argument("--num-envs", type=int, default=1)
    training.add_argument("--checkpoint-dir", required=True)
    training.add_argument("--checkpoint-interval", type=int, required=True)
    training.add_argument("--update-interval", type=int, default=250)
    training.add_argument("--overwrite", action="store_true", help="archive existing run directory before a fresh run")
    training.add_argument("--config")
    training.add_argument("--headless", action="store_true")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--scenario-bank")
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--mode", choices=("nominal", "stress"), default="nominal")
    evaluation.add_argument("--deterministic", action="store_true")
    evaluation.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.total_steps < 1 or args.num_envs < 1 or args.checkpoint_interval < 1 or args.update_interval < 1:
            parser.error("Training counts must be positive")
        train(args)
    else:
        if args.episodes < 1:
            parser.error("Episode count must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
