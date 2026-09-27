"""Independent-actor PPO sanity trainer for BCOD single-vessel diagnostics.

The production four-vessel runner and legacy actor-critic are untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .audit_learning import scenario
from .core import Benchmark, BenchmarkConfig, NAMES
from .runner import make_adapter


class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(47, 128), nn.Tanh(), nn.Linear(128, 128),
                                   nn.Tanh(), nn.Linear(128, 2))
        self.critic = nn.Sequential(nn.Linear(47, 128), nn.Tanh(), nn.Linear(128, 128),
                                    nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.tensor([-0.7, math.log(0.10)]))

    def mean(self, obs):
        return self.actor(obs)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def transform(latent):
        squashed = torch.tanh(latent)
        return torch.stack(((squashed[..., 0] + 1) / 2, squashed[..., 1]), dim=-1)

    def log_probability(self, obs, latent):
        normal = torch.distributions.Normal(self.mean(obs), self.log_std.exp())
        log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (normal.log_prob(latent) - log_jacobian).sum(-1) - math.log(0.5)

    def sample(self, obs):
        normal = torch.distributions.Normal(self.mean(obs), self.log_std.exp())
        latent = normal.sample()
        return self.transform(latent), latent, self.log_probability(obs, latent)

    def latent_entropy(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp()).entropy().sum(-1)


def initialize_case_a(policy, observation, target_surge=0.05):
    """Calibrate only the diagnostic actor bias at the initial observation."""
    desired = torch.tensor([math.atanh(2 * target_surge - 1), 0.], dtype=torch.float32)
    with torch.no_grad():
        current = policy.mean(torch.as_tensor(observation, dtype=torch.float32)[None])[0]
        policy.actor[-1].bias.add_(desired - current)


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Timeouts bootstrap their final observation; neither boundary crosses a reset."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    next_values = torch.as_tensor(next_values, dtype=torch.float32)
    terminated = torch.as_tensor(terminated, dtype=torch.bool)
    truncated = torch.as_tensor(truncated, dtype=torch.bool)
    if not (rewards.shape == values.shape == next_values.shape == terminated.shape == truncated.shape):
        raise ValueError("GAE arrays must have the same shape")
    advantage = torch.empty_like(rewards)
    following = torch.zeros((), dtype=torch.float32)
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        following = delta + gamma * lam * following * (~(terminated[t] | truncated[t]))
        advantage[t] = following
    return advantage, advantage + values


def ppo_update(policy, actor_optimizer, critic_optimizer, batch, *, epochs, minibatch_size,
               clip_epsilon, entropy_coefficient, value_coefficient, max_grad_norm):
    observations, latents, old_logp, returns, advantages = batch
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    records = []
    for _ in range(epochs):
        for indices in torch.randperm(len(observations)).split(minibatch_size):
            obs, latent = observations[indices], latents[indices]
            ratio = torch.exp(policy.log_probability(obs, latent) - old_logp[indices])
            clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.minimum(ratio * advantages[indices],
                                         clipped * advantages[indices]).mean()
            entropy = policy.latent_entropy(obs).mean()
            actor_optimizer.zero_grad()
            (policy_loss - entropy_coefficient * entropy).backward()
            actor_norm = float(torch.nn.utils.clip_grad_norm_(
                list(policy.actor.parameters()) + [policy.log_std], max_grad_norm))
            actor_optimizer.step()

            predicted = policy.value(obs)
            value_loss = 0.5 * (predicted - returns[indices]).square().mean()
            critic_optimizer.zero_grad()
            (value_coefficient * value_loss).backward()
            critic_norm = float(torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), max_grad_norm))
            critic_optimizer.step()
            with torch.no_grad():
                log_ratio = policy.log_probability(obs, latent) - old_logp[indices]
                records.append({
                    "policy_loss": float(policy_loss), "value_loss": float(value_loss),
                    "latent_entropy": float(entropy),
                    "approx_kl": float(((log_ratio.exp() - 1) - log_ratio).mean()),
                    "clip_fraction": float(((ratio - 1).abs() > clip_epsilon).float().mean()),
                    "actor_gradient_norm_before_clipping": actor_norm,
                    "critic_gradient_norm_before_clipping": critic_norm})
    with torch.no_grad():
        prediction = policy.value(observations)
        explained_variance = 1 - float((returns - prediction).var(unbiased=False) /
                                       (returns.var(unbiased=False) + 1e-8))
    result = {k: float(np.mean([row[k] for row in records])) for k in records[0]}
    result["explained_variance"] = explained_variance
    result["actor_gradient_norm_after_clipping"] = min(result["actor_gradient_norm_before_clipping"], max_grad_norm)
    result["critic_gradient_norm_after_clipping"] = min(result["critic_gradient_norm_before_clipping"], max_grad_norm)
    return result


@contextmanager
def preserve_training_rng():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def evaluate(policy, env, kind, step, count=3):
    rows = []
    with preserve_training_rng(), torch.no_grad():
        for index in range(count):
            obs = env.reset(scenario(kind, 700000 + index))
            surge, yaw, length = [], [], 0
            while True:
                action = policy.transform(policy.mean(torch.as_tensor(
                    obs[NAMES[0]], dtype=torch.float32)[None]))[0]
                command = (float(action[0]), float(action[1]))
                actions = {name: (0., 0.) for name in NAMES}
                actions[NAMES[0]] = command
                obs, _, done, truncated, info = env.step(actions)
                surge.append(command[0]); yaw.append(command[1]); length += 1
                if done or truncated:
                    break
            position = env.truth[NAMES[0]]
            rows.append({"seed": 700000 + index,
                         "success": bool(info["per_agent_success"][NAMES[0]] and not info["fleet_failure_due_to_collision"]),
                         "collision": NAMES[0] in info["physical_colliders"],
                         "final_distance_m": math.dist((position.x_m, position.y_m), env.scenario.goals[0]),
                         "episode_length": length, "rollout_mean_surge": float(np.mean(surge)),
                         "rollout_mean_signed_yaw": float(np.mean(yaw)),
                         "rollout_mean_abs_yaw": float(np.mean(np.abs(yaw)))})
    return {"environment_steps": step, "scenario_rows": rows,
            "success_count": sum(row["success"] for row in rows),
            "collision_count": sum(row["collision"] for row in rows),
            "mean_final_distance_m": float(np.mean([row["final_distance_m"] for row in rows])),
            "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
            "deterministic_surge": float(np.mean([row["rollout_mean_surge"] for row in rows])),
            "deterministic_signed_yaw": float(np.mean([row["rollout_mean_signed_yaw"] for row in rows])),
            "deterministic_abs_yaw": float(np.mean([row["rollout_mean_abs_yaw"] for row in rows])),
            "raw_std": policy.log_std.exp().detach().tolist(),
            "action_statistics_basis": "deterministic evaluation rollout average"}


def train(args):
    if args.steps != 10000 or args.rollout_horizon != 250:
        raise ValueError("First PPO sanity run requires 10,000 steps and 250-step rollouts")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    config = BenchmarkConfig()
    env = Benchmark(make_adapter("bcod-reduced", config), config, diagnostic_allow_sparse=True)
    eval_env = Benchmark(make_adapter("bcod-reduced", config), config, diagnostic_allow_sparse=True)
    policy = PPOPolicy()
    actor_optimizer = torch.optim.Adam(list(policy.actor.parameters()) + [policy.log_std], lr=args.learning_rate)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=args.learning_rate)
    manifest = {"algorithm": "standard_single_agent_PPO", "simulator": "bcod-reduced",
                "learning_agent": NAMES[0], "task": "Case A: straight goal, no obstacles",
                "seed": args.seed, "steps": args.steps, "dt_s": config.dt_s, "gamma": config.gamma,
                "rollout_horizon": args.rollout_horizon, "gae_lambda": args.gae_lambda,
                "clip_epsilon": args.clip_epsilon, "optimization_epochs": args.optimization_epochs,
                "minibatch_size": args.minibatch_size, "entropy_coefficient": args.entropy_coefficient,
                "entropy_definition": "latent Gaussian entropy", "value_coefficient": args.value_coefficient,
                "max_grad_norm": args.max_grad_norm, "learning_rate": args.learning_rate,
                "actor_architecture": "47-128-128-2", "critic_architecture": "47-128-128-1",
                "initial_target_deterministic_surge": 0.05, "initial_target_deterministic_yaw": 0.,
                "initial_raw_yaw_std": 0.10, "initial_raw_surge_std": math.exp(-0.7),
                "terminal_convention": "true terminal zero bootstrap; timeout final-observation bootstrap; GAE reset at either episode boundary"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        obs = env.reset(scenario("A", 1100000))
        initialize_case_a(policy, obs[NAMES[0]])
        evaluations, updates = [], []
        def checkpoint(step):
            result = evaluate(policy, eval_env, "A", step, args.eval_count)
            result["completed_rollouts"] = len(updates)
            evaluations.append(result)
            (output / "evaluations.json").write_text(json.dumps(evaluations, indent=2) + "\n")
            torch.save({"policy": policy.state_dict(), "environment_steps": step, "manifest": manifest},
                       output / f"checkpoint-{step}.pt")
            print(f"PPO Case A {step}: {result['success_count']}/{args.eval_count} success, "
                  f"distance {result['mean_final_distance_m']:.2f} m, "
                  f"surge {result['deterministic_surge']:.3f}, yaw {result['deterministic_signed_yaw']:.3f}", flush=True)
        checkpoint(0)
        episode_index = 0
        for start in range(0, args.steps, args.rollout_horizon):
            collected = []
            for offset in range(args.rollout_horizon):
                own = torch.as_tensor(obs[NAMES[0]], dtype=torch.float32)
                with torch.no_grad():
                    action, latent, old_logp = policy.sample(own[None])
                    value = policy.value(own[None])[0]
                commands = {name: (0., 0.) for name in NAMES}
                commands[NAMES[0]] = tuple(float(v) for v in action[0])
                next_obs, reward, done, truncated, info = env.step(commands)
                assert set(info["physical_colliders"]).issubset({NAMES[0]})
                assert not done or info["per_agent_success"][NAMES[0]] or NAMES[0] in info["physical_colliders"]
                with torch.no_grad():
                    next_value = policy.value(torch.as_tensor(next_obs[NAMES[0]], dtype=torch.float32)[None])[0]
                collected.append((own, latent[0], old_logp[0], value,
                                  reward[NAMES[0]], next_value, done, truncated))
                obs = next_obs
                if done or truncated:
                    episode_index += 1
                    obs = env.reset(scenario("A", 1100000 + episode_index))
            assert len(collected) == args.rollout_horizon
            observations = torch.stack([row[0] for row in collected])
            latents = torch.stack([row[1] for row in collected])
            old_logp = torch.stack([row[2] for row in collected])
            advantages, returns = gae([row[4] for row in collected], [row[3] for row in collected],
                                      [row[5] for row in collected], [row[6] for row in collected],
                                      [row[7] for row in collected], config.gamma, args.gae_lambda)
            result = ppo_update(policy, actor_optimizer, critic_optimizer,
                                (observations, latents, old_logp, returns.detach(), advantages.detach()),
                                epochs=args.optimization_epochs, minibatch_size=args.minibatch_size,
                                clip_epsilon=args.clip_epsilon, entropy_coefficient=args.entropy_coefficient,
                                value_coefficient=args.value_coefficient, max_grad_norm=args.max_grad_norm)
            result.update({"environment_steps": start + args.rollout_horizon,
                           "optimizer_minibatches": args.optimization_epochs * math.ceil(args.rollout_horizon / args.minibatch_size),
                           "raw_std": policy.log_std.exp().detach().tolist(),
                           "sampled_surge_mean": float(np.mean([float(policy.transform(row[1])[0]) for row in collected])),
                           "sampled_abs_yaw_mean": float(np.mean([abs(float(policy.transform(row[1])[1])) for row in collected]))})
            updates.append(result)
            (output / "updates.json").write_text(json.dumps(updates, indent=2) + "\n")
            if start + args.rollout_horizon in (1000, 2500, 5000, 7500, 10000):
                checkpoint(start + args.rollout_horizon)
        assert len(updates) == 40
    finally:
        env.close(); eval_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--rollout-horizon", type=int, default=250)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--optimization-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-count", type=int, default=3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
