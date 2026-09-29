"""Offline behavior cloning from any observation-only demonstration policy."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .evaluate import evaluate_bank
from .gym_env import NavigationGymEnv
from .runner import make_model, save_policy, write_json


@dataclass(frozen=True)
class DemonstrationSet:
    observations: np.ndarray
    actions: np.ndarray
    scenario_hashes: tuple[str, ...]


def collect_demonstrations(task, *, scenario_sampler: Callable,
                           controller_factory: Callable, seed: int,
                           episodes: int) -> DemonstrationSet:
    """Collect (local observation, action) pairs without simulator truth access."""
    if episodes < 1:
        raise ValueError("At least one demonstration episode is required")
    rng = np.random.default_rng(seed)
    observations, actions, hashes = [], [], []
    env = NavigationGymEnv(task)
    try:
        for _ in range(episodes):
            scenario = scenario_sampler(rng)
            controller = controller_factory()
            hashes.append(scenario.geometry_hash())
            obs, _ = env.reset(seed=scenario.seed, options={"scenario": scenario})
            for _step in range(task.deadline_steps + 1):
                action, _ = controller.predict(obs, deterministic=True)
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(np.asarray(action, dtype=np.float32).copy())
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            else:
                raise AssertionError("Demonstration exceeded task deadline")
    finally:
        env.close()
    return DemonstrationSet(np.stack(observations), np.stack(actions), tuple(hashes))


def behavior_clone(model, dataset: DemonstrationSet, *, epochs: int = 10,
                   batch_size: int = 256, learning_rate: float = 3e-4,
                   seed: int = 11) -> list[float]:
    """Fit only the policy branch; PPO starts afterward with its own optimizer."""
    if dataset.observations.ndim != 2 or dataset.actions.shape != (len(dataset.observations), 2):
        raise ValueError("Demonstration observation/action shapes disagree")
    if not np.isfinite(dataset.observations).all() or not np.isfinite(dataset.actions).all():
        raise ValueError("Demonstrations must be finite")
    if np.any(dataset.actions < -1) or np.any(dataset.actions > 1):
        raise ValueError("Demonstration actions exceed the declared action space")
    params = list(model.policy.mlp_extractor.policy_net.parameters()) + \
             list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    observations = torch.as_tensor(dataset.observations, device=model.device)
    actions = torch.as_tensor(dataset.actions, device=model.device)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        batch_losses = []
        for indices in np.array_split(rng.permutation(len(actions)),
                                      max(1, (len(actions) + batch_size - 1) // batch_size)):
            features = model.policy.extract_features(observations[indices])
            if isinstance(features, tuple):
                features = features[0]
            latent = model.policy.mlp_extractor.forward_actor(features)
            predicted = model.policy.action_net(latent).clamp(-1., 1.)
            loss = torch.mean((predicted - actions[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
    return losses


def build_bc_policy(settings, task, dataset: DemonstrationSet, output: Path,
                    dev_bank, *, epochs: int = 10, batch_size: int = 256,
                    learning_rate: float = 3e-4, seed: int = 11) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "demonstrations.npz", observations=dataset.observations,
                        actions=dataset.actions)
    dataset_sha = hashlib.sha256((output / "demonstrations.npz").read_bytes()).hexdigest()
    model, env = make_model({**settings, "seed": seed}, task)
    try:
        losses = behavior_clone(model, dataset, epochs=epochs, batch_size=batch_size,
                                learning_rate=learning_rate, seed=seed)
        save_policy(model, output / "bc-policy.zip", task)
        report = evaluate_bank(model, task, dev_bank, checkpoint_step=0)
        write_json(output / "bc-dev.json", report)
        manifest = {"method": "offline_behavior_cloning", "seed": seed,
                    "episodes": len(dataset.scenario_hashes),
                    "transitions": len(dataset.actions),
                    "dataset_sha256": dataset_sha,
                    "scenario_hashes": dataset.scenario_hashes,
                    "epochs": epochs, "batch_size": batch_size,
                    "learning_rate": learning_rate, "epoch_losses": losses,
                    "dev_bank_hash": report["scenario_bank_hash"],
                    "dev_success_count": report["success_count"],
                    "dev_collision_count": report["collision_count"]}
        write_json(output / "manifest.json", manifest)
        return manifest
    finally:
        env.close()
