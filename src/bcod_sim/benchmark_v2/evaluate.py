"""Deterministic, separate-environment evaluation for navigation V2.

Evaluation never uses a training environment and restores process RNG states even
when a rollout fails.  The scenario geometry hash deliberately excludes its seed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np

from .scenarios import Scenario


@contextmanager
def preserve_training_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        import torch
    except ImportError:  # evaluation itself does not require a Torch import
        torch = None
    torch_state = torch.get_rng_state().clone() if torch is not None else None
    cuda_states = torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _record(value):
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {key: getattr(value, key) for key in ("x_m", "y_m", "heading_rad", "surge_mps", "yaw_rps")
            if hasattr(value, key)}


def _agent(info, field):
    singular = {"truth": "truth", "readings": "reading"}.get(field)
    if singular is not None and singular in info:
        return _record(info[singular])
    values = info.get(field, {})
    if isinstance(values, dict):
        return _record(values.get("vessel_0"))
    return None


def _pose(info):
    pose = _agent(info, "truth")
    if pose is None:
        raise ValueError("Evaluation info must include vessel_0 scoring truth")
    return pose


def _distance(pose, goal):
    return math.hypot(pose["x_m"] - goal[0], pose["y_m"] - goal[1])


def _heading_error(pose, goal):
    desired = math.atan2(goal[1] - pose["y_m"], goal[0] - pose["x_m"])
    return abs(math.atan2(math.sin(desired - pose["heading_rad"]),
                          math.cos(desired - pose["heading_rad"])))


def evaluate_scenario(model, env, scenario: Scenario, *, checkpoint_step: int = 0,
                      trajectory_path: str | Path | None = None) -> dict:
    """Roll out one explicit scenario using actor means only.

    ``env`` must be a distinct evaluation environment, owned by the caller.
    Its reset accepts an explicit scenario.  No auto-reset or training update occurs.
    """
    if len(scenario.agent_ids) != 1:
        raise ValueError("V2 single-agent evaluation requires one vessel")
    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be nonnegative")
    with preserve_training_rng():
        obs, info = env.reset(seed=scenario.seed, options={"scenario": scenario})
        pose = _pose(info)
        goal = scenario.goals[0]
        initial_distance = _distance(pose, goal)
        minimum_distance = initial_distance
        path_length = 0.0
        total_reward = 0.0
        rewards = {}
        speeds = []
        heading_errors = []
        requested_yaws = []
        achieved_yaws = []
        commands = []
        trajectory = [{"step": 0, "truth": pose, "reading": _agent(info, "readings"),
                       "distance_to_goal_m": initial_distance}]
        done = False
        terminated = truncated = False
        steps = 0
        while not done:
            predicted = model.predict(obs, deterministic=True)
            action = np.asarray(predicted[0] if isinstance(predicted, tuple) else predicted,
                                dtype=np.float32).reshape(-1)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("Model produced an invalid two-component action")
            obs, reward, terminated, truncated, info = env.step(action)
            next_pose = _pose(info)
            reading = _agent(info, "readings")
            steps += 1
            path_length += math.hypot(next_pose["x_m"] - pose["x_m"],
                                      next_pose["y_m"] - pose["y_m"])
            distance = _distance(next_pose, goal)
            minimum_distance = min(minimum_distance, distance)
            total_reward += float(reward)
            components = info.get("reward_components", {})
            if "vessel_0" in components:
                components = components["vessel_0"]
            for component, value in components.items():
                rewards[component] = rewards.get(component, 0.0) + float(value)
            if reading is not None:
                speeds.append(float(reading.get("surge_mps", 0.0)))
                achieved_yaws.append(float(reading.get("yaw_rps", 0.0)))
            heading_errors.append(_heading_error(next_pose, goal))
            requested_yaws.append(float(action[1]) * env.config.max_yaw_rps)
            commands.append(action.tolist())
            trajectory.append({"step": steps, "truth": next_pose, "reading": reading,
                               "action": action.tolist(), "reward": float(reward),
                               "reward_components": components,
                               "distance_to_goal_m": distance})
            pose = next_pose
            done = bool(terminated or truncated)
            if steps > env.config.deadline_steps + 1:
                raise RuntimeError("Evaluation episode exceeded the task deadline")
        reason = info.get("terminal_reason") or info.get("reason")
        if reason is None:
            reason = "external_truncation" if truncated else "unknown"
        report = {
            "checkpoint_step": checkpoint_step, "scenario_seed": scenario.seed,
            "scenario_split": scenario.split, "geometry_hash": scenario.geometry_hash(),
            "terminal_reason": reason, "success": reason == "success",
            "collision": reason == "collision", "deadline": reason == "deadline",
            "terminated": bool(terminated), "truncated": bool(truncated),
            "initial_distance_to_goal_m": initial_distance,
            "final_distance_to_goal_m": _distance(pose, goal),
            "minimum_distance_to_goal_m": minimum_distance,
            "episode_length": steps, "path_length_m": path_length,
            "mean_speed_mps": float(np.mean(speeds)) if speeds else None,
            "mean_heading_error_rad": float(np.mean(heading_errors)),
            "mean_commanded_yaw_rate_radps": float(np.mean(requested_yaws)),
            "mean_achieved_yaw_rate_radps": float(np.mean(achieved_yaws)) if achieved_yaws else None,
            "mean_action_surge": float(np.mean([a[0] for a in commands])),
            "mean_action_yaw": float(np.mean([a[1] for a in commands])),
            "mean_absolute_action_yaw": float(np.mean([abs(a[1]) for a in commands])),
            "total_reward": total_reward, "reward_components": rewards,
        }
        if trajectory_path is not None:
            path = Path(trajectory_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"report": report, "trajectory": trajectory}, indent=2) + "\n")
            report["trajectory_path"] = str(path)
        return report


def evaluate_bank(model, env_factory: Callable, scenarios: Iterable[Scenario], *,
                  checkpoint_step: int = 0, trajectory_dir: str | Path | None = None) -> dict:
    """Evaluate distinct geometries, constructing and closing one separate env."""
    cases = tuple(scenarios)
    hashes = [case.geometry_hash() for case in cases]
    if not cases or len(set(hashes)) != len(cases):
        raise ValueError("Evaluation bank is empty or repeats scenario geometry")
    reports = []
    with preserve_training_rng():
        env = env_factory()
        try:
            for index, case in enumerate(cases):
                path = None
                if trajectory_dir is not None:
                    path = Path(trajectory_dir) / f"step-{checkpoint_step}-{case.split}-{index}.json"
                reports.append(evaluate_scenario(model, env, case, checkpoint_step=checkpoint_step,
                                                 trajectory_path=path))
        finally:
            env.close()
    return {
        "checkpoint_step": checkpoint_step, "scenario_count": len(cases),
        "scenario_bank_hash": hashlib.sha256(
            json.dumps(hashes, separators=(",", ":")).encode()).hexdigest(),
        "success_count": sum(r["success"] for r in reports),
        "collision_count": sum(r["collision"] for r in reports),
        "deadline_count": sum(r["deadline"] for r in reports),
        "mean_final_distance_to_goal_m": float(np.mean([r["final_distance_to_goal_m"] for r in reports])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in reports])),
        "mean_reward": float(np.mean([r["total_reward"] for r in reports])),
        "episodes": reports,
    }
