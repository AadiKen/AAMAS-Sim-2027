"""Isolated Stable-Baselines3 PPO entry point for Navigation V2."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import uuid

import gymnasium
from gymnasium.utils.env_checker import check_env as gym_check_env
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import yaml

from .config import TaskConfig
from .evaluate import evaluate_bank, evaluate_scenario
from .gym_env import ACTION_VERSION, NavigationGymEnv
from .scenarios import case_a_reference, case_a_validation

DEFAULT_CONFIG = Path("configs/benchmark_v2/case_a_sb3.yaml")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _read_config(path: Path) -> tuple[dict, TaskConfig]:
    settings = yaml.safe_load(path.read_text())
    task = TaskConfig(**settings["task"])
    if settings["algorithm"] != "sb3_ppo" or settings["policy"] != "MlpPolicy":
        raise ValueError("V2 runner requires SB3 PPO with MlpPolicy")
    if settings["n_envs"] != 1 or task.agent_count != 1:
        raise ValueError("V2 runner requires exactly one physical vessel and one environment")
    if settings["n_steps"] <= 0 or settings["batch_size"] <= 0 or settings["n_epochs"] <= 0:
        raise ValueError("Invalid PPO rollout/batch/epoch settings")
    return settings, task


def _versions() -> dict:
    names = ("bcod-sim", "stable-baselines3", "gymnasium", "torch", "numpy", "PyYAML")
    return {"python": sys.version.split()[0], **{n: metadata.version(n) for n in names}}


def _git_revision() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _make_model(settings: dict, task: TaskConfig):
    fixture = case_a_reference(settings["seed"])
    env = DummyVecEnv([lambda: NavigationGymEnv(task, scenario=fixture)])
    model = PPO(
        "MlpPolicy", env, seed=settings["seed"], device=settings["device"],
        n_steps=settings["n_steps"], batch_size=settings["batch_size"],
        n_epochs=settings["n_epochs"], learning_rate=settings["learning_rate"],
        gamma=task.gamma, gae_lambda=settings["gae_lambda"],
        clip_range=settings["clip_range"], ent_coef=settings["ent_coef"],
        vf_coef=settings["vf_coef"], max_grad_norm=settings["max_grad_norm"],
        verbose=1,
    )
    return model, env


class EpisodeLogger(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                record = {"environment_steps": self.num_timesteps,
                          "terminal_reason": info.get("terminal_reason"),
                          "episode_length": info.get("steps"),
                          "final_goal_distance_m": info.get("final_goal_distance_m", {}).get("vessel_0"),
                          "path_length_m": info.get("path_length_m", {}).get("vessel_0"),
                          "reward_components": info.get("reward_components", {}).get("vessel_0")}
                with self.path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
        return True


def _evaluate(model, task: TaskConfig, seed: int, step: int, run_dir: Path) -> dict:
    fixture = case_a_reference(seed)
    env = NavigationGymEnv(task)
    try:
        reference = evaluate_scenario(model, env, fixture, checkpoint_step=step,
            trajectory_path=run_dir / "trajectories" / f"reference-step-{step}.json")
    finally:
        env.close()
    bank = evaluate_bank(model, lambda: NavigationGymEnv(task), case_a_validation(seed),
                         checkpoint_step=step, trajectory_dir=run_dir / "trajectories" / f"validation-step-{step}")
    result = {"checkpoint_step": step, "reference": reference, "validation": bank}
    _write_json(run_dir / "evaluations" / f"step-{step}.json", result)
    with (run_dir / "evaluations.jsonl").open("a") as stream:
        stream.write(json.dumps({"checkpoint_step": step,
            "reference_success": reference["success"],
            "validation_success_count": bank["success_count"],
            "validation_mean_reward": bank["mean_reward"]}) + "\n")
    return result


def _status(model, settings, task, *, state: str, requested: int) -> dict:
    steps = int(model.num_timesteps)
    return {"state": state, "requested_environment_steps": requested,
            "actual_environment_steps": steps, "vector_steps": steps,
            "agent_transitions": steps * task.agent_count,
            "completed_rollouts": steps // settings["n_steps"],
            "optimizer_epochs": int(model._n_updates),
            "task_agent_count": task.agent_count,
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}


def _train_chunks(model, settings, task, run_dir, target_steps):
    callback = EpisodeLogger(run_dir / "episodes.jsonl")
    evaluation_interval = int(settings["evaluation_interval_steps"])
    best_reward = None
    previous_eval = sorted((run_dir / "evaluations").glob("step-*.json"))
    for path in previous_eval:
        value = json.loads(path.read_text())["validation"]["mean_reward"]
        best_reward = value if best_reward is None else max(best_reward, value)
    try:
        while model.num_timesteps < target_steps:
            chunk = min(evaluation_interval, target_steps - model.num_timesteps)
            if chunk % settings["n_steps"]:
                raise ValueError("Training target and evaluation interval must align with PPO rollout length")
            model.learn(total_timesteps=chunk, callback=callback, reset_num_timesteps=False,
                        progress_bar=False)
            step = int(model.num_timesteps)
            model.save(run_dir / "checkpoints" / f"step-{step}.zip")
            model.save(run_dir / "latest.zip")
            result = _evaluate(model, task, settings["seed"], step, run_dir)
            score = result["validation"]["mean_reward"]
            if best_reward is None or score > best_reward:
                best_reward = score
                model.save(run_dir / "best_by_evaluation_reward.zip")
            _write_json(run_dir / "status.json", _status(model, settings, task,
                        state="running" if step < target_steps else "complete", requested=target_steps))
    except BaseException as exc:
        model.save(run_dir / "latest.zip")
        status = _status(model, settings, task, state="interrupted" if isinstance(exc, KeyboardInterrupt)
                         else "failed", requested=target_steps)
        status["error"] = repr(exc)
        _write_json(run_dir / "status.json", status)
        raise


def train(config_path: Path, run_dir: Path, steps: int | None = None, seed: int | None = None) -> dict:
    settings, task = _read_config(config_path)
    if seed is not None:
        settings["seed"] = seed
    target = int(steps or settings["total_timesteps"])
    if target <= 0 or target % settings["n_steps"]:
        raise ValueError("Step count must be a positive multiple of n_steps")
    if settings["evaluation_interval_steps"] % settings["n_steps"]:
        raise ValueError("Evaluation interval must align with n_steps")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "checkpoints", "evaluations", "trajectories"):
        (run_dir / name).mkdir()
    _write_json(run_dir / "dependency-versions.json", _versions())
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(settings, sort_keys=True))
    lock_path = Path("configs/benchmark_v2/requirements.lock")
    manifest = {"run_id": str(uuid.uuid4()), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "label": "Stable-Baselines3 PPO Navigation V2", "algorithm": "sb3_ppo",
                "source_config": str(config_path), "code_revision": _git_revision(),
                "dependencies": _versions(),
                "dependency_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                "task": asdict(task), "action_version": ACTION_VERSION,
                "observation_version": "legacy47-plus-time-v2", "action_space": "Box(-1,1,shape=(2,))",
                "observation_space": "Box(-1,1,shape=(48,))",
                "reference_geometry_hash": case_a_reference(settings["seed"]).geometry_hash(),
                "validation_geometry_hashes": [s.geometry_hash() for s in case_a_validation(settings["seed"])],
                "scenario_bank_hash": hashlib.sha256(json.dumps(
                    [s.geometry_hash() for s in case_a_validation(settings["seed"])],
                    separators=(",", ":")).encode()).hexdigest(),
                "reward_terminal_convention": "Phi(terminal)=0; deadline is terminated",
                "gamma_per_step": task.gamma, "ppo": {k: settings[k] for k in (
                    "n_steps", "batch_size", "n_epochs", "learning_rate", "gae_lambda", "clip_range",
                    "ent_coef", "vf_coef", "max_grad_norm", "n_envs", "policy", "device")},
                "configured_total_timesteps": settings["total_timesteps"],
                "requested_timesteps": target, "seed": settings["seed"]}
    _write_json(run_dir / "manifest.json", manifest)
    model, env = _make_model(settings, task)
    model.set_logger(configure(str(run_dir / "logs"), ["stdout", "csv"]))
    try:
        _evaluate(model, task, settings["seed"], 0, run_dir)
        _write_json(run_dir / "status.json", _status(model, settings, task, state="running", requested=target))
        _train_chunks(model, settings, task, run_dir, target)
        return json.loads((run_dir / "status.json").read_text())
    finally:
        env.close()


def resume(run_dir: Path, additional_steps: int) -> dict:
    settings, task = _read_config(run_dir / "config.resolved.yaml")
    if additional_steps <= 0 or additional_steps % settings["n_steps"]:
        raise ValueError("additional_steps must be a positive multiple of n_steps")
    if not (run_dir / "latest.zip").exists():
        raise FileNotFoundError("No latest checkpoint to resume")
    env = DummyVecEnv([lambda: NavigationGymEnv(task, scenario=case_a_reference(settings["seed"]))])
    model = PPO.load(run_dir / "latest.zip", env=env, device=settings["device"])
    model.set_logger(configure(str(run_dir / "logs"), ["stdout", "csv"]))
    target = int(model.num_timesteps) + additional_steps
    with (run_dir / "continuations.jsonl").open("a") as stream:
        stream.write(json.dumps({"from_environment_steps": int(model.num_timesteps),
                                 "additional_steps": additional_steps,
                                 "boundary": "environment reset; model and optimizer restored"}) + "\n")
    try:
        _train_chunks(model, settings, task, run_dir, target)
        return json.loads((run_dir / "status.json").read_text())
    finally:
        env.close()


def check(seed: int, config_path: Path = DEFAULT_CONFIG) -> dict:
    _, task = _read_config(config_path)
    gym_env = NavigationGymEnv(task, scenario=case_a_reference(seed))
    sb3_env = NavigationGymEnv(task, scenario=case_a_reference(seed))
    try:
        gym_check_env(gym_env, skip_render_check=True)
        sb3_check_env(sb3_env, warn=True)
    finally:
        gym_env.close()
        sb3_env.close()
    scripted = NavigationGymEnv(task, scenario=case_a_reference(seed))
    try:
        scripted.reset(seed=seed)
        for step in range(task.deadline_steps):
            _, _, terminated, truncated, info = scripted.step([0.0, 0.0])
            if terminated or truncated:
                break
        baseline = {"terminal_reason": info["terminal_reason"], "episode_length": step + 1,
                    "final_distance_m": info["final_goal_distance_m"]["vessel_0"]}
    finally:
        scripted.close()
    return {"gymnasium_check": "passed", "sb3_check": "passed", "scripted_baseline": baseline}


def summarize(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    evaluations = []
    for path in sorted((run_dir / "evaluations").glob("step-*.json"),
                       key=lambda p: int(p.stem.split("-")[-1])):
        record = json.loads(path.read_text())
        evaluations.append({"step": record["checkpoint_step"],
                            "reference_success": record["reference"]["success"],
                            "validation_success_count": record["validation"]["success_count"],
                            "validation_mean_reward": record["validation"]["mean_reward"]})
    return {"run_id": manifest["run_id"], "status": status, "evaluations": evaluations,
            "artifacts": {"manifest": str(run_dir / "manifest.json"),
                          "progress": str(run_dir / "logs" / "progress.csv"),
                          "episodes": str(run_dir / "episodes.jsonl"),
                          "latest_checkpoint": str(run_dir / "latest.zip"),
                          "best_by_evaluation_reward": str(run_dir / "best_by_evaluation_reward.zip"),
                          "trajectories": str(run_dir / "trajectories")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stable-Baselines3 PPO Navigation V2")
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check")
    p_check.add_argument("--seed", type=int, default=11)
    p_check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_train = sub.add_parser("train")
    p_train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_train.add_argument("--run-dir", type=Path, required=True)
    p_train.add_argument("--total-timesteps", "--steps", dest="steps", type=int)
    p_train.add_argument("--seed", type=int)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--run-dir", type=Path, required=True)
    p_eval.add_argument("--checkpoint", type=Path)
    p_resume = sub.add_parser("resume")
    p_resume.add_argument("--run-dir", type=Path, required=True)
    p_resume.add_argument("--additional-timesteps", type=int, required=True)
    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check(args.seed, args.config)
    elif args.command == "train":
        result = train(args.config, args.run_dir, args.steps, args.seed)
    elif args.command == "resume":
        result = resume(args.run_dir, args.additional_timesteps)
    elif args.command == "evaluate":
        settings, task = _read_config(args.run_dir / "config.resolved.yaml")
        model = PPO.load(args.checkpoint or args.run_dir / "latest.zip", device=settings["device"])
        result = _evaluate(model, task, settings["seed"], int(model.num_timesteps), args.run_dir)
    else:
        result = summarize(args.run_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
