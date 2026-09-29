"""Stock SB3 PPO only: versioned S0 navigation training and evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml

from . import ACTION_VERSION, OBSERVATION_VERSION, TASK_VERSION
from .config import TaskConfig
from .control.actions import ACTION_SCHEMAS
from .control.high_level import CONTROLLER_VERSION
from .evaluate import evaluate_bank, task_metric_key
from .gym_env import NavigationGymEnv
from .observations import FIELDS, SCHEMA_HASH
from .scenarios import case_a_reference, sample_s1, s0_validation_bank, s1_validation_bank
from .scripted import GeometricController

DEFAULT_CONFIG = Path("configs/benchmark_v3/s0_ppo.yaml")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def read_config(path: Path, backend: str | None = None, seed: int | None = None,
                action_mode: str | None = None):
    settings = yaml.safe_load(path.read_text())
    policy_action = settings.setdefault("policy_action", {"mode": "low_level"})
    if action_mode is not None:
        policy_action["mode"] = action_mode.replace("-", "_")
    settings["task"]["action_mode"] = policy_action["mode"]
    for key in ("max_heading_offset_rad", "heading_k_p"):
        if key in policy_action:
            settings["task"][key] = policy_action[key]
    if backend is not None:
        settings["task"]["dynamics"] = backend
    if seed is not None:
        settings["seed"] = seed
    task = TaskConfig(**settings["task"])
    if settings.get("scenario_stage", "S0") not in {"S0", "S1"}:
        raise ValueError("Unsupported V3 scenario stage")
    if settings["algorithm"] != "sb3_ppo" or settings["policy"] != "MlpPolicy" or settings["n_envs"] not in (1, 8):
        raise ValueError("V3 single-agent training requires stock SB3 PPO and 1 or 8 environments")
    if settings["validation_count"] < 50:
        raise ValueError("S0 held-out bank must contain at least 50 geometries")
    return settings, task


def versions():
    return {"python": sys.version.split()[0], **{name: metadata.version(name) for name in
        ("bcod-sim", "stable-baselines3", "gymnasium", "torch", "numpy", "PyYAML")}}


def revision():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def schema_record(task: TaskConfig):
    return {"task_version": TASK_VERSION, "observation_version": OBSERVATION_VERSION,
            "action_version": ACTION_SCHEMAS[task.action_mode],
            "action_mode": task.action_mode,
            "action_schema_version": ACTION_SCHEMAS[task.action_mode],
            "heading_controller_version": CONTROLLER_VERSION,
            "heading_controller": {"k_p": task.heading_k_p,
                                   "max_heading_offset_rad": task.max_heading_offset_rad},
            "observation_schema_hash": SCHEMA_HASH,
            "observation_shape": [len(FIELDS)], "action_shape": [2],
            "normalization": {"half_width_m": task.half_width_m,
                              "visibility_m": task.visibility_m,
                              "max_surge_mps": task.max_surge_mps,
                              "max_yaw_rps": task.max_yaw_rps,
                              "deadline_steps": task.deadline_steps}}


def save_policy(model, path: Path, task: TaskConfig):
    model.save(path)
    write_json(path.with_suffix(".schema.json"), schema_record(task))


def load_policy(path: Path, task: TaskConfig, *, env=None):
    sidecar = path.with_suffix(".schema.json")
    if not sidecar.exists():
        raise ValueError("Checkpoint has no action schema metadata; explicit migration required")
    actual = json.loads(sidecar.read_text())
    expected = schema_record(task)
    if actual.get("action_schema_version", actual.get("action_version")) != expected["action_schema_version"]:
        raise ValueError(f"Action schema mismatch: checkpoint={actual.get('action_version')} environment={expected['action_schema_version']}")
    if actual != expected:
        raise ValueError("Checkpoint is not a compatible V3 policy bundle; explicit migration required")
    model = PPO.load(path, env=env, device="cpu")
    if tuple(model.observation_space.shape) != (len(FIELDS),) or tuple(model.action_space.shape) != (2,):
        raise ValueError("Checkpoint spaces do not match V3 schema")
    return model


def make_model(settings, task):
    env = DummyVecEnv([lambda: NavigationGymEnv(task, scenario_stage=settings.get("scenario_stage", "S0"))
                       for _ in range(settings["n_envs"])])
    model = PPO("MlpPolicy", env, seed=settings["seed"], device=settings["device"],
                n_steps=settings["n_steps"], batch_size=settings["batch_size"],
                n_epochs=settings["n_epochs"], learning_rate=settings["learning_rate"],
                gamma=task.gamma, gae_lambda=settings["gae_lambda"],
                clip_range=settings["clip_range"], ent_coef=settings["ent_coef"],
                vf_coef=settings["vf_coef"], max_grad_norm=settings["max_grad_norm"], verbose=0)
    return model, env


class EpisodeLogger(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                row = {"environment_steps": self.num_timesteps,
                       "scenario_hash": info.get("scenario_hash"),
                       "reason": info.get("terminal_reason"), "episode_length": info.get("steps"),
                       "final_goal_distance_m": info.get("final_goal_distance_m", {}).get("vessel_0"),
                       "episode_return": info.get("episode_returns", {}).get("vessel_0"),
                       "reward_components": info.get("episode_reward_components", {}).get("vessel_0")}
                with self.path.open("a") as file:
                    file.write(json.dumps(row) + "\n")
        return True


def evaluate_checkpoint(model, task, settings, run_dir, step):
    bank = s0_validation_bank if settings.get("scenario_stage", "S0") == "S0" else s1_validation_bank
    cases = bank(settings["validation_count"], seed=settings["validation_seed"])
    report = evaluate_bank(model, task, cases, checkpoint_step=step,
                           trajectory_dir=run_dir / "trajectories")
    write_json(run_dir / "evaluations" / f"step-{step}.json", report)
    with (run_dir / "evaluations.jsonl").open("a") as file:
        file.write(json.dumps({k: v for k, v in report.items() if k != "episodes"}) + "\n")
    return report


def status(model, settings, requested, state):
    steps = int(model.num_timesteps)
    return {"state": state, "requested_environment_steps": requested,
            "actual_environment_steps": steps, "vector_steps": steps // settings["n_envs"],
            "agent_transitions": steps,
            "rollout_transitions": settings["n_steps"] * settings["n_envs"],
            "completed_rollouts": steps // (settings["n_steps"] * settings["n_envs"]),
            "optimization_epochs": int(model._n_updates),
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}


def train_one(config_path: Path, run_dir: Path, *, seed=None, backend=None, total_timesteps=None,
              action_mode=None):
    settings, task = read_config(config_path, backend=backend, seed=seed, action_mode=action_mode)
    target = int(total_timesteps or settings["total_timesteps"])
    interval = int(settings["evaluation_interval_steps"])
    rollout_transitions = settings["n_envs"] * settings["n_steps"]
    if target <= 0 or target % rollout_transitions or interval % rollout_transitions:
        raise ValueError("Training/evaluation steps must align with complete PPO rollouts")
    run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("logs", "checkpoints", "evaluations", "trajectories"):
        (run_dir / folder).mkdir()
    stage = settings.get("scenario_stage", "S0")
    bank = (s0_validation_bank if stage == "S0" else s1_validation_bank)(
        settings["validation_count"], seed=settings["validation_seed"])
    bank_hash = hashlib.sha256(json.dumps([c.geometry_hash() for c in bank],
                                          separators=(",", ":")).encode()).hexdigest()
    lock = Path("configs/benchmark_v3/requirements.lock")
    manifest = {"run_id": str(uuid.uuid4()), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_revision": revision(), "dependencies": versions(),
                "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "backend": task.dynamics, "task": asdict(task), "schema": schema_record(task),
                "scenario_train_split": f"{stage}-train", "scenario_validation_split": f"{stage}-validation",
                "validation_bank_count": len(bank), "validation_bank_hash": bank_hash,
                "selection_order": (["fleet_success", "collision_rate_ascending",
                                     "deadline_rate_ascending", "success_completion_steps_ascending",
                                     "path_efficiency_descending"] if stage == "S1" else
                                    ["fleet_success", "collision_rate_ascending",
                                     "success_completion_steps_ascending", "path_efficiency_descending"]),
                "ppo": {k: settings[k] for k in ("policy", "device", "n_envs", "n_steps", "batch_size",
                    "n_epochs", "learning_rate", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm")},
                "gamma_per_step": task.gamma, "seed": settings["seed"],
                "training_environment_seeds": [settings["seed"] + i for i in range(settings["n_envs"])],
                "rollout_transitions": rollout_transitions,
                "requested_timesteps": target}
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "dependency-versions.json", versions())
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(settings, sort_keys=True))
    model, env = make_model(settings, task)
    model.set_logger(configure(str(run_dir / "logs"), ["csv"]))
    best = None
    try:
        # Physically save the untrained model before evaluating it.
        save_policy(model, run_dir / "checkpoints" / "step-0.zip", task)
        baseline = evaluate_checkpoint(model, task, settings, run_dir, 0)
        best = task_metric_key(baseline, stage=stage)
        save_policy(model, run_dir / "best-by-task-metric.zip", task)
        write_json(run_dir / "best-by-task-metric.json", {"checkpoint_step": 0, "metric": best})
        write_json(run_dir / "status.json", status(model, settings, target, "running"))
        callback = EpisodeLogger(run_dir / "episodes.jsonl")
        while model.num_timesteps < target:
            chunk = min(interval, target - model.num_timesteps)
            model.learn(total_timesteps=chunk, callback=callback,
                        reset_num_timesteps=False, progress_bar=False)
            step = int(model.num_timesteps)
            save_policy(model, run_dir / "checkpoints" / f"step-{step}.zip", task)
            save_policy(model, run_dir / "latest.zip", task)
            report = evaluate_checkpoint(model, task, settings, run_dir, step)
            metric = task_metric_key(report, stage=stage)
            if metric > best:
                best = metric
                save_policy(model, run_dir / "best-by-task-metric.zip", task)
                write_json(run_dir / "best-by-task-metric.json", {"checkpoint_step": step, "metric": best})
            write_json(run_dir / "status.json", status(model, settings, target,
                                                       "complete" if step >= target else "running"))
        return summarize(run_dir)
    except BaseException as error:
        save_policy(model, run_dir / "latest.zip", task)
        record = status(model, settings, target, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        record["error"] = repr(error)
        write_json(run_dir / "status.json", record)
        raise
    finally:
        env.close()


def resume_s1(run_dir: Path, *, total_timesteps: int = 102400):
    """Extend a completed S1 PPO run without replacing its historical checkpoints."""
    settings, task = read_config(run_dir / "config.resolved.yaml")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    prior = json.loads((run_dir / "status.json").read_text())
    start = int(prior["actual_environment_steps"])
    interval = int(settings["evaluation_interval_steps"])
    if (settings.get("scenario_stage") != "S1" or task.dynamics != "kinematic"
            or prior["state"] != "complete" or start != 51200
            or total_timesteps != 102400 or prior["optimization_epochs"] != 250
            or total_timesteps % settings["n_steps"] or interval % settings["n_steps"]):
        raise ValueError("Resume requires completed 51,200-step kinematic S1 and target 102,400")
    bank = s1_validation_bank(settings["validation_count"], seed=settings["validation_seed"])
    bank_hash = hashlib.sha256(json.dumps([c.geometry_hash() for c in bank],
                                   separators=(",", ":")).encode()).hexdigest()
    if (bank_hash != manifest["validation_bank_hash"] or
            manifest["task"] != asdict(task) or manifest["schema"] != schema_record(task)
            or manifest["gamma_per_step"] != task.gamma or manifest["seed"] != settings["seed"]
            or manifest["requested_timesteps"] != start or
            manifest["ppo"] != {k: settings[k] for k in manifest["ppo"]}):
        raise ValueError("Frozen S1 run contract changed")
    if (run_dir / "resume-manifest.json").exists() or any(
            (run_dir / "checkpoints" / f"step-{step}.zip").exists()
            for step in range(start + interval, total_timesteps + 1, interval)):
        raise FileExistsError("Extension artifacts already exist")
    baseline_checkpoint = run_dir / "checkpoints" / f"step-{start}.zip"
    if not baseline_checkpoint.exists():
        raise FileNotFoundError(baseline_checkpoint)
    baseline_sha = hashlib.sha256(baseline_checkpoint.read_bytes()).hexdigest()
    env = DummyVecEnv([lambda: NavigationGymEnv(task, scenario_stage="S1")])
    try:
        model = load_policy(baseline_checkpoint, task, env=env)
        if model.num_timesteps != start or model._n_updates != 250 or not model.policy.optimizer.state:
            raise ValueError("PPO counters or optimizer state were not restored")
        # Historical runs did not save the active environment or RNG snapshots.
        # Advance the scenario generator past all sampled episodes, then resume
        # with the next scenario in the same seeded sequence. The interrupted
        # in-progress episode cannot be reconstructed from the old artifacts.
        completed_episodes = sum(1 for line in (run_dir / "episodes.jsonl").open() if line.strip())
        scenario_rng = np.random.default_rng(settings["seed"])
        for _ in range(completed_episodes + 1):
            sample_s1(scenario_rng)
        env.envs[0]._np_random = scenario_rng
        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        for name in ("latest.zip", "latest.schema.json", "best-by-task-metric.zip",
                     "best-by-task-metric.schema.json", "best-by-task-metric.json", "status.json"):
            shutil.copy2(run_dir / name, run_dir / f"before-resume-{name}")
        write_json(run_dir / "resume-manifest.json", {
            "run_id": manifest["run_id"], "seed": settings["seed"], "backend": task.dynamics,
            "start_steps": start, "target_steps": total_timesteps,
            "source_checkpoint": str(baseline_checkpoint), "source_checkpoint_sha256": baseline_sha,
            "optimizer_restored": True, "optimizer_state_entries": len(model.policy.optimizer.state),
            "start_optimizer_epochs": model._n_updates,
            "validation_bank_hash": bank_hash,
            "rng_note": "Prior torch RNG and active environment state were not saved. "
                        "Action RNG was reseeded with the original seed; the scenario RNG was "
                        "advanced past sampled episodes. This is a parameter/optimizer continuation, "
                        "not bitwise identical to an uninterrupted run."})
        model.set_logger(configure(str(run_dir / "logs" / f"resume-{start}"), ["csv"]))
        best_record = json.loads((run_dir / "best-by-task-metric.json").read_text())
        best = tuple(best_record["metric"])
        callback = EpisodeLogger(run_dir / "episodes.jsonl")
        write_json(run_dir / "status.json", status(model, settings, total_timesteps, "running"))
        while model.num_timesteps < total_timesteps:
            model.learn(total_timesteps=interval, callback=callback,
                        reset_num_timesteps=False, progress_bar=False)
            step = int(model.num_timesteps)
            if (run_dir / "evaluations" / f"step-{step}.json").exists():
                raise FileExistsError(f"Evaluation already exists at step {step}")
            save_policy(model, run_dir / "checkpoints" / f"step-{step}.zip", task)
            save_policy(model, run_dir / "latest.zip", task)
            report = evaluate_checkpoint(model, task, settings, run_dir, step)
            metric = task_metric_key(report, stage="S1")
            if metric > best:
                best = metric
                save_policy(model, run_dir / "best-by-task-metric.zip", task)
                write_json(run_dir / "best-by-task-metric.json",
                           {"checkpoint_step": step, "metric": best})
            write_json(run_dir / "status.json", status(model, settings, total_timesteps,
                       "complete" if step == total_timesteps else "running"))
        if (model.num_timesteps != 102400 or model._n_updates != 500
                or hashlib.sha256(baseline_checkpoint.read_bytes()).hexdigest() != baseline_sha):
            raise AssertionError("Resume counters or preserved baseline checkpoint failed")
        return summarize(run_dir)
    except BaseException as error:
        record = status(model, settings, total_timesteps,
                        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        record["error"] = repr(error)
        write_json(run_dir / "status.json", record)
        raise
    finally:
        env.close()


def check(config_path, backend, seed):
    _, task = read_config(config_path, backend=backend, seed=seed)
    env = NavigationGymEnv(task)
    try:
        gym_check_env(env, skip_render_check=True)
        sb3_check_env(env)
    finally:
        env.close()
    # Case A remains a control/integration baseline, never S0 learnability evidence.
    fixture = case_a_reference(seed)
    env = NavigationGymEnv(task, scenario=fixture)
    try:
        env.reset(seed=seed)
        for index in range(task.deadline_steps):
            _, _, ended, cutoff, info = env.step([0., 0.])
            if ended or cutoff:
                break
        baseline = {"terminal_reason": info["terminal_reason"], "episode_length": index + 1,
                    "final_goal_distance_m": info["final_goal_distance_m"]["vessel_0"]}
    finally:
        env.close()
    return {"backend": task.dynamics, "gymnasium_checker": "passed", "sb3_checker": "passed",
            "scripted_case_a_control": baseline}


def summarize(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = []
    for file in sorted((run_dir / "evaluations").glob("step-*.json"),
                       key=lambda path: int(path.stem.split("-")[-1])):
        report = json.loads(file.read_text())
        records.append({k: report[k] for k in ("checkpoint_step", "success_rate", "collision_rate",
            "deadline_rate", "mean_episode_length", "mean_minimum_goal_distance_m",
            "mean_final_goal_distance_m", "mean_path_efficiency", "mean_reward_diagnostic_only",
            "mean_commanded_speed_mps", "mean_achieved_surge_mps",
            "mean_commanded_yaw_rate_radps", "mean_achieved_yaw_rate_radps") if k in report})
    return {"run_id": manifest["run_id"], "backend": manifest["backend"],
            "status": json.loads((run_dir / "status.json").read_text()),
            "best_by_task_metric": json.loads((run_dir / "best-by-task-metric.json").read_text()),
            "evaluations": records, "artifacts": {"run_dir": str(run_dir),
                "initial_policy": str(run_dir / "checkpoints" / "step-0.zip"),
                "final_policy": str(run_dir / "latest.zip"),
                "best_policy": str(run_dir / "best-by-task-metric.zip")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Navigation V3 with stock SB3 PPO")
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check")
    p_check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_check.add_argument("--backend", choices=("kinematic", "bcod-reduced", "bcod-full"), default="kinematic")
    p_check.add_argument("--seed", type=int, default=11)
    p_train = sub.add_parser("train")
    p_train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_train.add_argument("--backend", choices=("kinematic", "bcod-reduced", "bcod-full"), default="kinematic")
    p_train.add_argument("--seed", type=int)
    p_train.add_argument("--seeds", type=int, nargs="+")
    p_train.add_argument("--total-timesteps", type=int)
    p_train.add_argument("--action-mode", choices=("low-level", "high-level"))
    p_train.add_argument("--run-dir", type=Path)
    p_train.add_argument("--run-root", type=Path)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--run-dir", type=Path, required=True)
    p_eval.add_argument("--checkpoint", type=Path)
    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--run-dir", type=Path, required=True)
    p_resume = sub.add_parser("resume-s1")
    p_resume.add_argument("--run-dir", type=Path, required=True)
    p_resume.add_argument("--total-timesteps", type=int, default=102400)
    p_scripted = sub.add_parser("scripted-s0")
    p_scripted.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_scripted.add_argument("--backend", choices=("kinematic", "bcod-reduced", "bcod-full"),
                            default="kinematic")
    p_qualify = sub.add_parser("qualify-s1")
    p_qualify.add_argument("--config", type=Path, required=True)
    p_qualify.add_argument("--backend", choices=("kinematic", "bcod-reduced", "bcod-full"), required=True)
    p_qualify.add_argument("--seed", type=int, required=True)
    p_qualify.add_argument("--run-dir", type=Path, required=True)
    p_qualify.add_argument("--anneal-lr", action="store_true")
    p_qualify.add_argument("--initialization", type=Path,
                           help="Compatible policy checkpoint; copies weights into a fresh PPO optimizer")
    p_qualify.add_argument("--final-test-bank", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check(args.config, args.backend, args.seed)
    elif args.command == "train":
        if args.seeds is not None:
            if args.seed is not None or args.run_dir is not None or args.run_root is None:
                parser.error("--seeds requires --run-root and excludes --seed/--run-dir")
            result = [train_one(args.config, args.run_root / f"seed-{seed}", seed=seed,
                                backend=args.backend, total_timesteps=args.total_timesteps,
                                action_mode=args.action_mode)
                      for seed in args.seeds]
        else:
            if args.run_dir is None or args.run_root is not None:
                parser.error("single-seed train requires --run-dir")
            result = train_one(args.config, args.run_dir, seed=args.seed,
                               backend=args.backend, total_timesteps=args.total_timesteps,
                               action_mode=args.action_mode)
    elif args.command == "evaluate":
        settings, task = read_config(args.run_dir / "config.resolved.yaml")
        checkpoint = args.checkpoint or args.run_dir / "latest.zip"
        model = load_policy(checkpoint, task)
        result = evaluate_checkpoint(model, task, settings, args.run_dir, int(model.num_timesteps))
    elif args.command == "resume-s1":
        result = resume_s1(args.run_dir, total_timesteps=args.total_timesteps)
    elif args.command == "scripted-s0":
        settings, task = read_config(args.config, backend=args.backend)
        result = evaluate_bank(GeometricController(), task,
                               s0_validation_bank(settings["validation_count"],
                                                  seed=settings["validation_seed"]), checkpoint_step=0)
    elif args.command == "qualify-s1":
        from .qualification import qualify_one
        result = qualify_one(args.config, args.run_dir, seed=args.seed,
                             backend=args.backend, anneal_lr=args.anneal_lr,
                             initialization=args.initialization,
                             use_final_test_bank=args.final_test_bank)
    else:
        result = summarize(args.run_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
