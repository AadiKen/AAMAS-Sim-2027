"""Train/dev/test qualification for the frozen V3 single-agent PPO task."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from stable_baselines3.common.logger import configure

from .evaluate import evaluate_bank, task_metric_key
from .runner import (EpisodeLogger, load_policy, make_model, read_config, revision,
                     save_policy, schema_record, status, versions, write_json)
from .scenarios import S1_GENERATOR_VERSION, s1_dev_bank, s1_test_bank, s1_final_test_bank


SELECTION_ORDER = ("success_count_desc", "collision_count_asc", "deadline_count_asc",
                   "mean_success_completion_steps_asc", "mean_path_efficiency_desc")
EVALUATION_INTERVAL = 10240
MAX_TRANSITIONS = 51200


def bank_hash(cases) -> str:
    return hashlib.sha256(json.dumps([case.geometry_hash() for case in cases],
                                     separators=(",", ":")).encode()).hexdigest()


def selection_reason(old: dict, new: dict) -> str | None:
    """Strict lexicographic improvement; exact ties retain the earlier checkpoint."""
    old_key = task_metric_key(old, stage="S1")
    new_key = task_metric_key(new, stage="S1")
    if new_key <= old_key:
        return None
    for name, before, after in zip(SELECTION_ORDER, old_key, new_key):
        if after != before:
            return f"{name}: {before} -> {after}"
    raise AssertionError("Improvement lacked a deciding criterion")


def initialize_policy_from_checkpoint(model, checkpoint: Path, task) -> None:
    """Copy compatible policy weights while retaining a fresh target optimizer."""
    source = load_policy(checkpoint, task)
    model.policy.load_state_dict(source.policy.state_dict())
    del source


def qualify_one(config_path: Path, run_dir: Path, *, seed: int, backend: str,
                anneal_lr: bool = False, stage: str = "S1",
                initialization: Path | None = None,
                use_final_test_bank: bool = False) -> dict:
    settings, task = read_config(config_path, backend=backend, seed=seed)
    if stage != "S1" or settings.get("scenario_stage") != "S1":
        raise ValueError("This qualification entry point requires S1")
    if (settings["n_envs"], settings["n_steps"], settings["batch_size"],
            settings["n_epochs"], settings["learning_rate"]) != (8, 256, 64, 10, .0003):
        raise ValueError("S1 qualification requires frozen Config D")
    if settings["evaluation_interval_steps"] != EVALUATION_INTERVAL:
        raise ValueError("Checkpoint cadence differs from frozen S1")
    dev = s1_dev_bank()
    dev_digest = bank_hash(dev)
    test = s1_final_test_bank() if use_final_test_bank else s1_test_bank()
    test_digest = bank_hash(test)
    if dev_digest != "3666670b1d6bddfe6144e7cd24b7f7777cc61dff06b88aac4f820e37d683f561":
        raise ValueError("S1-dev bank changed")
    run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("logs", "checkpoints", "dev-evaluations", "dev-trajectories", "test"):
        (run_dir / folder).mkdir()
    manifest = {
        "protocol": "S1-train-dev-test-v1", "backend": backend, "seed": seed,
        "code_revision": revision(), "dependencies": versions(),
        "task": asdict(task), "schema": schema_record(task),
        "scenario_generator_version": S1_GENERATOR_VERSION,
        "train_split": "S1-train", "dev_split": "S1-dev",
        "test_split": "S1-final-test" if use_final_test_bank else "S1-test",
        "dev_bank_sha256": dev_digest, "test_bank_sha256": test_digest,
        "dev_bank_count": len(dev), "test_bank_count": len(test),
        "initialization": ({"method": "S0_policy_weights_fresh_optimizer",
                            "checkpoint": str(initialization),
                            "sha256": hashlib.sha256(initialization.read_bytes()).hexdigest()}
                           if initialization is not None else {"method": "random"}),
        "selection_order": SELECTION_ORDER, "exact_tie_preference": "earlier_checkpoint",
        "early_stopping": {"consecutive_nonimprovements": 3, "minimum_transitions": 20480},
        "maximum_environment_transitions": MAX_TRANSITIONS,
        "evaluation_interval_transitions": EVALUATION_INTERVAL,
        "ppo": {k: settings[k] for k in ("policy", "device", "n_envs", "n_steps",
                 "batch_size", "n_epochs", "learning_rate", "gae_lambda", "clip_range",
                 "ent_coef", "vf_coef", "max_grad_norm")},
        "gamma_per_step": task.gamma,
        "learning_rate_schedule": ("linear_3e-4_to_0_over_51200" if anneal_lr else "constant_3e-4"),
        "test_policy": "evaluate selected best-dev exactly once after training ends",
    }
    write_json(run_dir / "manifest.json", manifest)
    model, env = make_model(settings, task)
    if initialization is not None:
        initialize_policy_from_checkpoint(model, initialization, task)
    if anneal_lr:
        model.lr_schedule = lambda _progress: settings["learning_rate"] * max(
            0., 1. - model.num_timesteps / MAX_TRANSITIONS)
    model.set_logger(configure(str(run_dir / "logs"), ["csv"]))
    best_report = None
    best_step = 0
    no_improve = 0
    try:
        def evaluate_dev(step):
            report = evaluate_bank(model, task, dev, checkpoint_step=step,
                                   trajectory_dir=run_dir / "dev-trajectories")
            write_json(run_dir / "dev-evaluations" / f"step-{step}.json", report)
            return report

        save_policy(model, run_dir / "checkpoints" / "step-0.zip", task)
        best_report = evaluate_dev(0)
        save_policy(model, run_dir / "best-dev.zip", task)
        write_json(run_dir / "best-dev.json", {"checkpoint_step": 0,
                   "metric": task_metric_key(best_report, stage="S1"), "reason": "initial checkpoint"})
        write_json(run_dir / "status.json", status(model, settings, MAX_TRANSITIONS, "running"))
        callback = EpisodeLogger(run_dir / "episodes.jsonl")
        while model.num_timesteps < MAX_TRANSITIONS:
            model.learn(total_timesteps=EVALUATION_INTERVAL, callback=callback,
                        reset_num_timesteps=False, progress_bar=False)
            step = int(model.num_timesteps)
            save_policy(model, run_dir / "checkpoints" / f"step-{step}.zip", task)
            save_policy(model, run_dir / "latest.zip", task)
            report = evaluate_dev(step)
            reason = selection_reason(best_report, report)
            if reason is not None:
                best_report, best_step, no_improve = report, step, 0
                save_policy(model, run_dir / "best-dev.zip", task)
                write_json(run_dir / "best-dev.json", {"checkpoint_step": step,
                           "metric": task_metric_key(report, stage="S1"), "reason": reason})
            else:
                no_improve += 1
            with (run_dir / "selection-history.jsonl").open("a") as file:
                file.write(json.dumps({"step": step, "selected": reason is not None,
                    "reason": reason or "no lexicographic improvement",
                    "consecutive_nonimprovements": no_improve,
                    "best_dev_step": best_step}) + "\n")
            stopped = step < MAX_TRANSITIONS and step >= 20480 and no_improve >= 3
            state = "early_stopped" if stopped else "complete" if step == MAX_TRANSITIONS else "running"
            write_json(run_dir / "status.json", status(model, settings, MAX_TRANSITIONS, state))
            if stopped:
                break
        final_step = int(model.num_timesteps)
        save_policy(model, run_dir / "final.zip", task)
        write_json(run_dir / "final.json", {"checkpoint_step": final_step,
                   "best_dev_checkpoint_step": best_step,
                   "early_stopped": final_step < MAX_TRANSITIONS})
    except BaseException as error:
        record = status(model, settings, MAX_TRANSITIONS,
                        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        record["error"] = repr(error)
        write_json(run_dir / "status.json", record)
        raise
    finally:
        env.close()
    # S1-test is first evaluated only after optimization and checkpoint selection end.
    test_file = run_dir / "test" / "best-dev-once.json"
    if test_file.exists():
        raise FileExistsError(test_file)
    selected = load_policy(run_dir / "best-dev.zip", task)
    test_report = evaluate_bank(selected, task, test, checkpoint_step=best_step,
                                trajectory_dir=run_dir / "test" / "trajectories")
    if test_report["scenario_bank_hash"] != test_digest:
        raise AssertionError("S1-test bank changed")
    write_json(test_file, test_report)
    completed_hashes = {json.loads(line)["scenario_hash"] for line in
                        (run_dir / "episodes.jsonl").read_text().splitlines()}
    reserved_hashes = {case.geometry_hash() for case in (*dev, *s1_test_bank(),
                                                         *s1_final_test_bank())}
    if completed_hashes & reserved_hashes:
        raise AssertionError("Training scenario overlapped dev/test")
    return {"run_dir": str(run_dir), "seed": seed, "backend": backend,
            "best_dev_step": best_step, "final_step": final_step,
            "dev_success_count": best_report["success_count"],
            "test_success_count": test_report["success_count"],
            "test_collision_count": test_report["collision_count"],
            "test_deadline_count": test_report["deadline_count"]}
