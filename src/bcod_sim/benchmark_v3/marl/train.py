"""Bounded M0 BenchMARL qualification; all optimizer math remains in BenchMARL."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import shutil

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models import MlpConfig
import torch

from ..centralized_state import CENTRALIZED_STATE_VERSION, STATE_SCHEMA_HASH
from ..observations import SCHEMA_HASH as ACTOR_SCHEMA_HASH
from .evaluate import BenchMARLActorPolicy, evaluate_m0
from .scenarios import M0_SCENARIO_VERSION, bank_hash, m0_dev_bank, m0_test_bank
from .task import V3BenchMARLTask, V3M0Task


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def score(report):
    n = report["scenario_count"]
    return (report["fleet_success_count"], -report["collision_count"],
            -report["deadline_count"],
            -(report["mean_success_completion_steps"] or 1e9),
            report["mean_path_efficiency"])


class StopOnDevPlateau(Exception):
    pass


class DevSelection(Callback):
    def __init__(self, out, *, evaluation_interval, min_frames, max_frames, patience=3):
        super().__init__()
        self.out = Path(out)
        self.evaluation_interval = evaluation_interval
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.patience = patience
        self.best = None
        self.best_step = None
        self.nonimprovements = 0
        self.last_evaluated_step = 0

    def evaluate(self, step):
        actor = BenchMARLActorPolicy(self.experiment.policy)
        report = evaluate_m0(actor, m0_dev_bank(),
                             trace_dir=self.out / "trajectories" / f"dev-step-{step}")
        write_json(self.out / "dev" / f"step-{step}.json", report)
        metric = score(report)
        reason = None
        if self.best is None or metric > self.best:
            reason = "initial" if self.best is None else next(
                field for field, old, new in zip(
                    ("fleet_success", "collision", "deadline", "completion", "path_efficiency"),
                    self.best, metric) if old != new)
            self.best, self.best_step = metric, step
            self.nonimprovements = 0
            torch.save(self.experiment.policy.state_dict(), self.out / "best-dev-actor.pt")
            torch.save(self.experiment.state_dict(), self.out / "best-dev.pt")
        else:
            self.nonimprovements += 1
        with (self.out / "selection.jsonl").open("a") as file:
            file.write(json.dumps({"step": step, "score": metric, "selected": reason is not None,
                                   "reason": reason, "best_step": self.best_step,
                                   "consecutive_nonimprovements": self.nonimprovements}) + "\n")
        self.last_evaluated_step = step
        return report

    def on_train_end(self, training_td, group):
        if group != "agents":
            return
        for value in training_td.values(True, True):
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite BenchMARL training diagnostic")
        step = int(self.experiment.total_frames)
        if step % self.evaluation_interval:
            return
        torch.save(self.experiment.state_dict(), self.out / "checkpoints" / f"step-{step}.pt")
        torch.save(self.experiment.policy.state_dict(), self.out / "latest-actor.pt")
        self.evaluate(step)
        if step < self.max_frames and step >= self.min_frames and self.nonimprovements >= self.patience:
            raise StopOnDevPlateau(step)


def make_experiment(out, seed, *, frames, rollout_frames, envs, minibatch_size,
                    minibatch_iters, callback, stage_path=None,
                    normalize_advantage=False, task_config=None, scenario_sampler=None,
                    centralized_state=True):
    if task_config is None and scenario_sampler is None:
        task = V3M0Task(stage_path=stage_path)
    else:
        if task_config is None or scenario_sampler is None:
            raise ValueError("task_config and scenario_sampler must be supplied together")
        task = V3BenchMARLTask(task_config, scenario_sampler=scenario_sampler,
                               centralized_state=centralized_state)
    algorithm = MappoConfig.get_from_yaml()
    model = MlpConfig.get_from_yaml()
    config = replace(ExperimentConfig.get_from_yaml(),
                     gamma=.99 ** .2,  # Match the versioned V3 potential shaping.
                     max_n_frames=frames,
                     on_policy_collected_frames_per_batch=rollout_frames,
                     on_policy_n_envs_per_worker=envs,
                     on_policy_n_minibatch_iters=minibatch_iters,
                     on_policy_minibatch_size=minibatch_size,
                     render=False, evaluation=False, loggers=["csv"],
                     checkpoint_interval=rollout_frames, checkpoint_at_end=True,
                     keep_checkpoints_num=None, save_folder=str(out / "benchmarl"))
    (out / "benchmarl").mkdir(parents=True, exist_ok=True)
    experiment = Experiment(task=task, algorithm_config=algorithm,
                            model_config=model, critic_model_config=MlpConfig.get_from_yaml(),
                            seed=seed, config=config, callbacks=[callback])
    if normalize_advantage:
        # TorchRL's published ClipPPOLoss switch; BenchMARL retains ownership
        # of advantage computation and every optimization step.
        loss, _ = experiment.algorithm.get_loss_and_updater("agents")
        loss.normalize_advantage = True
    return experiment, algorithm, model, config


def train_one(out: Path, seed: int, *, frames=24000, rollout_frames=6000,
              envs=10, minibatch_size=400, minibatch_iters=45,
              task_config=None, scenario_sampler=None, centralized_state=True):
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    callback = DevSelection(out, evaluation_interval=rollout_frames,
                            min_frames=max(2*rollout_frames, min(frames, 12000)),
                            max_frames=frames)
    experiment, algorithm, model, config = make_experiment(
        out, seed, frames=frames, rollout_frames=rollout_frames, envs=envs,
        minibatch_size=minibatch_size, minibatch_iters=minibatch_iters,
        callback=callback, task_config=task_config, scenario_sampler=scenario_sampler,
        centralized_state=centralized_state)
    import benchmarl, torchrl, tensordict, pettingzoo
    manifest = {"trainer": "benchmarl", "algorithm": "MAPPO",
                "stage": "M0", "backend": "kinematic", "seed": seed,
                "algorithm": "BenchMARL MAPPO", "action_schema": "desired-speed-heading-v1",
                "actor_observation_schema_hash": ACTOR_SCHEMA_HASH,
                "centralized_state_schema_version": CENTRALIZED_STATE_VERSION,
                "centralized_state_schema_hash": STATE_SCHEMA_HASH,
                "scenario_version": M0_SCENARIO_VERSION,
                "dev_bank_hash": bank_hash(m0_dev_bank()),
                "test_bank_hash": bank_hash(m0_test_bank()),
                "python": platform.python_version(),
                "versions": {m.__name__: m.__version__ for m in
                    (torch, torchrl, tensordict, benchmarl, pettingzoo)},
                "mappo": asdict(algorithm), "model": asdict(model),
                "experiment_config": asdict(config),
                "test_usage": "deferred until all seeds finish and configuration is frozen"}
    write_json(out / "manifest.json", manifest)
    torch.save(experiment.state_dict(), out / "checkpoints" / "step-0.pt")
    torch.save(experiment.policy.state_dict(), out / "step-zero-actor.pt")
    callback.evaluate(0)
    status = "running"
    try:
        experiment.run()
        status = "complete"
    except StopOnDevPlateau:
        status = "early_stopped"
    except BaseException as error:
        status = "failed"
        write_json(out / "status.json", {"status": status, "error": repr(error),
                                          "frames": experiment.total_frames})
        raise
    finally:
        torch.save(experiment.policy.state_dict(), out / "final-actor.pt")
    final_step = int(experiment.total_frames)
    if callback.last_evaluated_step != final_step:
        callback.evaluate(final_step)
    write_json(out / "status.json", {"status": status, "frames": final_step,
                                      "best_dev_step": callback.best_step,
                                      "iterations": experiment.n_iters_performed})
    return {"run_dir": str(out), "seed": seed, "frames": final_step,
            "best_dev_step": callback.best_step, "status": status}


def evaluate_selected(out: Path):
    target = out / "test" / "best-dev-once.json"
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(exist_ok=True)
    # Actor parameters are loaded into the same installed BenchMARL public policy.
    callback = DevSelection(out, evaluation_interval=6000, min_frames=12000,
                            max_frames=10**12)
    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest["experiment_config"]
    experiment, _, _, _ = make_experiment(
        out, manifest["seed"], frames=config["max_n_frames"],
        rollout_frames=config["on_policy_collected_frames_per_batch"],
        envs=config["on_policy_n_envs_per_worker"],
        minibatch_size=config["on_policy_minibatch_size"],
        minibatch_iters=config["on_policy_n_minibatch_iters"], callback=callback)
    try:
        experiment.policy.load_state_dict(torch.load(out / "best-dev-actor.pt",
                                                      map_location="cpu", weights_only=True))
        report = evaluate_m0(BenchMARLActorPolicy(experiment.policy), m0_test_bank(),
                             trace_dir=out / "trajectories" / "test-best-dev")
        write_json(target, report)
        return {"run_dir": str(out), "fleet_success": report["fleet_success_count"],
                "collisions": report["collision_count"], "deadlines": report["deadline_count"]}
    finally:
        experiment.close()


def summarize(out: Path):
    """Read saved run artifacts without constructing an environment or changing them."""
    manifest = json.loads((out / "manifest.json").read_text())
    status = json.loads((out / "status.json").read_text())
    result = {"run_dir": str(out), "seed": manifest["seed"], **status,
              "dev_bank_hash": manifest["dev_bank_hash"],
              "test_bank_hash": manifest["test_bank_hash"]}
    target = out / "test" / "best-dev-once.json"
    if target.exists():
        report = json.loads(target.read_text())
        result["test"] = {name: report[name] for name in
                          ("fleet_success_count", "collision_count", "deadline_count",
                           "per_agent_success_count", "mean_episode_length",
                           "mean_path_efficiency")}
    return result


def check():
    """Check the frozen M0 API contract without collecting training data."""
    from torchrl.envs.utils import check_env_specs
    task = V3M0Task(deadline_steps=40)
    env = task.get_env_fun(1, True, 11, "cpu")()
    try:
        check_env_specs(env)
        actor = task.observation_spec(env)["agents", "observation"]
        critic = task.state_spec(env)["state"]
        if tuple(actor.shape) != (2, 78) or tuple(critic.shape) != (89,):
            raise AssertionError("M0 actor/critic schema mismatch")
        return {"status": "passed", "actor_shape": list(actor.shape),
                "critic_shape": list(critic.shape),
                "dev_bank_hash": bank_hash(m0_dev_bank()),
                "test_bank_hash": bank_hash(m0_test_bank())}
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="V3 M0 established BenchMARL MAPPO")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--frames", type=int, default=24000)
    train.add_argument("--rollout-frames", type=int, default=6000)
    train.add_argument("--envs", type=int, default=10)
    train.add_argument("--minibatch-size", type=int, default=400)
    train.add_argument("--minibatch-iters", type=int, default=45)
    test = sub.add_parser("evaluate-selected")
    test.add_argument("--run-dir", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--run-dir", type=Path, required=True)
    sub.add_parser("check")
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_one(args.run_dir, args.seed, frames=args.frames,
                           rollout_frames=args.rollout_frames, envs=args.envs,
                           minibatch_size=args.minibatch_size,
                           minibatch_iters=args.minibatch_iters)
    elif args.command == "evaluate-selected":
        result = evaluate_selected(args.run_dir)
    elif args.command == "summarize":
        result = summarize(args.run_dir)
    else:
        result = check()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
