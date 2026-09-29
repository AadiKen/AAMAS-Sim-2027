"""Frozen M0 curriculum around established BenchMARL/TorchRL MAPPO."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil

from benchmarl.experiment.callback import Callback
import torch
import yaml

from .. import OBSERVATION_VERSION, TASK_VERSION
from ..centralized_state import CENTRALIZED_STATE_VERSION, STATE_SCHEMA_HASH
from ..observations import SCHEMA_HASH as ACTOR_SCHEMA_HASH
from .curriculum import Curriculum, CurriculumController, CurriculumStage, write_stage
from .evaluate import BenchMARLActorPolicy, evaluate_m0
from .scenarios import (M0_CURRICULUM_VERSION, bank_hash, m0_curriculum_bank,
                        reserved_m0_curriculum_hashes)
from .train import make_experiment, score, write_json


class CurriculumStageFailed(Exception):
    pass


class CurriculumPlateau(Exception):
    pass


class CurriculumFinished(Exception):
    pass


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe(path):
    recipe = yaml.safe_load(Path(path).read_text())
    if (recipe["recipe_version"] != "m0-highlevel-mappo-curriculum-v1" or
            recipe["backend"] != "kinematic" or
            recipe["action_schema"] != "desired-speed-heading-v1" or
            recipe["agent_count"] != 2):
        raise ValueError("Unsupported or unfrozen M0 curriculum recipe")
    curriculum = Curriculum(tuple(CurriculumStage(**stage) for stage in recipe["stages"]))
    return recipe, curriculum


def bank_manifest():
    return {f"{stage}-{split}": bank_hash(m0_curriculum_bank(stage, split))
            for stage in "ABC" for split in ("dev", "test")}


class CurriculumSelection(Callback):
    def __init__(self, out, curriculum, evaluation_interval, *, smoke_dev_cases=None):
        super().__init__()
        self.out = Path(out)
        self.controller = CurriculumController(curriculum)
        self.stage_path = self.out / "current-stage.json"
        self.evaluation_interval = evaluation_interval
        self.best = None
        self.best_step = None
        self.nonimprovements = 0
        self.last_evaluated_step = -1
        self.failed_stage = None
        self.stage_metrics = []
        self.smoke_dev_cases = smoke_dev_cases

    def evaluate(self, step):
        stage = self.controller.stage.name
        bank = m0_curriculum_bank(stage, "dev")
        if self.smoke_dev_cases is not None:
            bank = bank[:self.smoke_dev_cases]
        report = evaluate_m0(BenchMARLActorPolicy(self.experiment.policy), bank,
                             trace_dir=self.out / "trajectories" / f"{stage}-dev-step-{step}")
        write_json(self.out / "dev" / f"{stage}-step-{step}.json", report)
        metric = score(report)
        changed = self.best is None or metric > self.best
        if changed:
            self.best, self.best_step, self.nonimprovements = metric, step, 0
            torch.save(self.experiment.policy.state_dict(),
                       self.out / f"best-{stage}-actor.pt")
            torch.save(self.experiment.state_dict(), self.out / f"best-{stage}.pt")
            if stage == "C":
                torch.save(self.experiment.policy.state_dict(), self.out / "best-dev-actor.pt")
                torch.save(self.experiment.state_dict(), self.out / "best-dev.pt")
        else:
            self.nonimprovements += 1
        record = {"stage": stage, "step": step, "score": metric,
                  "selected": changed, "best_step": self.best_step,
                  "success": report["fleet_success_count"],
                  "collision": report["collision_count"],
                  "deadline": report["deadline_count"],
                  "mean_episode_length": report["mean_episode_length"],
                  "mean_saturation": report["mean_heading_controller_saturation_rate"]}
        self.stage_metrics.append(record)
        with (self.out / "selection.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.last_evaluated_step = step
        if step == 0:
            return "continue"
        result = self.controller.record(
            step, success_rate=report["fleet_success_count"] / report["scenario_count"],
            collision_rate=report["collision_count"] / report["scenario_count"])
        write_json(self.out / "stage-history.json", self.controller.history)
        if result == "promote":
            torch.save(self.experiment.state_dict(),
                       self.out / "checkpoints" / f"transition-{stage}-step-{step}.pt")
            write_stage(self.stage_path, self.controller.stage.name)
            self.best = None
            self.best_step = None
            self.nonimprovements = 0
        elif result == "fail":
            self.failed_stage = stage
        return result

    def on_train_end(self, training_td, group):
        if group != "agents":
            return
        for value in training_td.values(True, True):
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite BenchMARL training diagnostic")
        step = int(self.experiment.total_frames)
        if step % self.evaluation_interval:
            return
        torch.save(self.experiment.state_dict(),
                   self.out / "checkpoints" / f"step-{step}.pt")
        torch.save(self.experiment.policy.state_dict(), self.out / "latest-actor.pt")
        result = self.evaluate(step)
        if result == "fail":
            raise CurriculumStageFailed(self.failed_stage)
        if result == "complete":
            raise CurriculumFinished(step)
        if (self.controller.stage.name == "C" and
                step - self.controller.entry_step >= 12000 and
                self.nonimprovements >= 3):
            raise CurriculumPlateau(step)


def train_curriculum(out: Path, seed: int, recipe_path: Path, *, smoke_dev_cases=None):
    recipe, curriculum = load_recipe(recipe_path)
    if seed not in recipe["seeds"]:
        raise ValueError("Seed is not declared in the frozen recipe")
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    stage_path = out / "current-stage.json"
    write_stage(stage_path, curriculum.stages[0].name)
    reserved_m0_curriculum_hashes()  # Assert all fixed banks are disjoint.
    max_frames = sum(stage.maximum_training_steps for stage in curriculum.stages)
    callback = CurriculumSelection(out, curriculum, recipe["evaluation_interval"],
                                   smoke_dev_cases=smoke_dev_cases)
    experiment, algorithm, model, config = make_experiment(
        out, seed, frames=max_frames, rollout_frames=recipe["rollout_frames"],
        envs=recipe["n_envs"], minibatch_size=recipe["minibatch_size"],
        minibatch_iters=recipe["minibatch_iterations"], callback=callback,
        stage_path=stage_path, normalize_advantage=True)
    import benchmarl, torchrl, pettingzoo, tensordict
    manifest = {"recipe": recipe, "recipe_hash": file_hash(recipe_path),
                "seed": seed, "scenario_version": M0_CURRICULUM_VERSION,
                "smoke_dev_cases": smoke_dev_cases,
                "task_version": TASK_VERSION, "observation_version": OBSERVATION_VERSION,
                "actor_schema_hash": ACTOR_SCHEMA_HASH,
                "centralized_state_version": CENTRALIZED_STATE_VERSION,
                "centralized_state_hash": STATE_SCHEMA_HASH,
                "bank_hashes": bank_manifest(),
                "versions": {m.__name__: m.__version__ for m in
                             (torch, torchrl, tensordict, benchmarl, pettingzoo)},
                "python": platform.python_version(),
                "algorithm_config": asdict(algorithm), "model_config": asdict(model),
                "experiment_config": asdict(config),
                "advantage_normalization": "TorchRL ClipPPOLoss.normalize_advantage=True",
                "value_normalization": "not exposed by BenchMARL 1.5.2 default MAPPO path"}
    write_json(out / "manifest.json", manifest)
    torch.save(experiment.state_dict(), out / "checkpoints" / "step-0.pt")
    torch.save(experiment.policy.state_dict(), out / "step-zero-actor.pt")
    callback.evaluate(0)
    status = "running"
    try:
        experiment.run()
        status = "complete"
    except CurriculumStageFailed:
        status = "promotion_failed"
    except CurriculumPlateau:
        status = "stage_C_plateau"
    except CurriculumFinished:
        status = "complete"
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error),
                                         "frames": int(experiment.total_frames)})
        raise
    finally:
        torch.save(experiment.policy.state_dict(), out / "final-actor.pt")
        final_checkpoint = out / "checkpoints" / f"step-{int(experiment.total_frames)}.pt"
        if final_checkpoint.exists():
            shutil.copy2(final_checkpoint, out / "final.pt")
        experiment.close()
    if not (out / "best-dev-actor.pt").exists():
        # Early A/B failures still retain their selected development policy,
        # explicitly marked by stage; it is ineligible for the C-test gate.
        stage = callback.controller.stage.name
        shutil.copy2(out / f"best-{stage}-actor.pt", out / "best-dev-actor.pt")
        shutil.copy2(out / f"best-{stage}.pt", out / "best-dev.pt")
    write_json(out / "selection-summary.json", {
        "seed": seed, "selection_stage": callback.controller.stage.name,
        "best_dev_step": callback.best_step,
        "selected_actor_sha256": file_hash(out / "best-dev-actor.pt"),
        "test_evaluation": ("eligible after all seeds finish"
                            if callback.controller.stage.name == "C" else
                            "not run: final M0-C stage not reached")})
    write_json(out / "status.json", {"status": status,
                                     "frames": int(experiment.total_frames),
                                     "stage": callback.controller.stage.name,
                                     "failed_stage": callback.failed_stage,
                                     "best_dev_step": callback.best_step,
                                     "rollouts": experiment.n_iters_performed})
    return {"run_dir": str(out), "seed": seed, "status": status,
            "stage": callback.controller.stage.name,
            "frames": int(experiment.total_frames)}


def evaluate_selected(out: Path):
    target = out / "test" / "best-dev-once.json"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((out / "manifest.json").read_text())
    status = json.loads((out / "status.json").read_text())
    if status["stage"] != "C" or not (out / "best-dev-actor.pt").exists():
        raise ValueError("Seed did not reach the final M0-C stage")
    recipe = manifest["recipe"]
    callback = CurriculumSelection(out, Curriculum(tuple(
        CurriculumStage(**s) for s in recipe["stages"])), recipe["evaluation_interval"])
    experiment, _, _, _ = make_experiment(
        out, manifest["seed"], frames=manifest["experiment_config"]["max_n_frames"],
        rollout_frames=recipe["rollout_frames"], envs=recipe["n_envs"],
        minibatch_size=recipe["minibatch_size"],
        minibatch_iters=recipe["minibatch_iterations"], callback=callback,
        stage_path=out / "current-stage.json", normalize_advantage=True)
    try:
        experiment.policy.load_state_dict(torch.load(out / "best-dev-actor.pt",
                                                     map_location="cpu", weights_only=True))
        report = evaluate_m0(BenchMARLActorPolicy(experiment.policy),
                             m0_curriculum_bank("C", "test"),
                             trace_dir=out / "trajectories" / "test-best-dev")
        report["selected_actor_sha256"] = file_hash(out / "best-dev-actor.pt")
        write_json(target, report)
        return {"run_dir": str(out), "success": report["fleet_success_count"],
                "collision": report["collision_count"],
                "deadline": report["deadline_count"]}
    finally:
        experiment.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen M0 high-level MAPPO curriculum")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=Path(
        "configs/benchmark_v3/marl/m0_highlevel_mappo_curriculum_v1.yaml"))
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--smoke-dev-cases", type=int, default=None,
                       help="Integration smoke only: evaluate the first N development cases")
    evaluate = sub.add_parser("evaluate-selected")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (train_curriculum(args.run_dir, args.seed, args.config,
                               smoke_dev_cases=args.smoke_dev_cases)
              if args.command == "train" else evaluate_selected(args.run_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
