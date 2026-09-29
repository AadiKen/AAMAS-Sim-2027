"""Recipe-driven MARL orchestration around the existing BenchMARL implementations."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import TaskConfig


def _require_marl():
    try:
        import benchmarl  # noqa: F401
        import torchrl  # noqa: F401
        import tensordict  # noqa: F401
    except ImportError as exc:
        raise ImportError("MARL dependencies are not installed. Install with:\n\npip install 'bcod-sim[marl]'") from exc


def train_marl(recipe, run_dir: Path, seed: int, *, recipe_path: Path | None = None):
    _require_marl()
    env = recipe.environment
    if env.scenario != "M0":
        raise ValueError("Only the established M0 scenario sampler is currently implemented")
    from .scenarios import sample_m0_train
    from .train import train_one
    config = TaskConfig(agent_count=env.agent_count, dynamics=env.backend,
                        action_mode=env.action_mode, deadline_steps=env.deadline_steps)
    sampler = sample_m0_train
    stage_path = None
    if recipe.curriculum:
        from .curriculum_train import train_curriculum
        if recipe_path is None:
            raise ValueError("Curriculum requires its recipe file path")
        return train_curriculum(run_dir, seed, recipe_path)
    t = recipe.trainer
    return train_one(run_dir, seed, frames=t.total_steps or 24000,
        rollout_frames=t.rollout_frames or recipe.evaluation.interval_steps,
        envs=t.environments or 10, minibatch_size=t.minibatch_size or 400,
        minibatch_iters=t.minibatch_iterations or 45,
        task_config=config, scenario_sampler=sampler,
        centralized_state=bool(t.centralized_critic))


def evaluate_marl_run(run_dir: Path):
    _require_marl()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    curriculum = "recipe" in manifest and isinstance(manifest["recipe"], dict)
    if curriculum and manifest["recipe"].get("recipe_version", "").startswith("m0-"):
        from .curriculum_train import evaluate_selected
    else:
        from .train import evaluate_selected
    return evaluate_selected(run_dir)


def summarize_marl_run(run_dir: Path):
    # Summary consumes JSON artifacts only; no BenchMARL import is needed.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    result = {"run_dir": str(run_dir), "seed": manifest.get("seed"), **status}
    for key in ("dev_bank_hash", "test_bank_hash", "bank_hashes", "scenario_version"):
        if key in manifest:
            result[key] = manifest[key]
    target = run_dir / "test" / "best-dev-once.json"
    if target.exists():
        result["test"] = json.loads(target.read_text())
    return result
