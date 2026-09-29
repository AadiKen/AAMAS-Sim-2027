"""Unified paper benchmark commands; trainer implementations stay in their modules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import yaml

from .recipe import load_recipe


def _sarl_train(recipe_path, run_dir, seed):
    from bcod_sim.benchmark_v3.runner import train_recipe
    return train_recipe(load_recipe(recipe_path), Path(run_dir), seed=seed)


def train(recipe_path, run_dir, *, seed=None, all_seeds=False):
    recipe = load_recipe(recipe_path)
    seeds = recipe.seeds if all_seeds else (seed if seed is not None else recipe.seeds[0],)
    if any(item not in recipe.seeds for item in seeds):
        raise ValueError(f"Requested seed must be declared in recipe: {recipe.seeds}")
    root = Path(run_dir)
    if all_seeds and root.exists() and any((root / f"seed-{item}").exists() for item in seeds):
        raise FileExistsError("At least one requested seed run directory already exists")
    outputs = []
    for item in seeds:
        target = root / f"seed-{item}" if all_seeds or len(seeds) > 1 else root
        # Existing trainer creates its run directory atomically.
        if recipe.trainer.library == "stable_baselines3":
            result = _sarl_train(recipe_path, target, item)
        else:
            try:
                from bcod_sim.benchmark_v3.marl.dispatch import train_marl
            except ImportError as exc:
                raise ImportError("MARL dependencies are not installed. Install with:\n\npip install 'bcod-sim[marl]'") from exc
            result = train_marl(recipe, target, item, recipe_path=Path(recipe_path))
        shutil.copy2(recipe_path, target / "recipe.yaml")
        (target / "recipe.resolved.yaml").write_text(yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=True))
        outputs.append(result)
    return outputs[0] if len(outputs) == 1 else outputs


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bcod benchmark")
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check"); check.add_argument("recipe", type=Path)
    train_p = commands.add_parser("train"); train_p.add_argument("recipe", type=Path)
    train_p.add_argument("--run-dir", type=Path); train_p.add_argument("--seed", type=int)
    train_p.add_argument("--all-seeds", action="store_true")
    for name in ("evaluate", "summarize"):
        item = commands.add_parser(name); item.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    if args.action == "check":
        recipe = load_recipe(args.recipe)
        if recipe.trainer.library == "benchmarl":
            from bcod_sim.benchmark_v3.config import TaskConfig
            env = recipe.environment
            TaskConfig(agent_count=env.agent_count, dynamics=env.backend,
                       action_mode=env.action_mode, deadline_steps=env.deadline_steps)
        print(yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=True))
        return 0
    if args.action == "train":
        if args.seed is not None and args.all_seeds: parser.error("--seed and --all-seeds are mutually exclusive")
        default_root = Path("runs") / args.recipe.stem
        dest = args.run_dir or (default_root / f"seed-{args.seed}" if args.seed is not None and not args.all_seeds else default_root)
        try:
            result = train(args.recipe, dest, seed=args.seed, all_seeds=args.all_seeds)
        except ImportError as exc:
            parser.error(str(exc))
        print(result); return 0
    manifest = args.run / "manifest.json"
    if not manifest.exists(): raise FileNotFoundError(manifest)
    data = json.loads(manifest.read_text())
    trainer, algorithm = data.get("trainer"), data.get("algorithm")
    if trainer == "stable_baselines3" and algorithm == "PPO":
        from bcod_sim.benchmark_v3.runner import evaluate_run, summarize
        result = evaluate_run(args.run) if args.action == "evaluate" else summarize(args.run)
    elif trainer == "benchmarl" and algorithm == "MAPPO":
        try:
            from bcod_sim.benchmark_v3.marl.dispatch import evaluate_marl_run, summarize_marl_run
        except ImportError as exc:
            raise ImportError("MARL dependencies are not installed. Install with:\n\npip install 'bcod-sim[marl]'") from exc
        try:
            result = evaluate_marl_run(args.run) if args.action == "evaluate" else summarize_marl_run(args.run)
        except ImportError as exc:
            parser.error(str(exc))
    else:
        raise ValueError(f"Run manifest has unsupported trainer/algorithm: {trainer!r}/{algorithm!r}")
    print(result); return 0
