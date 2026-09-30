"""Extend an EOD BenchMARL run from a full BenchMARL experiment checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from benchmarl.experiment import Experiment

from .eod_benchmarl import EODCallback, SeededM0Sampler, _write_scalar_training_csv

# The initial ``python -m ...eod_benchmarl`` run pickled these top-level
# classes as __main__ objects. Restore those names for pickle compatibility.
sys.modules["__main__"].EODCallback = EODCallback
sys.modules["__main__"].SeededM0Sampler = SeededM0Sampler


def resume(run_dir: Path, target_frames: int):
    run_dir = run_dir.resolve()
    status_path = run_dir / "status.json"
    checkpoint = (run_dir / "benchmarl").glob("*/checkpoints/checkpoint_*.pt")
    candidates = sorted(checkpoint, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not candidates:
        raise FileNotFoundError("No full BenchMARL experiment checkpoints found")
    source = candidates[-1]
    checkpoint_frames = int(source.stem.rsplit("_", 1)[1])
    if checkpoint_frames >= target_frames:
        raise ValueError("Target must exceed the latest full-checkpoint frame count")
    exp = Experiment.reload_from_file(str(source), experiment_patch={"max_n_frames": target_frames})
    callback = exp.callbacks[0]
    callback.out = run_dir
    previous_wall = 0.0
    development_path = run_dir / "development.csv"
    if development_path.exists():
        with development_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            previous_wall = float(rows[-1].get("wall_seconds", 0.0))
    elif status_path.exists():
        previous_wall = float(json.loads(status_path.read_text()).get("wall_seconds", 0.0))
    callback.started = time.monotonic() - previous_wall
    prior_best = json.loads((run_dir / "best_dev.json").read_text())
    callback.best = tuple(prior_best["score"])
    callback.last_eval = checkpoint_frames
    start = time.monotonic()
    succeeded = False
    try:
        exp.run()
        succeeded = True
    except BaseException as error:
        (run_dir / "resume_failure.json").write_text(json.dumps({
            "error": repr(error), "frames": int(exp.total_frames),
            "resumed_from": str(source)}, indent=2) + "\n")
        raise
    finally:
        if succeeded and callback.last_eval != int(exp.total_frames):
            callback.evaluate(int(exp.total_frames), checkpoint=True)
        torch.save(exp.policy.state_dict(), run_dir / "checkpoints/final-actor.pt")
        wall = time.monotonic() - callback.started
        status = {"status": "complete" if succeeded else "failed",
                  "environment_steps": int(exp.total_frames),
                  "iterations": exp.n_iters_performed, "wall_seconds": wall,
                  "steps_per_second": exp.total_frames / max(wall, 1e-9),
                  "wall_time_includes_initial_and_development_evaluations": True,
                  "resume_wall_seconds": time.monotonic() - start,
                  "resumed_from": str(source)}
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    _write_scalar_training_csv(run_dir)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(resume(args.run_dir, args.target_frames), indent=2))


if __name__ == "__main__":
    main()
