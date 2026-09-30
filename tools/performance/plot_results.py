#!/usr/bin/env python3
"""Render figures from completed Task A/B CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(root / "multiprocess/summary.csv")
    rows = [r for r in rows if r.get("aggregate_env_steps_s_mean")]
    xs = [int(r["workers"]) for r in rows]
    ys = [float(r["aggregate_env_steps_s_mean"]) for r in rows]
    plt.figure(figsize=(6.4, 4.2)); plt.plot(xs, ys, "o-"); plt.xscale("log", base=2)
    plt.xticks(xs, [str(x) for x in xs]); plt.xlabel("Persistent worker processes")
    plt.ylabel("Aggregate environment steps / s"); plt.grid(True, alpha=.3); plt.tight_layout()
    plt.savefig(out / "multiprocess_throughput.png", dpi=180); plt.close()
    eff = [float(r["parallel_efficiency"]) for r in rows]
    plt.figure(figsize=(6.4, 4.2)); plt.plot(xs, eff, "o-"); plt.xscale("log", base=2)
    plt.xticks(xs, [str(x) for x in xs]); plt.xlabel("Persistent worker processes")
    plt.ylabel("Parallel efficiency"); plt.ylim(0, 1.05); plt.grid(True, alpha=.3); plt.tight_layout()
    plt.savefig(out / "parallel_efficiency.png", dpi=180); plt.close()
    prof = read_csv(root / "profile/grouped_profile.csv")
    prof = sorted(prof, key=lambda r: float(r["share_of_profiled_python_wall_pct"]))
    labels = [r["subsystem"].replace("_", " ") for r in prof]
    vals = [float(r["share_of_profiled_python_wall_pct"]) for r in prof]
    h = max(3.8, .36 * len(prof))
    plt.figure(figsize=(8, h)); plt.barh(labels, vals); plt.xlabel("Exclusive cProfile self time (%)")
    plt.tight_layout(); plt.savefig(out / "profile_breakdown.png", dpi=180); plt.close()


if __name__ == "__main__":
    main()
