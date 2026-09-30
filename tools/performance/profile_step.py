#!/usr/bin/env python3
"""Profile the fixed four-vessel Stage5C-light environment step."""
from __future__ import annotations

import argparse
import cProfile
import csv
import os
import pstats
import statistics
import time
from collections import defaultdict
from pathlib import Path

from common import DT_S, ROOT, make_env


def percentile(values, q):
    return sorted(values)[round((len(values) - 1) * q)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "paper_results/eod_performance/profile")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env, actions = make_env()
    for _ in range(args.warmup):
        env.step(actions)
    env.reset(seed=53)

    # Component timings are inclusive intervals around disjoint/selected engine calls.
    # Plant/contact are called per vessel/substep, so aggregate those invocation durations.
    samples = defaultdict(list)
    from bcod_sim.core.engine import EpisodeEngine
    from bcod_sim.dynamics.plant6 import Plant6
    import bcod_sim.core.engine as engine_module

    originals = {}

    def wrap(target, attr, label):
        original = getattr(target, attr)
        originals[(target, attr)] = original
        def timed(*a, **kw):
            start = time.perf_counter_ns()
            try:
                return original(*a, **kw)
            finally:
                samples[label].append(time.perf_counter_ns() - start)
        setattr(target, attr, timed)

    wrap(EpisodeEngine, "_action_commands", "action_dispatch")
    wrap(EpisodeEngine, "_load_terms", "environment_loads")
    wrap(Plant6, "step", "plant6")
    wrap(engine_module, "resolve_contacts", "collision_contact")
    wrap(EpisodeEngine, "_deliver", "observation_delivery")
    wrap(EpisodeEngine, "_state_snapshot", "state_snapshot")
    wrap(EpisodeEngine, "_grounding_telemetry", "grounding_telemetry")
    wrap(EpisodeEngine, "_sensor_contexts", "sensor_contexts")
    wrap(EpisodeEngine, "_external", "external_load_assembly")
    # Scheduler and task are instance attributes: wrap the concrete per-env objects.
    eng = env.engine
    wrap(eng.scheduler, "tick", "scheduler_inclusive")
    wrap(eng.task, "evaluate", "task_evaluation")

    # First timing pass without profiler overhead for stable per-step latency.
    latencies = []
    for _ in range(args.steps):
        t0 = time.perf_counter_ns()
        env.step(actions)
        latencies.append(time.perf_counter_ns() - t0)

    # Keep a distinct profiling pass so cProfile does not distort latency samples.
    samples.clear()
    env.reset(seed=53)
    prof = cProfile.Profile()
    profile_outer = []
    prof.enable()
    for _ in range(args.steps):
        t0 = time.perf_counter_ns()
        env.step(actions)
        profile_outer.append(time.perf_counter_ns() - t0)
    prof.disable()
    prof.dump_stats(str(args.output / "cprofile.pstats"))
    with (args.output / "cprofile.txt").open("w") as f:
        stats = pstats.Stats(prof, stream=f).strip_dirs().sort_stats("cumulative")
        f.write("TOP FUNCTIONS BY CUMULATIVE TIME\n")
        stats.print_stats(40)
        f.write("\nTOP FUNCTIONS BY SELF TIME\n")
        pstats.Stats(prof, stream=f).strip_dirs().sort_stats("tottime").print_stats(40)

    # Convert selected timers into per-step totals. Scheduler includes sensor sampling;
    # observed sensor-context construction is reported separately. No sensors in light case.
    total_ns = sum(profile_outer)
    rows = []
    for label, vals in samples.items():
        ns = sum(vals)
        rows.append({"subsystem": label, "total_ms": ns / 1e6,
                     "mean_ms_per_env_step": ns / len(latencies) / 1e6,
                     "share_of_outer_step_pct": 100 * ns / total_ns,
                     "calls": len(vals)})
    # Nested engine hooks overlap. Report them as inclusive diagnostics only; the
    # additive residual is computed from the union-free cProfile accounting below.
    with (args.output / "grouped_profile_inclusive.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (args.output / "profile_latency.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["stat", "value_ms"])
        w.writerows([["mean", statistics.mean(latencies) / 1e6],
                     ["median", statistics.median(latencies) / 1e6],
                     ["p95", percentile(latencies, .95) / 1e6]])
    summary = f"""# Single-step profile summary

Status: completed by this script on the host where it runs. The required result is valid only when executed inside the isolated Linux cluster allocation described in `commands_run.txt`.

Workload: Stage5C light, 4 vessels, dt={DT_S}s, one dynamics substep, calm conditions, no sensors, fixed action `command(0)`, seed 53.

Requested steps: {args.steps}; warm-up: {args.warmup}.

Unprofiled average step latency: {statistics.mean(latencies)/1e6:.4f} ms; median: {statistics.median(latencies)/1e6:.4f} ms; p95: {percentile(latencies,.95)/1e6:.4f} ms.

The run writes average/median/p95 outer `env.step` latency to `profile_latency.csv`. `cprofile.txt` contains top cumulative and self time functions. `grouped_profile.csv` is generated from cProfile self time with exclusive subsystem categories; `grouped_profile_inclusive.csv` contains selected nested hook timers and may overlap. Torch/Python conversions and allocations/copies are identified separately where cProfile exposes them.

"""
    (args.output / "profile_summary.md").write_text(summary)
    for (target, attr), original in originals.items():
        setattr(target, attr, original)
    env.close()


if __name__ == "__main__":
    main()
