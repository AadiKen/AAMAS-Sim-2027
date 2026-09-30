#!/usr/bin/env python3
"""Persistent one-environment-per-process CPU scaling benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import statistics
import subprocess
import time
import traceback
from pathlib import Path

from common import ROOT


def physical_cpus():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    try:
        out = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)
        cores = {}
        for line in out.splitlines():
            if line.startswith("#"):
                continue
            cpu, core, socket = map(int, line.split(","))
            if cpu in allowed:
                cores.setdefault((socket, core), cpu)
        if cores:
            return sorted(cores.values())
    except Exception:
        pass
    # Conservative sysfs fallback. If topology cannot be proven, do not claim
    # one worker per physical core: permit only one worker.
    try:
        cores = {}
        for cpu in allowed:
            base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            package = int((base / "physical_package_id").read_text())
            core = int((base / "core_id").read_text())
            cores.setdefault((package, core), cpu)
        return sorted(cores.values()) or [allowed[0]]
    except Exception:
        return [allowed[0]] if allowed else []


def worker(index, cpu, steps, warmup, ready, go, result_q):
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {cpu})
        from common import make_env
        env, actions = make_env()
        for _ in range(warmup):
            env.step(actions)
        env.reset(seed=53)
        usage0 = resource.getrusage(resource.RUSAGE_SELF)
        ready.put({"index": index, "pid": os.getpid(), "cpu": cpu,
                   "startup_s": time.perf_counter(), "rss_before": usage0.ru_maxrss * 1024})
        go.wait()
        latencies = []
        cpu0 = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter()
        for _ in range(steps):
            t0 = time.perf_counter_ns()
            env.step(actions)
            latencies.append((time.perf_counter_ns() - t0) / 1e9)
        wall = time.perf_counter() - start
        cpu1 = resource.getrusage(resource.RUSAGE_SELF)
        rss = cpu1.ru_maxrss * 1024
        result_q.put({"index": index, "pid": os.getpid(), "steps": steps,
                      "wall_s": wall, "cpu_s": (cpu1.ru_utime + cpu1.ru_stime) - (cpu0.ru_utime + cpu0.ru_stime),
                      "rss_bytes": rss, "latencies_s": latencies})
        env.close()
    except BaseException:
        result_q.put({"index": index, "error": traceback.format_exc()})


def csv_append(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())


def collect_until_exit(q, procs, expected, timeout_s):
    values = []
    deadline = time.monotonic() + timeout_s
    while len(values) < expected and time.monotonic() < deadline:
        try:
            values.append(q.get(timeout=0.5))
        except queue.Empty:
            if all(not proc.is_alive() for proc in procs):
                break
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--counts", default="1,4,16,64")
    p.add_argument("--output", type=Path, default=ROOT / "paper_results/eod_performance/multiprocess")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cpus = physical_cpus()
    allocation = len(cpus)
    requested = [int(x) for x in args.counts.split(",")]
    effective = sorted(set(min(n, allocation) for n in requested))
    fallback_order = [64, 32, 16, 8, 4, 1]
    ctx = mp.get_context("spawn")
    raw_fields = ["requested_workers", "workers", "repetition", "steps_per_worker", "total_active_envs",
                  "aggregate_env_steps_s", "aggregate_vessel_steps_s", "mean_worker_step_ms",
                  "p50_step_ms", "p95_step_ms", "p99_step_ms", "total_rss_mb", "rss_per_worker_mb",
                  "cpu_utilization_pct", "startup_overhead_s", "steady_wall_s", "failure"]
    summary_fields = ["workers", "repetitions", "aggregate_env_steps_s_mean", "aggregate_env_steps_s_sd",
                      "aggregate_vessel_steps_s_mean", "p50_step_ms_median", "p95_step_ms_median",
                      "p99_step_ms_median", "total_rss_mb_mean", "rss_per_worker_mb_mean",
                      "cpu_utilization_pct_mean", "startup_overhead_s_mean", "speedup_vs_1w", "parallel_efficiency"]
    raw_path = args.output / "raw.csv"
    # Preserve data from partial/retried invocations in separate append-only rows.
    by_count = {}
    for requested_count in requested:
        count = min(requested_count, allocation)
        if count != requested_count:
            print(f"CAP: requested {requested_count} workers, allocation has {allocation} physical CPUs; using {count}", flush=True)
        completed = []
        reps = args.reps
        retry_counts = [count] + [n for n in fallback_order if n < count and n not in effective]
        for worker_count in retry_counts:
            while reps >= 1 and len(completed) < args.reps:
                for rep in range(len(completed), reps):
                    ready_q, result_q, go = ctx.Queue(), ctx.Queue(), ctx.Event()
                    startup_begin = time.perf_counter()
                    procs = [ctx.Process(target=worker, args=(i, cpus[i % allocation], args.steps,
                              args.warmup, ready_q, go, result_q)) for i in range(worker_count)]
                    for proc in procs: proc.start()
                    ready = collect_until_exit(ready_q, procs, worker_count, 300)
                    if len(ready) != worker_count:
                        for proc in procs: proc.terminate()
                        for proc in procs: proc.join()
                        row = {"requested_workers": requested_count, "workers": worker_count,
                               "repetition": rep + 1, "failure": f"worker startup failed ({len(ready)}/{worker_count} ready)"}
                        csv_append(raw_path, [row], raw_fields)
                        continue
                    startup_s = time.perf_counter() - startup_begin
                    go.set()
                    results = collect_until_exit(result_q, procs, worker_count, 600)
                    for proc in procs: proc.join(timeout=5)
                    err = next((x.get("error") for x in results if x.get("error")), None)
                    if err or len(results) != worker_count:
                        row = {"requested_workers": requested_count, "workers": worker_count,
                               "repetition": rep + 1, "failure": err or f"received {len(results)}/{worker_count} worker records"}
                        csv_append(raw_path, [row], raw_fields)
                        continue
                    walls = [x["wall_s"] for x in results]
                    wall = max(walls)
                    all_lat = [v for x in results for v in x["latencies_s"]]
                    total_steps = sum(x["steps"] for x in results)
                    total_rss = sum(x["rss_bytes"] for x in results)
                    cpu_s = sum(x["cpu_s"] for x in results)
                    pct = lambda q: 1000 * sorted(all_lat)[min(len(all_lat)-1, round((len(all_lat)-1)*q))]
                    row = {"requested_workers": requested_count, "workers": worker_count,
                           "repetition": rep + 1, "steps_per_worker": args.steps,
                           "total_active_envs": worker_count,
                           "aggregate_env_steps_s": total_steps / wall,
                           "aggregate_vessel_steps_s": total_steps * 4 / wall,
                           "mean_worker_step_ms": 1000 * statistics.mean(all_lat),
                           "p50_step_ms": pct(.50), "p95_step_ms": pct(.95), "p99_step_ms": pct(.99),
                           "total_rss_mb": total_rss / 2**20,
                           "rss_per_worker_mb": total_rss / worker_count / 2**20,
                           "cpu_utilization_pct": 100 * cpu_s / wall,
                           "startup_overhead_s": startup_s, "steady_wall_s": wall, "failure": ""}
                    csv_append(raw_path, [row], raw_fields)
                    completed.append(row)
                if len(completed) < min(args.reps, reps):
                    break
                break
            if completed:
                worker_count = completed[-1]["workers"]
                break
            reps = max(1, reps // 2)
            if worker_count == 1:
                break
        by_count[requested_count] = completed

    # Aggregate successful rows by actual worker count; speedup uses measured one-worker baseline.
    valid = [r for rows in by_count.values() for r in rows]
    actual_counts = sorted(set(r["workers"] for r in valid))
    one = [r["aggregate_env_steps_s"] for r in valid if r["workers"] == 1]
    baseline = statistics.mean(one) if one else None
    summaries, speed_rows = [], []
    for n in actual_counts:
        rows = [r for r in valid if r["workers"] == n]
        mean = lambda k: statistics.mean(r[k] for r in rows)
        throughputs = [r["aggregate_env_steps_s"] for r in rows]
        speedup = mean("aggregate_env_steps_s") / baseline if baseline else None
        summaries.append({"workers": n, "repetitions": len(rows),
            "aggregate_env_steps_s_mean": mean("aggregate_env_steps_s"),
            "aggregate_env_steps_s_sd": statistics.stdev(throughputs) if len(rows)>1 else 0,
            "aggregate_vessel_steps_s_mean": mean("aggregate_vessel_steps_s"),
            "p50_step_ms_median": statistics.median(r["p50_step_ms"] for r in rows),
            "p95_step_ms_median": statistics.median(r["p95_step_ms"] for r in rows),
            "p99_step_ms_median": statistics.median(r["p99_step_ms"] for r in rows),
            "total_rss_mb_mean": mean("total_rss_mb"), "rss_per_worker_mb_mean": mean("rss_per_worker_mb"),
            "cpu_utilization_pct_mean": mean("cpu_utilization_pct"), "startup_overhead_s_mean": mean("startup_overhead_s"),
            "speedup_vs_1w": speedup, "parallel_efficiency": speedup / n if speedup is not None else None})
        speed_rows.append({"workers": n, "speedup": speedup,
                           "parallel_efficiency": speedup / n if speedup is not None else None})
    for filename, rows, fields in (("summary.csv", summaries, summary_fields),
                                   ("speedup.csv", speed_rows, ["workers", "speedup", "parallel_efficiency"])):
        with (args.output / filename).open("w", newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


if __name__ == "__main__":
    main()
