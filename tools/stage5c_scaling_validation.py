"""Stage 5C CPU scaling, latency, stability and memory campaign."""

import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/validation"))
from test_stage5a_marl import command, make_world

from bcod_sim.core.engine import EpisodeEngine
from bcod_sim.core.environment_loads import LinearEnvironmentLoads
from bcod_sim.rl.vector_env import VectorEnvironment
from bcod_sim.sensors.base import SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.lidar import LiDAR
from bcod_sim.sensors.sonar import Sonar


PROCESS = psutil.Process()
SEED = 53
IRREGULAR = {"kind": "irregular", "spectrum": "jonswap",
             "significant_height_m": 0.01, "peak_period_s": 5.,
             "direction_rad": 0.2, "component_count": 8, "seed": 31}


def package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def names_for(count):
    return tuple(f"agent_{i:03d}" for i in range(count))


def make_case(count, workload="light", *, max_steps=128, contact="none"):
    names = names_for(count)
    sensors = {name: (("gps", 5 if i % 2 else 10, 0.01)
                      if workload != "light" else ("none", 10, 0))
               for i, name in enumerate(names)}
    current = [0.02, 0., 0.] if workload != "light" else [0, 0, 0]
    wind = [0.01, 0.01, 0.] if workload != "light" else [0, 0, 0]
    environment = {"current": {"kind": "uniform", "ned_mps": current},
                   "wind": {"kind": "uniform", "ned_mps": wind},
                   "waves": IRREGULAR if workload == "heavy" else {"kind": "calm"},
                   "visibility_m": 1000}
    loads = ({name: LinearEnvironmentLoads(0.01, 0.01, 0.01, 0.)
              for name in names} if workload != "light" else {})
    spawn = {name: ((i // 2) * 20. if contact == "sparse" else
                    0.1 * i if contact == "dense" else 20. * i, 0., 0.)
             for i, name in enumerate(names)}
    obstacles = ({"id": "marker", "position_ned_m": (5., 0., 0.),
                  "shape": {"kind": "sphere", "radius_m": 0.3}},) if workload != "light" else ()
    return make_world(names, max_steps=max_steps, goal=(1e9, 0., 0.),
        sensor_by_name=sensors, spawn_by_name=spawn,
        mass_by_name={name: 10 + (i % 4) * 2 for i, name in enumerate(names)},
        environment_config=environment, environment_loads_by_name=loads,
        static_entities=obstacles, bottom=2.)


def check_step(env, result, expected):
    observations, rewards, terms, truncs, infos = result
    if set(rewards) != expected or set(infos) != expected:
        raise AssertionError("Agent identity or reward mapping changed under load")
    if any(not math.isfinite(float(value)) for value in rewards.values()):
        raise AssertionError("Nonfinite reward under load")
    for name in expected:
        state = env.engine.states[name]
        if not all(torch.isfinite(value).all().item() for value in
                   (state.position_ned, state.q_body_to_ned, state.nu_body)):
            raise AssertionError(f"Nonfinite state under load: {name}")
        for value in observations.get(name, {}).values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                raise AssertionError(f"Nonfinite observation under load: {name}")
    return len({event.contact_id for info in infos.values()
                for event in info.get("contact_events", ())})


def summarize(label, count, latencies, wall, cpu, rss_before, rss_after, contacts, rep):
    values = np.asarray(latencies, dtype=float)
    return {"workload": label, "count": count, "repetition": rep,
            "steps": len(values), "steps_per_s": len(values) / wall,
            "mean_step_ms": 1000 * float(values.mean()),
            "p50_ms": 1000 * float(np.percentile(values, 50)),
            "p95_ms": 1000 * float(np.percentile(values, 95)),
            "p99_ms": 1000 * float(np.percentile(values, 99)),
            "cpu_pct_one_core": 100 * cpu / wall,
            "rss_before_mb": rss_before / 2**20,
            "rss_after_mb": rss_after / 2**20,
            "contacts": contacts, "realtime_factor": len(values) * 0.1 / wall}


def bench_agents(count, workload, *, reps=3, steps=12):
    rows = []
    for rep in range(reps):
        env = make_case(count, workload)
        env.reset(seed=SEED + rep)
        actions = {name: command(0) for name in env.agents}
        for _ in range(2):
            env.step(actions)
        env.reset(seed=SEED + rep)
        rss = PROCESS.memory_info().rss
        cpu = PROCESS.cpu_times()
        cpu_before = cpu.user + cpu.system
        latencies, contacts = [], 0
        start = time.perf_counter()
        for _ in range(steps):
            before = time.perf_counter()
            result = env.step(actions)
            latencies.append(time.perf_counter() - before)
            contacts += check_step(env, result, set(names_for(count)))
        wall = time.perf_counter() - start
        cpu = PROCESS.cpu_times()
        rows.append(summarize(workload, count, latencies, wall,
            cpu.user + cpu.system - cpu_before, rss, PROCESS.memory_info().rss,
            contacts, rep))
        env.close()
    return rows


def bench_vector(agent_count, env_count, *, reps=3, steps=8):
    rows = []
    names = names_for(agent_count)
    for rep in range(reps):
        engines = {index: make_case(agent_count, "light").engine
                   for index in range(env_count)}
        vector = VectorEnvironment(engines)
        seeds = {index: SEED + rep for index in engines}
        actions = {index: {name: command(0) for name in names} for index in engines}
        vector.reset(seeds=seeds)
        vector.step(actions)
        vector.reset(seeds=seeds)
        rss = PROCESS.memory_info().rss
        cpu = PROCESS.cpu_times(); cpu_before = cpu.user + cpu.system
        latencies = []
        start = time.perf_counter()
        for _ in range(steps):
            before = time.perf_counter()
            result = vector.step(actions)
            latencies.append(time.perf_counter() - before)
            if set(result.frames) != set(engines):
                raise AssertionError("Vector environment identity changed")
            for frame in result.frames.values():
                if set(frame.states) != set(names) or any(
                    not torch.isfinite(state.nu_body).all().item()
                    for state in frame.states.values()):
                    raise AssertionError("Vector state corrupted under load")
        wall = time.perf_counter() - start
        cpu = PROCESS.cpu_times()
        row = summarize("vector_light", env_count, latencies, wall,
            cpu.user + cpu.system - cpu_before, rss, PROCESS.memory_info().rss,
            0, rep)
        row.update({"agent_count": agent_count, "env_count": env_count,
                    "aggregate_env_steps_per_s": env_count * steps / wall,
                    "per_env_steps_per_s": steps / wall})
        rows.append(row)
    return rows


def sensor_case(kind):
    base = make_case(1, "medium").engine
    vessel = base.vessels[names_for(1)[0]]
    def sensor_config(sensor_id, rate):
        return SensorConfig(sensor_id, sensor_id, 0, vessel.vessel_id, (0, 0, 0),
            (1, 0, 0, 0), rate, 0, 0., 7, "1", "stage5c")
    sensors = []
    if kind != "dynamics":
        sensors += [GPS(sensor_config("gps", 10), origin_wgs84_rad_m=(0, 0, 0)),
                    IMU(sensor_config("imu", 10))]
    if kind in ("lidar", "mixed"):
        sensors.append(LiDAR(sensor_config("lidar", 5), min_range_m=0,
                             max_range_m=10, fov_rad=math.pi / 2, ray_count=8))
    if kind in ("sonar", "mixed"):
        sensors.append(Sonar(sensor_config("sonar", 5), min_range_m=0,
                             max_range_m=20, fov_rad=math.pi / 4, beam_count=4))
    new_vessel = replace(vessel, sensors=tuple(sensors))
    engine = EpisodeEngine(base.resolved, (new_vessel,))
    from bcod_sim.rl.pettingzoo_env import PettingZooParallelEnv
    return PettingZooParallelEnv(engine, observation_spaces={new_vessel.instance_id: object()},
        action_spaces={new_vessel.instance_id: object()})


def environment_case(kind):
    names = names_for(4)
    current = [0.02, 0.01, 0] if kind in ("current", "combined") else [0, 0, 0]
    wind = [0.02, -0.01, 0] if kind in ("wind", "combined") else [0, 0, 0]
    wave = (IRREGULAR if kind in ("irregular", "combined") else
            {"kind": "regular", "height_m": 0.02, "period_s": 5.,
             "direction_rad": 0.} if kind == "regular" else {"kind": "calm"})
    environment = {"current": {"kind": "uniform", "ned_mps": current},
                   "wind": {"kind": "uniform", "ned_mps": wind},
                   "waves": wave, "visibility_m": 1000}
    obstacles = ({"id": "marker", "position_ned_m": (5., 0., 0.),
                  "shape": {"kind": "sphere", "radius_m": 0.3}},) if kind in ("obstacles", "combined") else ()
    loads = {name: LinearEnvironmentLoads(0.01, 0.01, 0.01, 0.)
             for name in names}
    return make_world(names, max_steps=128, goal=(1e9, 0, 0),
        environment_config=environment, environment_loads_by_name=loads,
        static_entities=obstacles, bottom=2. if kind in ("bathymetry", "combined") else None)


def bench_custom(factory, label, *, reps=3, steps=12):
    rows = []
    for rep in range(reps):
        env = factory()
        env.reset(seed=SEED + rep)
        actions = {name: command(0) for name in env.agents}
        env.step(actions)
        env.reset(seed=SEED + rep)
        rss = PROCESS.memory_info().rss
        cpu = PROCESS.cpu_times(); cpu_before = cpu.user + cpu.system
        latencies, contacts = [], 0
        start = time.perf_counter()
        for _ in range(steps):
            before = time.perf_counter()
            result = env.step(actions)
            latencies.append(time.perf_counter() - before)
            contacts += check_step(env, result, set(actions))
        wall = time.perf_counter() - start
        cpu = PROCESS.cpu_times()
        rows.append(summarize(label, len(actions), latencies, wall,
            cpu.user + cpu.system - cpu_before, rss, PROCESS.memory_info().rss,
            contacts, rep))
    return rows


def write_csv(path, rows):
    if not rows:
        raise AssertionError(f"No benchmark data for {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def memory_test(out, *, steps=100000, sample_every=5000):
    env = make_case(1, "medium", max_steps=steps)
    env.reset(seed=SEED)
    actions = {name: command(0) for name in env.agents}
    rows = []
    for step in range(steps):
        result = env.step(actions)
        check_step(env, result, set(actions))
        if (step + 1) % sample_every == 0:
            gc.collect()
            rows.append({"step": step + 1, "rss_mb": PROCESS.memory_info().rss / 2**20,
                         "threads": PROCESS.num_threads(),
                         "scheduler_pending": len(env.engine.scheduler.pending),
                         "latest_packets": len(env.engine.latest_packets)})
    write_csv(out, rows)
    midpoint = rows[len(rows)//2]["rss_mb"]
    growth_mb = rows[-1]["rss_mb"] - midpoint
    if growth_mb > max(50., 0.05 * midpoint):
        raise AssertionError(f"Unexplained RSS growth in latter half: {growth_mb:.1f} MB")
    if max(row["scheduler_pending"] for row in rows) > 2:
        raise AssertionError("Sensor pending buffer grew unexpectedly")
    return {"steps": steps, "rss_latter_half_growth_mb": growth_mb,
            "rss_final_mb": rows[-1]["rss_mb"]}


def reset_cycles(out, *, cycles=200):
    env = make_case(4, "medium", max_steps=20)
    rows, reference = [], None
    for cycle in range(cycles):
        env.reset(seed=SEED)
        for _ in range(5):
            env.step({name: command(0) for name in env.agents})
        fingerprint = tuple(tuple(env.engine.states[name].position_ned.tolist())
                            for name in names_for(4))
        if reference is None: reference = fingerprint
        if fingerprint != reference:
            raise AssertionError("Fixed-seed reset cycle diverged")
        if cycle % 20 == 0:
            gc.collect()
            rows.append({"cycle": cycle, "rss_mb": PROCESS.memory_info().rss / 2**20,
                         "threads": PROCESS.num_threads()})
    write_csv(out, rows)
    if rows[-1]["rss_mb"] - rows[len(rows)//2]["rss_mb"] > 50:
        raise AssertionError("Reset cycling leaked RSS")
    return {"cycles": cycles, "rss_final_mb": rows[-1]["rss_mb"]}


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "stage5_results" / f"stage5c-{run_id}"
    for folder in ("benchmark_configs", "plots", "failures"):
        (out / folder).mkdir(parents=True)
    manifest = {"run_id": run_id,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "hardware": {"platform": platform.platform(), "cpu_count_logical": psutil.cpu_count(),
                     "cpu_count_physical": psutil.cpu_count(logical=False),
                     "ram_bytes": psutil.virtual_memory().total,
                     "gpu_available": torch.cuda.is_available() or torch.backends.mps.is_available()},
        "python": platform.python_version(),
        "dependencies": {name: package_version(name) for name in
                         ("torch", "numpy", "psutil", "pytest", "pettingzoo")},
        "threads": {name: os.environ.get(name) for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "torch_threads": torch.get_num_threads(), "seed": SEED,
        "master_dt_s": 0.1, "warmup_steps": 2,
        "repetitions": 3,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "benchmark_configs" / "workloads.json").write_text(json.dumps({
        "light": "calm, no sensors", "medium": "GPS, current, wind, obstacle",
        "heavy": "heterogeneous masses/sensors, irregular waves, obstacle",
        "agent_counts": [1, 2, 4, 8, 16, 32, 64],
        "env_counts": [1, 2, 4, 8, 16, 32, 64]}, indent=2) + "\n")
    try:
        agents = [row for workload in ("light", "medium", "heavy")
                  for count in (1, 2, 4, 8, 16, 32, 64)
                  for row in bench_agents(count, workload)]
        write_csv(out / "agent_scaling.csv", agents)
        envs = [row for count in (1, 2, 4, 8, 16, 32, 64)
                for row in bench_vector(1, count)]
        write_csv(out / "env_scaling.csv", envs)
        pairs = ((1, 1), (1, 16), (4, 4), (4, 16),
                 (16, 1), (16, 4), (32, 1), (32, 4))
        matrix = [row for agents_count, env_count in pairs
                  for row in bench_vector(agents_count, env_count, reps=3, steps=5)]
        write_csv(out / "scale_matrix.csv", matrix)
        sensors = [row for kind in ("dynamics", "gps_imu", "lidar", "sonar", "mixed")
                   for row in bench_custom(lambda kind=kind: sensor_case(kind), kind)]
        write_csv(out / "sensors.csv", sensors)
        environments = [row for kind in ("still", "current", "wind", "regular",
            "irregular", "combined", "bathymetry", "obstacles")
            for row in bench_custom(lambda kind=kind: environment_case(kind), kind)]
        write_csv(out / "environment_cost.csv", environments)
        collisions = [row for density in ("none", "sparse", "dense")
                      for row in bench_custom(lambda density=density:
                          make_case(4, "medium", contact=density), density)]
        write_csv(out / "collision_density.csv", collisions)
        write_csv(out / "cpu_gpu.csv", [{"path": "CPU", "supported": True,
            "benchmark": "agent_scaling.csv"}, {"path": "GPU", "supported": False,
            "benchmark": "not available in current torch runtime"}])
        memory = memory_test(out / "memory.csv")
        resets = reset_cycles(out / "reset_cycles.csv")
        practical = {"tested_agent_count": 64, "tested_env_count": 64,
            "maximum_supported": "at least tested boundary; no universal maximum claimed"}
        rows = [row for row in agents if row["workload"] == "light"]
        count_values = (1, 2, 4, 8, 16, 32, 64)
        means = {count: statistics.mean(row["steps_per_s"] for row in rows
                                  if row["count"] == count)
                 for count in count_values}
        reproducibility = {}
        for workload in ("light", "medium", "heavy"):
            for count in count_values:
                values = [row["steps_per_s"] for row in agents
                          if row["workload"] == workload and row["count"] == count]
                mean = statistics.mean(values)
                reproducibility[f"{workload}:{count}"] = {
                    "mean": mean, "std": statistics.stdev(values),
                    "cv": statistics.stdev(values) / mean}
        baseline = means[1]
        scaling_efficiency = {count: means[count] / baseline
                              for count in count_values}
        summary = {"status": "PASS", "agent_steps_per_s_mean": means,
                   "scaling_efficiency_relative_to_one_agent": scaling_efficiency,
                   "reproducibility": reproducibility,
                   "memory": memory, "reset_cycles": resets,
                   "capacity": practical, "gpu": "UNSUPPORTED",
                   "benchmarks": {"agent_rows": len(agents), "env_rows": len(envs),
                                  "matrix_rows": len(matrix), "sensor_rows": len(sensors),
                                  "environment_rows": len(environments),
                                  "collision_rows": len(collisions)}}
        os.environ.setdefault("MPLCONFIGDIR", str(out / "plots" / ".mpl"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for workload in ("light", "medium", "heavy"):
            ax.plot(count_values, [statistics.mean(row["steps_per_s"] for row in agents
                if row["workload"] == workload and row["count"] == count)
                for count in count_values], marker="o", label=workload)
        ax.set(xlabel="Agents", ylabel="Environment steps per second",
               title="Stage 5C agent scaling")
        ax.set_xscale("log", base=2); ax.legend(); fig.tight_layout()
        fig.savefig(out / "plots" / "agent_scaling.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots()
        env_counts = (1, 2, 4, 8, 16, 32, 64)
        ax.plot(env_counts, [statistics.mean(row["aggregate_env_steps_per_s"]
            for row in envs if row["env_count"] == count) for count in env_counts], marker="o")
        ax.set(xlabel="Parallel environments", ylabel="Aggregate environment steps per second",
               title="Stage 5C vector scaling")
        ax.set_xscale("log", base=2); fig.tight_layout()
        fig.savefig(out / "plots" / "env_scaling.png", dpi=160); plt.close(fig)
    except Exception as exc:
        (out / "failures" / "exception.txt").write_text(repr(exc) + "\n")
        summary = {"status": "FAIL", "error": repr(exc)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "report.md").write_text("# Stage 5C scaling validation\n\n"
        f"Status: **{summary['status']}**. Hardware-specific CPU results are in the CSV files. "
        "Throughput and latency are observations, not arbitrary pass thresholds. "
        "GPU execution is unsupported in this runtime.\n\n"
        f"Memory: {summary.get('memory', 'not completed')}.\n\n"
        f"Capacity: {summary.get('capacity', 'not established')}.\n")
    print(out)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
