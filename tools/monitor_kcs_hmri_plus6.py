"""Live stationarity and numeric gate for the sole HMRI +6 degree case."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np

from bcod_sim.vessel_generation.hmri_wrench import separate_openfoam_wrench
from bcod_sim.vessel_generation.quality import wrench_stationarity

ROOT = Path("stage3_results/kcs-validation/track_b_hmri/static_drift_plus6_wall_ramped")
FORCES_DIR = ROOT / "postProcessing/forces"
LOG = ROOT / "solver.log"
HISTORY = ROOT / "monitor_history.jsonl"
SUMMARY = ROOT / "monitor_summary.json"
CONTROL = ROOT / "system/controlDict"
EXIT = ROOT / "solver.exit"
FLOATS = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
RAMP_END = 1.0
MIN_STEADY_TEST_TIME = 4.0  # ramp plus at least one Lpp/U hull transit
WINDOW = 1.0
POLL_S = 15
MAX_WALL_S = 3000


def force_history() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = []
    for path in FORCES_DIR.glob("*/forces.dat"):
        try:
            files.append((float(path.parent.name), path))
        except ValueError:
            continue
    if not files:
        return np.empty(0), np.empty((0, 6)), np.empty((0, 6))
    rows = []
    for _, path in sorted(files):
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                continue
            values = [float(x) for x in FLOATS.findall(line)]
            if len(values) == 13:
                rows.append(values)
    if not rows:
        return np.empty(0), np.empty((0, 6)), np.empty((0, 6))
    by_time = {row[0]: row for row in rows}
    raw = np.asarray([by_time[t] for t in sorted(by_time)])
    physical, resisting = [], []
    for row in raw:
        pair = separate_openfoam_wrench(
            row[1:4] + row[4:7], row[7:10] + row[10:13],
            foam_reference=(-0.0851, 0, 0), body_reference=(-0.0851, 0, 0),
            waterline_z_m=0,
        )
        physical.append([*pair[0].force_frd_n, *pair[0].moment_frd_nm])
        resisting.append([*pair[1].force_frd_n, *pair[1].moment_frd_nm])
    return raw[:, 0], np.asarray(physical), np.asarray(resisting)


def last(pattern: str, content: str) -> float | None:
    matches = re.findall(pattern, content)
    return float(matches[-1]) if matches else None


def yplus() -> dict | None:
    files = sorted((ROOT / "postProcessing/yPlus").glob("*/yPlus.dat"))
    result = None
    for path in files:
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 5 and fields[1] == "hull":
                try:
                    value = {"time_s": float(fields[0]), "min": float(fields[2]),
                             "max": float(fields[3]), "mean": float(fields[4])}
                except ValueError:
                    continue
                if result is None or value["time_s"] >= result["time_s"]:
                    result = value
    if result is None:
        return None
    available = []
    for field in ROOT.glob("processor*/*/yPlus"):
        try:
            field_time = float(field.parent.name)
        except ValueError:
            continue
        if field_time <= result["time_s"] + 1e-8:
            available.append((field_time, field))
    fields = []
    if available:
        distribution_time = max(time for time, _ in available)
        fields = [field for time, field in available if time == distribution_time]
        result["distribution_time_s"] = distribution_time
    all_values = []
    for field in fields:
        if not field.exists():
            continue
        match = re.search(r"\bhull\s*\{.*?value\s+nonuniform List<scalar>\s+(\d+)\s*\(\s*(.*?)\s*\)",
                          field.read_text(), re.S)
        if match:
            values = np.fromstring(match.group(2), sep=" ")
            if len(values) == int(match.group(1)):
                all_values.append(values)
    if all_values:
        values = np.concatenate(all_values)
        result["distribution_all_hull"] = {
            "faces": len(values), "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "fraction_30_to_300": float(np.mean((values >= 30) & (values <= 300))),
        }
    return result


def stop_at_write_now(reason: str) -> None:
    content = CONTROL.read_text()
    if "stopAt endTime;" in content:
        CONTROL.write_text(content.replace("stopAt endTime;", "stopAt writeNow;"))
    print(f"Requested solver stop: {reason}", flush=True)


def snapshot(t: np.ndarray, physical: np.ndarray, resisting: np.ndarray,
             log: str) -> tuple[dict, bool]:
    now = float(t[-1]); window = min(WINDOW, now - RAMP_END)
    mask = (t >= max(RAMP_END, now - WINDOW)) if window > 0 else (t >= 0)
    ts = t[mask]; values = physical[mask]
    channels = {}
    for name, col in (("X", 0), ("Y", 1), ("N", 5)):
        signal = values[:, col]
        midpoint = (ts[0]+ts[-1])/2 if len(ts) else 0
        first = signal[ts < midpoint]; second = signal[ts >= midpoint]
        channels[name] = {
            "mean": float(signal.mean()),
            "slope_per_s": float(np.polyfit(ts, signal, 1)[0]) if len(ts) >= 3 else None,
            "first_block_mean": float(first.mean()) if len(first) else None,
            "second_block_mean": float(second.mean()) if len(second) else None,
            "variance": float(signal.var()),
        }
    accepted = False
    if now >= MIN_STEADY_TEST_TIME and ts[-1]-ts[0] >= .9*WINDOW:
        only_active = np.zeros_like(values)
        only_active[:, [0, 1, 5]] = values[:, [0, 1, 5]]
        gate = wrench_stationarity(ts, only_active, force_scale_n=2960.77973625,
                                   moment_scale_nm=17024.4834834375, window_s=WINDOW)
        accepted = all(gate.channels[i].accepted for i in (0, 1, 5))
        for name, i in (("X", 0), ("Y", 1), ("N", 5)):
            channel = gate.channels[i]
            channels[name]["normalized_mean_change"] = channel.normalized_mean_change
            channels[name]["normalized_slope"] = channel.normalized_slope
            channels[name]["normalized_std"] = channel.normalized_standard_deviation
            channels[name]["accepted"] = channel.accepted
    result = {
        "time_s": now, "window_start_s": float(ts[0]), "ramp_excluded": bool(ts[0] >= RAMP_END),
        "physical_wrench_mean_X_Y_N": [channels[x]["mean"] for x in ("X", "Y", "N")],
        "bcod_resisting_wrench_mean_X_Y_N": [float(resisting[mask, i].mean()) for i in (0, 1, 5)],
        "channels": channels, "stationary_active_channels": accepted,
        "yplus_hull": yplus(),
        "deltaT_s": last(r"deltaT = ([\d.eE+-]+)", log),
        "maxCo_latest": last(r"(?m)^Courant Number mean: [^\n]* max: ([\d.eE+-]+)", log),
        "maxAlphaCo_latest": last(r"Interface Courant Number mean: [^\n]* max: ([\d.eE+-]+)", log),
        "p_initial_residual_latest": last(r"Solving for p_rgh, Initial residual = ([\d.eE+-]+)", log),
        "U_initial_residual_latest": last(r"Solving for U(?:x|y|z)?, Initial residual = ([\d.eE+-]+)", log),
        "k_initial_residual_latest": last(r"Solving for k, Initial residual = ([\d.eE+-]+)", log),
        "omega_initial_residual_latest": last(r"Solving for omega, Initial residual = ([\d.eE+-]+)", log),
    }
    return result, accepted


def main() -> None:
    start = time.monotonic(); last_bucket = -1; last_accept_time = None
    reason = None; latest = None
    while True:
        wall = time.monotonic() - start
        log = LOG.read_text() if LOG.exists() else ""
        t, physical, resisting = force_history()
        if len(t):
            bucket = int(t[-1] / 0.5)
            if bucket > last_bucket:
                latest, accepted = snapshot(t, physical, resisting, log)
                latest["monitor_wall_s"] = wall
                with HISTORY.open("a") as output:
                    output.write(json.dumps(latest) + "\n")
                print(json.dumps({"time_s": latest["time_s"],
                                  "physical_X_Y_N": latest["physical_wrench_mean_X_Y_N"],
                                  "stationary": accepted,
                                  "yplus": latest["yplus_hull"],
                                  "Co": latest["maxCo_latest"],
                                  "alphaCo": latest["maxAlphaCo_latest"]}), flush=True)
                last_bucket = bucket
                if accepted:
                    if last_accept_time is not None and t[-1] - last_accept_time >= 0.2:
                        reason = "qualified_X_Y_N_stationary_tail"
                        stop_at_write_now(reason)
                    else:
                        last_accept_time = float(t[-1])
                else:
                    last_accept_time = None
        latest_dt = last(r"deltaT = ([\d.eE+-]+)", log)
        latest_omega_max = last(r"bounding omega, min: [^\n]* max: ([\d.eE+-]+)", log)
        latest_k_max = last(r"bounding k, min: [^\n]* max: ([\d.eE+-]+)", log)
        if re.search(r"\bnan\b|\binf\b", log, re.I):
            reason = "nonfinite_solver_log"
            stop_at_write_now(reason)
        if (latest_dt is not None and latest_dt < 1e-7) or \
           (latest_omega_max is not None and latest_omega_max > 1e7) or \
           (latest_k_max is not None and latest_k_max > 1e4):
            reason = "catastrophic_numeric_growth"
            stop_at_write_now(reason)
        if wall >= MAX_WALL_S:
            reason = "maximum_wall_time"
            stop_at_write_now(reason)
        if EXIT.exists() or "\nEnd\n" in log or "FOAM FATAL" in log or reason:
            SUMMARY.write_text(json.dumps({"reason": reason or "solver_ended_or_failed",
                                           "last_snapshot": latest,
                                           "monitor_wall_s": wall}, indent=2) + "\n")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
