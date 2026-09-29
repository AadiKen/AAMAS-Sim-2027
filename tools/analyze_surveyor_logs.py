"""Read-only Surveyor log diagnostics; never fits hull coefficients."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def _read(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ("timestamp", "x_local", "y_local", "heading", "yaw_rate", "thrust",
                    "thrust_difference", "port_command", "starboard_command",
                    "inter_vessel_distance", "ax", "ay", "az"):
            row[key] = float(row[key]) if row[key] else math.nan
    return rows


def _correlation(rows: list[dict], lag_s: float) -> dict:
    times = np.array([r["timestamp"] for r in rows]); values = []
    for row in rows:
        if abs(row["thrust_difference"]) <= .1:
            continue
        index = int(np.argmin(np.abs(times-(row["timestamp"]+lag_s))))
        selected = rows[index]
        if abs(selected["timestamp"]-row["timestamp"]-lag_s) > .51 or not math.isfinite(selected["yaw_rate"]):
            continue
        values.append((row["thrust_difference"], selected["yaw_rate"]))
    if len(values) < 5:
        return {"sample_count": len(values), "correlation": None, "sign_agreement": None}
    array = np.asarray(values)
    return {"sample_count": len(values), "correlation": float(np.corrcoef(array.T)[0, 1]),
            "sign_agreement": float(np.mean(np.sign(array[:, 0]) == np.sign(array[:, 1])))}


def run(root: str | Path) -> dict:
    root = Path(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "plots" / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = root / "plots"; plots.mkdir(parents=True, exist_ok=True)
    data = {v: _read(root / "processed_logs" / f"{v}_normalized.csv") for v in ("USV1", "USV2")}
    results = {"heading_reference": "magnetic; declination and absolute yaw alignment unresolved",
               "yaw_rate_reference": "logged degrees/s converted to body-FRD positive yaw rad/s",
               "command_mapping": "port=T+D, starboard=T-D, clipped to [-1,1] after percent normalization",
               "vessels": {}}
    for vessel, rows in data.items():
        t0 = rows[0]["timestamp"]
        time = np.array([r["timestamp"]-t0 for r in rows])
        x = np.array([r["x_local"] for r in rows]); y = np.array([r["y_local"] for r in rows])
        heading = np.array([r["heading"] for r in rows]); yaw = np.array([r["yaw_rate"] for r in rows])
        thrust = np.array([r["thrust"] for r in rows]); diff = np.array([r["thrust_difference"] for r in rows])
        distance = np.array([r["inter_vessel_distance"] for r in rows])
        speed = np.full(len(rows), np.nan)
        for i in range(1, len(rows)):
            dt = time[i]-time[i-1]
            if 0 < dt <= 3 and all(math.isfinite(v) for v in (x[i], y[i], x[i-1], y[i-1])):
                speed[i] = math.hypot(x[i]-x[i-1], y[i]-y[i-1])/dt
        result = {"command_yaw_lag_checks": {str(lag): _correlation(rows, lag) for lag in (0, 1, 2, 3)},
                  "gps_speed_mps": {"median": float(np.nanmedian(speed)), "max": float(np.nanmax(speed))},
                  "distance_m": {"median": float(np.nanmedian(distance)), "max": float(np.nanmax(distance))},
                  "acceleration_z_mps2_median": float(np.nanmedian([r["az"] for r in rows]))}
        results["vessels"][vessel] = result
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(time, thrust, label="T"); axes[0].plot(time, diff, label="D")
        axes[0].plot(time, [r["port_command"] for r in rows], alpha=.5, label="port")
        axes[0].plot(time, [r["starboard_command"] for r in rows], alpha=.5, label="starboard")
        axes[0].set_ylabel("normalized command"); axes[0].legend(ncol=4, fontsize=7)
        axes[1].plot(time, yaw); axes[1].set_ylabel("yaw rate (rad/s)")
        axes[2].plot(time, np.degrees(heading)); axes[2].set_ylabel("magnetic heading (deg)")
        axes[3].plot(time, speed, label="GPS speed"); axes[3].plot(time, distance, label="separation")
        axes[3].set(xlabel="seconds from first row", ylabel="m or m/s"); axes[3].legend()
        fig.suptitle(f"{vessel}: recorded data; gaps not interpolated")
        fig.tight_layout();fig.savefig(plots / f"{vessel}_recorded_timeseries.png", dpi=130);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7))
    for vessel, rows in data.items():
        ax.plot([r["x_local"] for r in rows], [r["y_local"] for r in rows], marker=".", ms=2, label=vessel)
    ax.set(xlabel="east (m)", ylabel="north (m)", title="Measured GPS tracks; no simulation")
    ax.set_aspect("equal");ax.legend();ax.grid()
    fig.savefig(plots / "measured_trajectories.png", dpi=130);plt.close(fig)
    (root / "processed_logs" / "sensor_diagnostics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/analyze_surveyor_logs.py VALIDATION_OUTPUT_DIRECTORY")
    run(sys.argv[1])
