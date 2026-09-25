"""Analyze passive 60–80 s SST WAM-V probe and force records.

This script only reads an existing OpenFOAM case. Cross-correlations are
descriptive; they are not interpreted as causal evidence by themselves.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


CASE = Path("stage3_results/wamv-stage3-generalized/cfd/sst-baseline/"
            "ef3e2555a9e8824f36a7c5f3e5ead4145f29bd536ac036f46cfc69e1fd9e356e")
POINT = re.compile(r"^# Probe (\d+) \(([-+\d.eE]+) ([-+\d.eE]+) ([-+\d.eE]+)\)")


def probe_file(path: Path, *, vector: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    rows = []
    for line in path.read_text().splitlines():
        match = POINT.match(line)
        if match:
            positions.append(tuple(map(float, match.groups()[1:])))
        elif line and not line.startswith("#"):
            row = np.fromstring(line.replace("(", " ").replace(")", " "), sep=" ")
            if len(row):
                rows.append(row)
    points = np.asarray(positions)
    data = np.asarray(rows)
    expected = 1 + len(points) * (3 if vector else 1)
    if data.ndim != 2 or data.shape[1] != expected:
        raise ValueError(f"unexpected probe columns in {path}: {data.shape}, expected {expected}")
    values = data[:, 1:].reshape((len(data), len(points), 3)) if vector else data[:, 1:]
    return data[:, 0], points, values


def force_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            row = np.fromstring(line.replace("(", " ").replace(")", " "), sep=" ")
            if len(row):
                rows.append(row)
    data = np.asarray(rows)
    if data.ndim != 2 or data.shape[1] != 13:
        raise ValueError(f"unexpected force columns: {data.shape}")
    pressure = np.column_stack((data[:, 1:4], data[:, 7:10]))
    viscous = np.column_stack((data[:, 4:7], data[:, 10:13]))
    return data[:, 0], pressure, viscous


def interface_elevation(alpha: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal = points[:90].reshape((30, 3, 3))
    if not np.allclose(horizontal[:, 0, :2], horizontal[:, 2, :2]):
        raise ValueError("first 90 probes are not 30 vertical triplets")
    triplets = alpha[:, :90].reshape((len(alpha), 30, 3))
    slope = (triplets[:, :, 2] - triplets[:, :, 0]) / .25
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = (.5 - triplets[:, :, 1]) / slope
    eta[(np.abs(slope) < .25) | (np.abs(eta) > .25)] = np.nan
    return horizontal[:, 1, :2], eta


def detrend(values: np.ndarray) -> np.ndarray:
    grid = np.arange(len(values))
    good = np.isfinite(values)
    if good.sum() < 10:
        return np.full_like(values, np.nan)
    filled = np.interp(grid, grid[good], values[good])
    return filled - np.polyval(np.polyfit(grid, filled, 1), grid)


def lag_scan(left: np.ndarray, right: np.ndarray, dt: float, max_lag_s: float) -> dict:
    """Positive lag means a feature in right follows the feature in left."""
    left = detrend(left)
    right = detrend(right)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return {"valid": False}
    max_steps = min(int(max_lag_s / dt), len(left) // 3)
    results = []
    for lag in range(-max_steps, max_steps + 1):
        a, b = (left[:-lag], right[lag:]) if lag > 0 else (
            (left[-lag:], right[:lag]) if lag < 0 else (left, right))
        if len(a) > 10 and np.std(a) and np.std(b):
            results.append((lag, float(np.corrcoef(a, b)[0, 1])))
    if not results:
        return {"valid": False}
    # Propagation of the same scalar feature should preserve its sign. A
    # stronger negative half-cycle correlation is phase ambiguity, not travel.
    best = max(results, key=lambda pair: pair[1])
    best_absolute = max(results, key=lambda pair: abs(pair[1]))
    zero = next(value for lag, value in results if lag == 0)
    return {"valid": True, "best_lag_s": best[0] * dt,
            "best_correlation": best[1], "zero_lag_correlation": zero,
            "absolute_peak_lag_s": best_absolute[0] * dt,
            "absolute_peak_correlation": best_absolute[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.case
    probe_root = root / "postProcessing/diagnosticProbes/60"
    time, points, alpha = probe_file(probe_root / "alpha.water")
    measured = {"alpha.water": alpha}
    for field in ("p", "p_rgh", "k", "omega", "nut"):
        other_time, other_points, values = probe_file(probe_root / field)
        if not np.allclose(other_time, time) or not np.allclose(other_points, points):
            raise ValueError(f"{field} is not aligned with alpha.water")
        measured[field] = values
    other_time, other_points, velocity = probe_file(probe_root / "U", vector=True)
    if not np.allclose(other_time, time) or not np.allclose(other_points, points):
        raise ValueError("U is not aligned with alpha.water")
    measured["U"] = velocity
    horizontal, eta = interface_elevation(alpha, points)
    force_time, pressure, viscous = force_file(root / "postProcessing/forces/60/forces.dat")
    load = pressure + viscous
    cadence = np.diff(time)
    output = args.output or root.parent / "diagnostic_60_80"
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "interface_traces.csv", np.column_stack((time, eta)),
               delimiter=",", header="time_s," + ",".join(
                   f"eta_{i}_{x:g}_{y:g}_m" for i, (x, y) in enumerate(horizontal)),
               comments="")
    np.savetxt(output / "wrench_decomposition.csv",
               np.column_stack((force_time, pressure, viscous)),
               delimiter=",", header="time_s," + ",".join(
                   f"{part}_{channel}" for part in ("pressure", "viscous")
                   for channel in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")), comments="")
    interval = (time >= 60) & (time <= min(80, time[-1]))
    dt = float(np.median(cadence))
    routes = {
        "downstream": [6, 7, 8, 9, 10, 11, 12, 13, 14],
        "upstream": [5, 4, 3, 2, 1, 0],
        "port": [22, 23, 24, 25, 26, 27, 28, 29],
        "starboard": [21, 20, 19, 18, 17, 16, 15],
    }
    pair_lags = {}
    for route, indices in routes.items():
        pair_lags[route] = []
        for first, second in zip(indices, indices[1:]):
            pair_lags[route].append({
                "from_xy_m": horizontal[first].tolist(),
                "to_xy_m": horizontal[second].tolist(),
                "eta": lag_scan(eta[interval, first], eta[interval, second], dt, 3),
                "Ux": lag_scan(velocity[interval, first * 3 + 1, 0],
                               velocity[interval, second * 3 + 1, 0], dt, 3),
            })
    force_interval = (force_time >= 60) & (force_time <= min(80, force_time[-1]))
    force_summary = {}
    probe_load_lags = {}
    for channel, index in (("Fx", 0), ("Fz", 2), ("My", 4)):
        values = load[force_interval, index]
        force_summary[channel] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "first_5s_mean": float(np.mean(values[force_time[force_interval] < 65])),
            "last_5s_mean": float(np.mean(values[force_time[force_interval] >= max(60, force_time[-1] - 5)])),
            "slope_per_s": float(np.polyfit(force_time[force_interval], values, 1)[0]),
        }
        at_probe_times = np.interp(time[interval], force_time, load[:, index])
        probe_load_lags[channel] = {
            name: lag_scan(signal[interval], at_probe_times, dt, 2)
            for name, signal in (
                ("upstream_hull_eta", eta[:, 5]),
                ("downstream_hull_eta", eta[:, 6]),
                ("outlet_eta", eta[:, 14]),
                ("port_boundary_eta", eta[:, 29]),
                ("bow_pressure", measured["p"][:, 90]),
                ("stern_pressure", measured["p"][:, 92]),
                ("near_wake_k", measured["k"][:, 98]),
            )
        }
    summary = {
        "probe_count": len(points), "probe_samples": len(time),
        "probe_start_s": float(time[0]), "probe_end_s": float(time[-1]),
        "probe_max_cadence_s": float(np.max(cadence)),
        "force_end_s": float(force_time[-1]),
        "interface_valid_fraction": float(np.mean(np.isfinite(eta))),
        "routes": pair_lags, "force_summary": force_summary,
        "probe_to_load_lags": probe_load_lags,
        "pressure_probe_indices": list(range(90, 98)),
        "turbulence_probe_indices": list(range(98, 108)),
        "note": "Lag signs are descriptive; propagation requires coherent sequential lags across adjacent probes and fields.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in (
        "probe_count", "probe_samples", "probe_start_s", "probe_end_s",
        "probe_max_cadence_s", "force_end_s", "interface_valid_fraction")}, indent=2))


if __name__ == "__main__":
    main()
