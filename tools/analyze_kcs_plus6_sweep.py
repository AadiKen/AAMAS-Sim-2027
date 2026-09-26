"""Compare the bounded +6 degree diagnostic branches on one matched window."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import detrend, periodogram

from bcod_sim.vessel_generation.hmri_wrench import separate_openfoam_wrench

ROOT = Path("stage3_results/kcs-validation/track_b_hmri/plus6_diagnostic_sweep")
BRANCHES = ("A_control", "B_tight_cfl", "C_laminar", "D_all_water")
FLOATS = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
START = 16.8  # matched common post-restart interval requested for early classification
END = 18.2


def forces(case: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for file in (case / "postProcessing/forces").glob("*/forces.dat"):
        try:
            if float(file.parent.name) < 16.65:
                continue
        except ValueError:
            continue
        for line in file.read_text().splitlines():
            if line.startswith("#"):
                continue
            values = [float(v) for v in FLOATS.findall(line)]
            if len(values) == 13 and START <= values[0] <= END:
                rows.append(values)
    by_time = {row[0]: row for row in rows}
    raw = np.asarray([by_time[t] for t in sorted(by_time)])
    if not len(raw):
        raise ValueError(f"No diagnostic forces in {case}")
    total, pressure, viscous = [], [], []
    for row in raw:
        for target, force, moment in (
            (total, row[1:4]+row[4:7], row[7:10]+row[10:13]),
            (pressure, row[1:4], row[7:10]),
            (viscous, row[4:7], row[10:13]),
        ):
            physical, _ = separate_openfoam_wrench(
                force, moment, foam_reference=(-.0851, 0, 0),
                body_reference=(-.0851, 0, 0), waterline_z_m=0,
            )
            target.append([*physical.force_frd_n, *physical.moment_frd_nm])
    return raw[:, 0], np.asarray(total), np.asarray(pressure), np.asarray(viscous)


def dominant_frequency(t: np.ndarray, x: np.ndarray) -> dict:
    grid = np.arange(START, min(END, t[-1]), .01)
    if len(grid) < 100:
        return {"hz": None, "period_s": None}
    y = detrend(np.interp(grid, t, x))
    f, power = periodogram(y, fs=100, nfft=16384)
    select = (f >= .15) & (f <= 1.2)
    hz = float(f[select][np.argmax(power[select])])
    observed_periods = (grid[-1] - grid[0]) * hz
    return {"hz": hz, "period_s": 1/hz,
            "observed_periods": float(observed_periods),
            "resolved": bool(observed_periods >= 1.5)}


def main() -> None:
    traces = {name: forces(ROOT / name) for name in BRANCHES}
    results = []
    for name in BRANCHES:
        # The all-water conversion caused an initial hydrostatic adjustment;
        # compare its late interval with the control at the same times.
        lo, hi = (17.5, 18.19) if name == "D_all_water" else (16.8, 17.85)
        grid = np.arange(lo, hi, .002)
        ta, wa, _, _ = traces["A_control"]
        tb, wb, pb, vb = traces[name]
        if grid[-1] > min(ta[-1], tb[-1]):
            raise ValueError(f"Incomplete matched window for {name}")
        row = {"branch": name, "window_start_s": lo,
               "window_end_s": float(grid[-1]), "samples": len(grid)}
        for channel, col in (("Y", 1), ("N", 5)):
            a = np.interp(grid, ta, wa[:, col])
            x = np.interp(grid, tb, wb[:, col])
            pressure = np.interp(grid, tb, pb[:, col])
            viscous = np.interp(grid, tb, vb[:, col])
            row[channel] = {
                "mean": float(x.mean()), "rms_about_mean": float(x.std()),
                "peak_to_peak": float(np.ptp(x)),
                "amplitude_ratio_to_A": float(x.std()/a.std()),
                "peak_to_peak_ratio_to_A": float(np.ptp(x)/np.ptp(a)),
                "trace_correlation_to_A": float(np.corrcoef(a, x)[0, 1]),
                "pressure_rms": float(pressure.std()),
                "viscous_rms": float(viscous.std()),
                "dominant_frequency_resolved": False,
            }
        results.append(row)
    output = ROOT / "sweep_metrics.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)
    for row in results:
        print(row["branch"], row["window_start_s"], row["window_end_s"],
              "Y", row["Y"], "N", row["N"])


if __name__ == "__main__":
    main()
