"""Post-process the existing 0–80 s WAM-V SST load and numerical histories.

This script never invokes OpenFOAM. It uses the production stationarity and
numerical-history gates with their unchanged thresholds and historical scales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.quality import numerical_history_gate, wrench_stationarity


CASE = Path("stage3_results/wamv-stage3-generalized/cfd/sst-baseline/"
            "ef3e2555a9e8824f36a7c5f3e5ead4145f29bd536ac036f46cfc69e1fd9e356e")
FORCE_SCALE_N = 1431.0
MOMENT_SCALE_NM = 7057.0
CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
IMPORTANT = (0, 2, 4)
WINDOWS_S = (8., 10., 15., 20.)


def summarize_window(history: np.ndarray, end_s: float, duration_s: float) -> dict:
    segment = history[:np.searchsorted(history[:, 0], end_s + 1e-8, side="right")]
    if not len(segment) or segment[-1, 0] < end_s - .001:
        raise ValueError(f"incomplete force history through {end_s:g} s")
    gate = wrench_stationarity(segment[:, 0], segment[:, 1:],
        force_scale_n=FORCE_SCALE_N, moment_scale_nm=MOMENT_SCALE_NM,
        window_s=duration_s)
    selected = segment[segment[:, 0] >= end_s - duration_s]
    channels = {}
    for axis, name in enumerate(CHANNELS):
        data = gate.channels[axis]
        y = selected[:, axis + 1]
        # Descriptive detrended peak; not a separate stationarity criterion.
        grid = np.linspace(selected[0, 0], selected[-1, 0], 4097)
        uniform = np.interp(grid, selected[:, 0], y)
        detrended = uniform - np.polyval(np.polyfit(grid, uniform, 1), grid)
        spectrum = np.abs(np.fft.rfft(detrended * np.hanning(len(grid))))
        frequencies = np.fft.rfftfreq(len(grid), grid[1] - grid[0])
        usable = (frequencies >= 1 / duration_s) & (frequencies <= 2.)
        peak = int(np.flatnonzero(usable)[np.argmax(spectrum[usable])])
        amplitude = float(2 * spectrum[peak] / np.hanning(len(grid)).sum())
        scale = FORCE_SCALE_N if axis < 3 else MOMENT_SCALE_NM
        channels[name] = {
            "mean": data.mean,
            "rms": float(np.sqrt(np.mean(y * y))),
            "std": data.standard_deviation,
            "minimum": data.minimum,
            "maximum": data.maximum,
            "peak_to_peak": float(np.ptp(y)),
            "slope_per_s": data.slope_per_s,
            "first_half_mean": data.first_half_mean,
            "second_half_mean": data.second_half_mean,
            "first_second_mean_change": data.second_half_mean - data.first_half_mean,
            "normalized_mean_change": data.normalized_mean_change,
            "normalized_slope": data.normalized_slope,
            "normalized_std": data.normalized_standard_deviation,
            "physical_scale": scale,
            "absolute_mean_percent_of_scale": abs(data.mean) / scale * 100,
            "absolute_peak_percent_of_scale": max(abs(data.minimum), abs(data.maximum)) / scale * 100,
            "dominant_detrended_frequency_hz": float(frequencies[peak]),
            "dominant_detrended_period_s": float(1 / frequencies[peak]),
            "dominant_detrended_amplitude": amplitude,
            "regime": data.regime,
            "accepted": data.accepted,
            "periodic": vars(data.periodic) if data.periodic else None,
        }
    return {"start_s": float(end_s - duration_s), "end_s": float(end_s),
            "duration_s": float(duration_s),
            "important_channels_accepted": bool(all(gate.channels[i].accepted for i in IMPORTANT)),
            "all_six_gate_accepted": gate.accepted, "channels": channels}


def rolling_tail(history: np.ndarray, *, start_end_s: float = 28., end_s: float = 80.,
                 endpoint_step_s: float = 1.) -> list[dict]:
    records = []
    for endpoint in np.arange(start_end_s, end_s + endpoint_step_s / 2, endpoint_step_s):
        for duration in WINDOWS_S:
            if endpoint - duration < 0:
                continue
            result = summarize_window(history, float(endpoint), duration)
            records.append({"end_s": float(endpoint), "duration_s": duration,
                "start_s": result["start_s"],
                "important_channels_accepted": result["important_channels_accepted"],
                "channel_accepted": {name: result["channels"][name]["accepted"]
                                     for name in ("Fx", "Fz", "My")},
                "normalized_slope": {name: result["channels"][name]["normalized_slope"]
                                     for name in ("Fx", "Fz", "My")}})
    return records


def earliest_continuous_tail(records: list[dict], end_s: float = 80.,
                             candidate_start_s: float = 0.,
                             resolution_s: float = 1.) -> dict:
    # A candidate tail must contain at least one full 20 s window. Any
    # rolling 8/10/15/20 s window wholly inside the candidate tail must pass.
    candidates = np.arange(candidate_start_s, end_s - max(WINDOWS_S) + .001,
                           resolution_s)
    accepted = []
    for start in candidates:
        contained = [r for r in records if r["start_s"] >= start - 1e-9 and
                     r["end_s"] <= end_s + 1e-9]
        if contained and all(r["important_channels_accepted"] for r in contained):
            accepted.append(float(start))
    if not accepted:
        return {"start_s": None, "duration_s": 0., "later_refailure": None,
                "resolution_s": resolution_s}
    start = accepted[0]
    return {"start_s": start, "duration_s": end_s - start,
            "later_refailure": False, "resolution_s": resolution_s,
            "tested_windows": len([r for r in records if r["start_s"] >= start - 1e-9])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = args.case
    history = np.asarray(OpenFOAMAdapter.force_history(case), float)
    if history.shape[1] != 7 or not np.all(np.diff(history[:, 0]) > 0):
        raise ValueError("stitched six-axis force history is invalid")
    if abs(history[-1, 0] - 80.) > 1e-8:
        raise ValueError("80 s checkpoint is missing")
    final_windows = {f"{int(duration)}s": summarize_window(history, 80., duration)
                     for duration in WINDOWS_S}
    rolling = rolling_tail(history)
    tail = earliest_continuous_tail(rolling)
    # Refine the identified transition without rescanning the entire startup.
    refined_start = max(0., tail["start_s"] - 2.) if tail["start_s"] is not None else 0.
    refined = rolling_tail(history, start_end_s=refined_start + min(WINDOWS_S),
                           endpoint_step_s=.25)
    refined_tail = earliest_continuous_tail(refined, candidate_start_s=refined_start,
                                            resolution_s=.25)
    diagnostics = OpenFOAMAdapter.diagnose_case(case)
    numerical = numerical_history_gate(diagnostics, expected_start_s=0., expected_end_s=80.,
        max_courant=.5, max_alpha_courant=.5, max_timestep_s=.0006)
    output = args.output or case.parent / "stationary_tail_80s"
    output.mkdir(parents=True, exist_ok=True)
    result = {"force_samples": len(history), "force_end_s": float(history[-1, 0]),
              "force_scale_n": FORCE_SCALE_N, "moment_scale_nm": MOMENT_SCALE_NM,
              "unchanged_limits": {"significant_mean_change": .03,
                  "significant_normalized_slope": .03,
                  "significant_normalized_std": .10},
              "final_windows": final_windows, "rolling_windows": rolling,
              "refined_rolling_windows": refined,
              "continuous_tail": refined_tail, "coarse_tail": tail,
              "numerical_gate": numerical,
              "method": "Production wrench_stationarity and numerical_history_gate; "
                        "1 s rolling endpoints; all 8/10/15/20 s windows wholly inside "
                        "a candidate tail must pass on Fx/Fz/My; minimum tail 20 s."}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"force_samples": len(history),
                      "final_important_pass": {key: value["important_channels_accepted"]
                          for key, value in final_windows.items()},
                      "continuous_tail": refined_tail,
                      "numerical_gate_pass": numerical["accepted"]}, indent=2))


if __name__ == "__main__":
    main()
