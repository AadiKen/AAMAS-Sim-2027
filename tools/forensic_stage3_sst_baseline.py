"""Post-process existing WAM-V SST surge checkpoints; never run the solver.

The far-field surface estimate is the water fraction in the two 0.5 m
background cells straddling z=0. Near-hull estimates are relative changes
of the refined-column water fraction, not a reconstructed interface mesh.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter


CASE = Path("stage3_results/wamv-stage3-generalized/cfd/sst-baseline/"
            "ef3e2555a9e8824f36a7c5f3e5ead4145f29bd536ac036f46cfc69e1fd9e356e")
SCALE_FORCE = 1431.0
SCALE_MOMENT = 7057.0
NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def internal(path: Path, *, vector: bool = False) -> np.ndarray:
    source = path.read_text()
    kind = "vector" if vector else "scalar"
    match = re.search(
        rf"internalField\s+nonuniform List<{kind}>\s*(\d+)\s*\(\n(.*?)\n\)\s*;",
        source, re.S,
    )
    if match is None:
        raise ValueError(f"missing nonuniform internal field: {path}")
    values = np.fromstring(match.group(2).replace("(", " ").replace(")", " "), sep=" ")
    count = int(match.group(1))
    if len(values) != count * (3 if vector else 1):
        raise ValueError(f"incomplete internal field: {path}")
    return values.reshape((-1, 3)) if vector else values


def interpolated_spectrum(time: np.ndarray, values: np.ndarray, *, start: float,
                          end: float, spacing: float = .05) -> dict:
    grid = np.arange(start, end + spacing / 2, spacing)
    signal = np.interp(grid, time, values)
    trend = np.polyval(np.polyfit(grid, signal, 1), grid)
    detrended = signal - trend
    taper = np.hanning(len(grid))
    frequencies = np.fft.rfftfreq(len(grid), spacing)
    power = np.abs(np.fft.rfft(detrended * taper)) ** 2
    peaks = np.flatnonzero((power[1:-1] > power[:-2]) &
                           (power[1:-1] >= power[2:])) + 1
    peaks = [int(i) for i in peaks if .025 <= frequencies[i] <= 1.0]
    peaks.sort(key=lambda i: power[i], reverse=True)
    autocorrelation = np.correlate(detrended[::5], detrended[::5], mode="full")
    autocorrelation = autocorrelation[len(autocorrelation) // 2:]
    autocorrelation /= autocorrelation[0]
    lags = np.arange(len(autocorrelation)) * spacing * 5
    eligible = np.flatnonzero((lags >= 3) & (lags <= min(30, (end - start) / 2)))
    auto_peaks = [int(i) for i in eligible[1:-1]
                  if autocorrelation[i] > 0 and
                  autocorrelation[i] > autocorrelation[i - 1] and
                  autocorrelation[i] >= autocorrelation[i + 1]]
    auto_peaks.sort(key=lambda i: autocorrelation[i], reverse=True)
    return {
        "slope_per_s": float(np.polyfit(grid, signal, 1)[0]),
        "spectral_peaks_hz": [float(frequencies[i]) for i in peaks[:5]],
        "spectral_periods_s": [float(1 / frequencies[i]) for i in peaks[:5]],
        "autocorrelation_peak_lags_s": [float(lags[i]) for i in auto_peaks[:5]],
        "autocorrelation_peak_values": [float(autocorrelation[i]) for i in auto_peaks[:5]],
    }


def lag_correlation(time: np.ndarray, left: np.ndarray, right: np.ndarray,
                    *, start: float, end: float, max_lag_s: float = 15) -> dict:
    spacing = .1
    grid = np.arange(start, end + spacing / 2, spacing)
    a = np.interp(grid, time, left)
    b = np.interp(grid, time, right)
    a -= np.polyval(np.polyfit(grid, a, 1), grid)
    b -= np.polyval(np.polyfit(grid, b, 1), grid)
    lags = np.arange(-int(max_lag_s / spacing), int(max_lag_s / spacing) + 1)
    scores = []
    for lag in lags:
        aa, bb = (a[:-lag], b[lag:]) if lag > 0 else (
            (a[-lag:], b[:lag]) if lag < 0 else (a, b))
        scores.append(float(np.corrcoef(aa, bb)[0, 1]) if len(aa) > 5 else np.nan)
    best = int(np.nanargmax(np.abs(scores)))
    return {"lag_s_right_after_left": float(lags[best] * spacing),
            "correlation": scores[best],
            "zero_lag_correlation": scores[len(lags) // 2]}


def smoothed_extrema(time: np.ndarray, values: np.ndarray, *, start: float,
                     end: float, width_s: float = 3, separation_s: float = 4) -> dict:
    spacing = .01
    grid = np.arange(start, end + spacing / 2, spacing)
    raw = np.interp(grid, time, values)
    width = int(round(width_s / spacing))
    smooth = np.convolve(np.pad(raw, width // 2, mode="edge"),
                         np.ones(width) / width, mode="valid")[:len(grid)]
    neighborhood = int(round(separation_s / spacing))
    extrema = {}
    for kind, comparator in (("crests", max), ("troughs", min)):
        candidates = np.flatnonzero((smooth[1:-1] > smooth[:-2]) &
                                    (smooth[1:-1] >= smooth[2:])) + 1 if kind == "crests" else \
            np.flatnonzero((smooth[1:-1] < smooth[:-2]) &
                           (smooth[1:-1] <= smooth[2:])) + 1
        selected = []
        for index in candidates:
            if grid[index] < start + width_s or grid[index] > end - width_s:
                continue
            interval = smooth[max(0, index - neighborhood):
                              min(len(smooth), index + neighborhood + 1)]
            if smooth[index] != comparator(interval):
                continue
            if selected and index - selected[-1] < neighborhood:
                continue
            selected.append(index)
        extrema[kind] = [{"time_s": float(grid[i]), "smoothed_load": float(smooth[i])}
                         for i in selected]
    return extrema


def cycle_statistics(time: np.ndarray, values: np.ndarray,
                     crests: list[dict]) -> list[dict]:
    cycles = []
    for first, second in zip(crests[:-1], crests[1:]):
        start, end = first["time_s"], second["time_s"]
        part = (time >= start) & (time <= end)
        t, y = time[part], values[part]
        if len(t) < 2:
            continue
        mean = float(np.trapezoid(y, t) / (t[-1] - t[0]))
        cycles.append({"start_s": float(t[0]), "end_s": float(t[-1]),
                       "period_s": float(t[-1] - t[0]), "mean": mean,
                       "peak_to_peak": float(np.ptp(y)),
                       "rms": float(np.sqrt(np.trapezoid(y * y, t) / (t[-1] - t[0]))),
                       "fluctuation_rms": float(np.sqrt(
                           np.trapezoid((y - mean) ** 2, t) / (t[-1] - t[0])))} )
    return cycles


def force_decomposition(root: Path) -> np.ndarray:
    pattern = r"[-+0-9.eE]+"
    files = sorted(root.glob("postProcessing/forces/*/forces.dat"),
                   key=lambda path: float(path.parent.name))
    samples: dict[float, list[float]] = {}
    for index, path in enumerate(files):
        stop = float(files[index + 1].parent.name) if index + 1 < len(files) else np.inf
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            vectors = re.findall(rf"\(({pattern})\s+({pattern})\s+({pattern})\)", line)
            if len(vectors) != 4:
                continue
            instant = float(line.split()[0])
            if instant >= stop:
                continue
            values = np.asarray(vectors, dtype=float).reshape(4, 3)
            samples[instant] = [instant, *values.reshape(-1)]
    return np.asarray([samples[key] for key in sorted(samples)])


def probe_indices(centers: np.ndarray, xyz: tuple[float, float, float]) -> tuple[int, float]:
    target = np.asarray(xyz)
    distances = np.sum((centers - target) ** 2, axis=1)
    index = int(np.argmin(distances))
    return index, float(np.sqrt(distances[index]))


def field_traces(root: Path) -> tuple[list[dict], dict]:
    centers = internal(root / "14/C", vector=True)
    times = sorted(int(path.name) for path in root.iterdir()
                   if path.is_dir() and path.name.isdigit() and int(path.name) > 0 and
                   (path / "alpha.water").exists() and (path / "U").exists())
    far_xy = {
        "upstream10": (-10.25, .25), "upstream5": (-5.25, .25),
        "downstream3": (3.75, .25), "downstream5": (5.25, .25),
        "downstream10": (10.25, .25), "downstream15": (15.25, .25),
        "downstream20": (20.25, .25), "outlet": (21.75, .25),
        "port_side": (.25, 9.75), "starboard_side": (.25, -9.75),
    }
    alpha_index = {}
    for name, (x, y) in far_xy.items():
        lower, dl = probe_indices(centers, (x, y, -.25))
        upper, du = probe_indices(centers, (x, y, .25))
        if max(dl, du) > .08:
            raise ValueError(f"far-field probe {name} is not on background cells")
        alpha_index[name] = (lower, upper)
    velocity_index = {}
    for x in (3.75, 5.25, 10.25, 15.25, 20.25):
        for y in np.arange(-5.25, 5.26, .5):
            index, distance = probe_indices(centers, (x, float(y), -.25))
            if distance > .08:
                raise ValueError(f"wake probe {(x, y)} is not on background cells")
            velocity_index[(x, float(y))] = index
    refined = {}
    for name, (x, y) in {"near_center": (.25, .25),
                         "near_outer_hull": (.25, 1.75)}.items():
        mask = (np.hypot(centers[:, 0] - x, centers[:, 1] - y) < .1) & \
            (np.abs(centers[:, 2]) < .25)
        refined[name] = np.flatnonzero(mask)
    result = []
    for time in times:
        directory = root / str(time)
        alpha = internal(directory / "alpha.water")
        velocity = internal(directory / "U", vector=True)
        k, omega, nut = (internal(directory / field)
                         for field in ("k", "omega", "nut"))
        row: dict[str, float | int] = {"time_s": time}
        for name, (lower, upper) in alpha_index.items():
            row[f"eta_{name}_m"] = float(.5 * (alpha[lower] + alpha[upper] - 1))
        for name, indexes in refined.items():
            # Refined near-hull columns are about 0.125 m high. Compare
            # changes over time only; their absolute surface datum is rough.
            row[f"relative_eta_{name}_m"] = float(.125 * alpha[indexes].sum())
        for x in (3.75, 5.25, 10.25, 15.25, 20.25):
            left = velocity[velocity_index[(x, -1.25)], 0]
            right = velocity[velocity_index[(x, 1.25)], 0]
            cross = np.asarray([velocity[velocity_index[(x, float(y))], 0]
                                for y in np.arange(-5.25, 5.26, .5)])
            row[f"wake_ux_{x:g}_mps"] = float((left + right) / 2)
            row[f"wake_port_starboard_difference_{x:g}_mps"] = float(right - left)
            row[f"wake_deficit_width_{x:g}_m"] = float(.5 * np.sum(cross < .95))
            row[f"wake_deficit_max_{x:g}_mps"] = float(np.maximum(0, 1 - cross.min()))
            row[f"wake_k_{x:g}_m2ps2"] = float(np.mean([
                k[velocity_index[(x, y)]] for y in (-1.25, 1.25)]))
            row[f"wake_omega_{x:g}_ps"] = float(np.mean([
                omega[velocity_index[(x, y)]] for y in (-1.25, 1.25)]))
            row[f"wake_nut_{x:g}_m2ps"] = float(np.mean([
                nut[velocity_index[(x, y)]] for y in (-1.25, 1.25)]))
            row[f"wake_shear_proxy_{x:g}_ps"] = float(np.max(np.abs(np.diff(cross) / .5)))
        result.append(row)
    return result, {"times_s": times, "far_field_probes_xy_m": far_xy,
                    "near_hull_probe_xy_m": {"near_center": [.25, .25],
                                             "near_outer_hull": [.25, 1.75]},
                    "far_field_eta_method": "alpha in 0.5 m cells immediately below and above z=0",
                    "near_hull_eta_method": "relative refined-column alpha sum; not absolute elevation"}


def dispersion(frequency_hz: float, *, depth_m: float = 5.,
               gravity: float = 9.80665) -> dict:
    omega = 2 * np.pi * frequency_hz
    low, high = 0., max(omega * omega / gravity * 2, omega / np.sqrt(gravity * depth_m) * 2)
    for _ in range(100):
        middle = (low + high) / 2
        if gravity * middle * np.tanh(middle * depth_m) < omega * omega:
            low = middle
        else:
            high = middle
    k = (low + high) / 2
    kh = k * depth_m
    n = .5 * (1 + 2 * kh / np.sinh(2 * kh))
    return {"frequency_hz": frequency_hz, "k_per_m": float(k),
            "wavelength_m": float(2 * np.pi / k),
            "phase_speed_mps": float(omega / k),
            "group_speed_mps": float(n * omega / k)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.case
    output = args.output or root.parent / "forensic_60s"
    output.mkdir(parents=True, exist_ok=True)
    history = np.asarray(OpenFOAMAdapter.force_history(root))
    time, loads = history[:, 0], history[:, 1:]
    if time[-1] != 60:
        raise ValueError("forensic analysis requires the complete 60 s force history")
    spectra = {name: interpolated_spectrum(time, loads[:, index], start=10,
                                           end=60) for index, name in enumerate(NAMES)}
    extrema = {name: smoothed_extrema(time, loads[:, index], start=8, end=60)
               for index, name in enumerate(NAMES)}
    cycles = {name: cycle_statistics(time, loads[:, index], extrema[name]["crests"])
              for index, name in enumerate(NAMES)}
    correlations = {f"{left}_{right}": lag_correlation(
        time, loads[:, NAMES.index(left)], loads[:, NAMES.index(right)], start=10, end=60)
        for left, right in (("Fx", "Fz"), ("Fx", "My"), ("Fz", "My"))}
    decomposition = force_decomposition(root)
    fields, method = field_traces(root)
    with (output / "field_traces.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields[0]))
        writer.writeheader()
        writer.writerows(fields)
    geometry = {"hull_x_min_m": -2.37641, "hull_x_max_m": 2.555073,
                "hull_y_min_m": -1.2436289, "hull_y_max_m": 1.2436291,
                "domain_x_min_m": -12., "domain_x_max_m": 22.,
                "domain_y_min_m": -10., "domain_y_max_m": 10.,
                "depth_m": 5., "inflow_speed_mps": 1.}
    distances = {"hull_length_m": geometry["hull_x_max_m"] - geometry["hull_x_min_m"],
                 "hull_to_outlet_m": 22 - geometry["hull_x_max_m"],
                 "inlet_to_hull_m": geometry["hull_x_min_m"] + 12,
                 "hull_to_side_m": 10 - geometry["hull_y_max_m"]}
    travel = {"convective_s": {"hull_length": distances["hull_length_m"],
                               "hull_to_outlet": distances["hull_to_outlet_m"],
                               "inlet_to_hull": distances["inlet_to_hull_m"],
                               "hull_to_side": None},
              "wave_modes": []}
    for frequency in (.05, .075, .1, .2, .3, .4, .5):
        mode = dispersion(frequency)
        group = mode["group_speed_mps"]
        mode["one_way_to_outlet_s"] = distances["hull_to_outlet_m"] / group
        mode["outlet_round_trip_s"] = 2 * mode["one_way_to_outlet_s"]
        mode["one_way_to_side_s"] = distances["hull_to_side_m"] / group
        mode["side_round_trip_s"] = 2 * mode["one_way_to_side_s"]
        mode["one_way_to_inlet_s"] = distances["inlet_to_hull_m"] / group
        mode["inlet_round_trip_s"] = 2 * mode["one_way_to_inlet_s"]
        travel["wave_modes"].append(mode)
    summary = {"force_samples": len(history), "force_time_range_s": [float(time[0]),float(time[-1])],
               "spectra": spectra, "extrema": extrema, "cycles": cycles,
               "load_cross_correlations": correlations,
               "geometry": geometry, "distances": distances, "travel_times": travel,
               "field_method": method,
               "force_pressure_viscous_columns": ["time_s", "Fx_pressure", "Fy_pressure",
                   "Fz_pressure", "Fx_viscous", "Fy_viscous", "Fz_viscous",
                   "Mx_pressure", "My_pressure", "Mz_pressure", "Mx_viscous",
                   "My_viscous", "Mz_viscous"],
               "force_decomposition_rows": len(decomposition)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(output / "pressure_viscous_forces.csv", decomposition,
               delimiter=",", header=",".join(summary["force_pressure_viscous_columns"]),
               comments="")
    print(json.dumps({"output": str(output), "field_times": len(fields),
                      "force_samples": len(history), "cycle_counts": {
                          name: len(cycles[name]) for name in ("Fx", "Fz", "My")}}, indent=2))


if __name__ == "__main__":
    main()
