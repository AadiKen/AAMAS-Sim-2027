"""Steady-iteration force qualification and one-frame FRD conversion."""
from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np

from bcod_sim.vessel_generation.frame_contract import foam_wrench_to_body


_NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"


def parse_force_history(path: Path, *, force_scale_to_newtons: float = 1.,
                        before_iteration: float | None = None) -> np.ndarray:
    """Return iteration/time and physical OpenFOAM pressure+viscous wrench."""
    if force_scale_to_newtons <= 0:
        raise ValueError("Positive declared force scale required")
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        iteration = float(line.split()[0])
        if before_iteration is not None and iteration >= before_iteration:
            continue
        vectors = re.findall(rf"\(({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\)", line)
        if len(vectors) != 4:
            raise ValueError("Incomplete OpenFOAM force row")
        z = np.asarray(vectors, float)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite force history")
        rows.append([iteration, *(force_scale_to_newtons*(z[0]+z[1])),
                     *(force_scale_to_newtons*(z[2]+z[3]))])
    if not rows and before_iteration is None:
        raise ValueError("Empty force history")
    return np.asarray(rows).reshape(-1, 7)


def qualify_force_tail(history: np.ndarray, *, residual_log: str,
                       reference_y_n: tuple[float, float] | None = None) -> dict:
    """Apply the document's conservative last-500/1000 iteration checks."""
    if history.ndim != 2 or history.shape[1] != 7 or not np.isfinite(history).all():
        raise ValueError("Finite 7-column force history required")
    if len(history) < 3000 or history[-1, 0] < 3000:
        return {"status": "failed", "reason": "fewer_than_3000_iterations"}
    if np.any(np.diff(history[:, 0]) <= 0):
        raise ValueError("Force history must contain unique increasing iterations")
    if re.search(r"\b(?:nan|inf)\b", residual_log, re.I):
        return {"status": "failed", "reason": "nonfinite_solver_log"}
    last500 = history[-500:, :]
    prev500 = history[-1000:-500, :]
    last1000 = history[-1000:, :]
    means = last500[:, 1:].mean(axis=0)
    previous = prev500[:, 1:].mean(axis=0)
    band = np.ptp(last1000[:, [2, 6]], axis=0)
    refs = np.abs(means[[1, 5]])
    if reference_y_n is not None:
        benchmark = np.abs(reference_y_n)
        refs = np.where(refs < .01*benchmark, benchmark, refs)
    denominator = np.maximum(refs, 1e-12)
    change = np.abs(means[[1, 5]]-previous[[1, 5]])/denominator
    fraction = band/denominator
    residual_values = np.asarray([float(value) for value in re.findall(
        rf"Initial residual = ({_NUMBER})", residual_log)], float)
    residual_status = "unavailable"
    if len(residual_values) >= 1000:
        early = float(np.median(residual_values[:100]))
        previous_residual = float(np.median(residual_values[-1000:-500]))
        late = float(np.median(residual_values[-500:]))
        residual_status = ("three_orders" if late <= early*1e-3 else "flat"
                           if abs(late-previous_residual) <= .05*max(previous_residual, 1e-14)
                           else "growing" if late > previous_residual else "decreasing")
    if np.all(fraction <= .01) and np.all(change < .01) and residual_status in ("three_orders", "flat"):
        return {"status": "converged", "window": 500, "mean_foam": means.tolist(),
                "change_fraction": change.tolist(), "band_fraction": fraction.tolist(),
                "residual_status": residual_status}
    if (np.all(fraction <= .10) and np.any(fraction >= .01) and
        residual_status in ("three_orders", "flat")):
        return {"status": "oscillatory", "window": 1000,
                "mean_foam": last1000[:, 1:].mean(axis=0).tolist(),
                "change_fraction": change.tolist(), "band_fraction": fraction.tolist(),
                "residual_status": residual_status}
    return {"status": "failed", "reason": "unstable_force_tail",
            "change_fraction": change.tolist(), "band_fraction": fraction.tolist(),
            "residual_status": residual_status}


def case_force_history(case: Path) -> np.ndarray:
    """Stitch restarted force files by iteration; latest restart owns overlaps."""
    samples = {}
    paths = sorted(Path(case).glob("postProcessing/forces/*/forces.dat"),
                   key=lambda path: float(path.parent.name))
    for index, path in enumerate(paths):
        start = float(path.parent.name)
        samples = {t: row for t, row in samples.items() if t < start}
        cutoff = float(paths[index+1].parent.name) if index+1 < len(paths) else None
        for row in parse_force_history(path, before_iteration=cutoff):
            samples[float(row[0])] = row
    if not samples:
        raise ValueError("No force history")
    return np.asarray([samples[t] for t in sorted(samples)])


def physical_frd_row(*, case_id: str, u: float, v: float, r: float,
                     qualified: dict, foam_reference: tuple[float, float, float],
                     body_reference: tuple[float, float, float], waterline_z_m: float) -> dict:
    mean = qualified.get("mean_foam")
    row = {"case_id": case_id, "u_mps": u, "v_mps": v, "r_rad_s": r,
           "convergence_status": qualified["status"]}
    if mean is None:
        return row
    force, moment = foam_wrench_to_body(mean[:3], mean[3:], foam_reference=foam_reference,
                                        body_reference=body_reference,
                                        waterline_z_m=waterline_z_m, resisting=False)
    row.update(X_n=force[0], Y_n=force[1], N_nm=moment[2])
    return row


def write_cases_csv(path: Path, rows: list[dict]) -> None:
    fields = ("case_id", "u_mps", "v_mps", "r_rad_s", "X_n", "Y_n", "N_nm", "convergence_status", "mesh_profile", "cell_count", "wall_seconds", "core_count", "core_hours", "iterations")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
