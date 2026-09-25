"""Fail-closed Stage 3A conversion of solved cases into fitting observations."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np

from .cfd import OpenFOAMAdapter, CFDExecutionError
from .frame_contract import CONVENTION, foam_wrench_to_body
from .quality import numerical_history_gate, wrench_stationarity

SCHEMA = "bcod-qualified-observation-v1"
PARSER = "stage3a-observation-1"
AXES = {"surge": 0, "sway": 1, "heave": 2, "roll": 3, "pitch": 4, "yaw": 5}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(root: Path) -> dict[str, str]:
    paths = [root / "case_metadata.json", root / "constant/triSurface/hull.stl",
             root / "constant/polyMesh/points", root / "system/controlDict",
             root / "solver.log", root / "checkMesh.log", root / "openfoam_runtime.json"]
    paths += sorted(root.glob("postProcessing/forces/*/forces.dat"))
    paths += sorted((root / "solver_segments").glob("*.log"))
    paths += sorted(root.glob("[0-9]*/polyMesh/points"))
    for relative in ("constant/dynamicMeshDict","constant/triSurface/rotating.stl",
                     "system/snappyHexMeshDict","system/createBafflesDict",
                     "constant/polyMesh/boundary","constant/polyMesh/faces"):
        if (root/relative).is_file(): paths.append(root/relative)
    if (root / "constant/triSurface/source_hull.stl").is_file():
        paths.append(root / "constant/triSurface/source_hull.stl")
    if (root / "constant/dynamicMeshDict").is_file():
        paths.append(root / "checkMesh.latest.log")
    if any(not path.is_file() for path in paths) or not any("forces.dat" == p.name for p in paths):
        raise CFDExecutionError("case provenance files are incomplete")
    return {str(path.relative_to(root)): _sha(path) for path in paths}


def _metadata(root: Path) -> dict:
    if not (root / "case_metadata.json").is_file():
        raise CFDExecutionError("case metadata is missing")
    value = json.loads((root / "case_metadata.json").read_text())
    if value.get("frame_contract") != CONVENTION or value.get("case_id") != value.get("case_hash"):
        raise CFDExecutionError("case identity or frame contract is invalid")
    source=root / "constant/triSurface/source_hull.stl"
    if not source.is_file():
        if value.get("source_waterline_z_m") != 0:
            raise CFDExecutionError("waterline-shifted case has no original geometry evidence")
        source=root / "constant/triSurface/hull.stl"
    if _sha(source) != value.get("geometry_sha256"):
        raise CFDExecutionError("geometry hash mismatch")
    expected=OpenFOAMAdapter._waterline_aligned_stl(source.read_bytes(),value["source_waterline_z_m"])
    if expected != (root / "constant/triSurface/hull.stl").read_bytes():
        raise CFDExecutionError("waterline geometry transform mismatch")
    return value


def _reference_compatible(case: dict, reference: dict) -> None:
    keys = ("geometry_sha256", "fluid_model", "water_properties", "mesh_settings",
            "domain_settings", "cg_body_frd_m", "moment_reference_point_frd_m",
            "solver_moment_reference_point_m", "source_waterline_z_m", "openfoam_version")
    if any(case.get(key) != reference.get(key) for key in keys):
        raise CFDExecutionError("static reference does not match geometry, waterline, fluid, mesh, or reference point")
    if reference.get("motion_type") != "steady_velocity" or abs(reference.get("magnitude", 1)) > 1e-6:
        raise CFDExecutionError("reference is not a quiescent static case")


def _force(root: Path) -> np.ndarray:
    history = np.asarray(OpenFOAMAdapter.force_history(root), float)
    if history.ndim != 2 or history.shape[1] != 7 or len(history) < 8 or not np.isfinite(history).all() or np.any(np.diff(history[:, 0]) <= 0):
        raise CFDExecutionError("force history is missing, duplicated, or nonfinite")
    return history


def align_motion(time_s, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate commanded displacement, velocity, acceleration at force times."""
    t = np.asarray(time_s, float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise CFDExecutionError("force time basis is invalid")
    axis = AXES[metadata["dof"]]
    q = np.zeros((len(t), 6)); v = q.copy(); a = q.copy()
    motion = metadata["motion_type"]
    if motion.startswith("forced_"):
        omega = float(metadata["frequency_rad_s"]); amplitude = float(metadata["amplitude"])
        if omega <= 0 or amplitude <= 0: raise CFDExecutionError("invalid sinusoidal motion")
        q[:, axis] = amplitude * np.sin(omega * t)
        v[:, axis] = amplitude * omega * np.cos(omega * t)
        a[:, axis] = -amplitude * omega**2 * np.sin(omega * t)
    else:
        speed = float(metadata["magnitude"])
        q[:, axis] = speed * t; v[:, axis] = speed
    return q, v, a


def _numerics(root: Path, metadata: dict) -> dict:
    settings = metadata["solver_settings"]
    result = numerical_history_gate(OpenFOAMAdapter.diagnose_case(root),
        expected_start_s=0, expected_end_s=settings["end_time_s"],
        max_courant=settings["max_courant"], max_alpha_courant=settings["max_alpha_courant"],
        max_timestep_s=settings["timestep_s"], residual_limit=max(settings["residual_tolerance"], 1e-5))
    if not result["accepted"]: raise CFDExecutionError(f"numerical qualification failed: {result['gates']}")
    if not (root / "checkMesh.log").is_file():
        raise CFDExecutionError("mesh quality evidence is missing")
    check = (root / "checkMesh.log").read_text()
    if "Cell volumes OK" not in check or "Non-orthogonality check OK" not in check:
        raise CFDExecutionError("mesh quality evidence is missing")
    if metadata["motion_type"] != "steady_velocity":
        latest=(root / "checkMesh.latest.log")
        if not latest.is_file() or "Cell volumes OK" not in latest.read_text() or "Non-orthogonality check OK" not in latest.read_text():
            raise CFDExecutionError("deformed mesh quality evidence is missing")
        initial_concave=re.search(r"Concave cells .*number of cells: (\d+)",check)
        final_concave=re.search(r"Concave cells .*number of cells: (\d+)",latest.read_text())
        if final_concave and (not initial_concave or int(final_concave.group(1)) > int(initial_concave.group(1))):
            raise CFDExecutionError("mesh concavity increased during motion")
        log = (root / "solver.log").read_text()
        rotating=metadata["motion_type"]=="steady_rotation"
        motion_marker=("Selecting motion solver: solidBody" if rotating else "Solving for cellDisplacement")
        if "Selecting fvMeshMover motionSolver" not in log or motion_marker not in log:
            raise CFDExecutionError("mesh motion is not evidenced")
        if rotating and "nonConformalCyclic_on_nonCouple1 min/avg/max mesh flux error" not in log:
            raise CFDExecutionError("steady rotation has no NCC evidence")
        baseline = OpenFOAMAdapter._mesh_points(root / "constant/polyMesh/points")
        moved = [OpenFOAMAdapter._mesh_points(p) for p in root.glob("[0-9]*/polyMesh/points")]
        if not moved or any(p.shape != baseline.shape for p in moved) or max(np.max(abs(p-baseline)) for p in moved) <= 1e-8:
            raise CFDExecutionError("written mesh did not move")
    return result


def _window(history: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
    t = history[:, 0]; axis = AXES[metadata["dof"]]; motion = metadata["motion_type"]
    if motion == "steady_rotation":
        period=2*math.pi/abs(float(metadata["magnitude"]))
        complete=int(t[-1]//period)
        if complete<3:
            raise CFDExecutionError("steady rotation requires a startup revolution and two complete fitting revolutions")
        start=(complete-2)*period;stop=complete*period
        selected=history[(t>=start)&(t<=stop)]
        if len(selected)<32 or np.max(np.diff(selected[:,0]))>period/8:
            raise CFDExecutionError("steady rotation has incomplete revolution sampling")
        phase=np.linspace(0,1,65)
        waves=[]
        for revolution in range(2):
            begin=start+revolution*period
            segment=selected[(selected[:,0]>=begin)&(selected[:,0]<=begin+period)]
            if len(segment)<16:raise CFDExecutionError("steady rotation revolution is undersampled")
            waves.append(np.interp(begin+phase*period,segment[:,0],segment[:,1+axis]))
        scale=max(float(np.ptp(np.concatenate(waves))), 2. if axis>=3 else 10.)
        difference=float(np.sqrt(np.mean((waves[0]-waves[1])**2))/scale)
        if difference>.1:raise CFDExecutionError("steady rotation is not orientation-periodic")
        return selected,{"method":"two_orientation_revolutions_after_startup",
            "complete_revolutions":complete,"phase_rms_difference":difference,
            "active_channel":axis,"bounds_s":[start,stop]}
    if motion.startswith("forced_"):
        period = 2 * math.pi / float(metadata["frequency_rad_s"])
        complete = int(t[-1] // period)
        if complete < 4: raise CFDExecutionError("fewer than four completed forced-motion cycles")
        start = (complete-3)*period; stop = complete*period
        selected = history[(t >= start) & (t <= stop)]
        if len(selected) < 24 or np.max(np.diff(selected[:, 0])) > period/8:
            raise CFDExecutionError("forced-motion fitting samples have gaps or insufficient cadence")
        phases = (selected[:, 0]-start)/period
        if phases[0] > .125 or 3-phases[-1] > .125:
            raise CFDExecutionError("forced-motion cycle boundaries are missing")
        # A single anomalous force sample has two large adjacent jumps; keep
        # smooth physical peaks and harmonics, but reject isolated impulses.
        jumps=np.diff(selected[:, 1+axis])
        typical=max(float(np.median(abs(jumps))), .01)
        if any(abs(jumps[i]) > 12*typical and abs(jumps[i+1]) > 12*typical and
               jumps[i]*jumps[i+1] < 0 for i in range(len(jumps)-1)):
            raise CFDExecutionError("isolated force spike in fitting window")
        # Fit known-frequency harmonics independently in three cycles. This
        # checks phase, amplitude and mean without mistaking oscillation for drift.
        metrics=[]
        for cycle in range(3):
            subset = selected[(selected[:, 0] >= start+cycle*period) & (selected[:, 0] < start+(cycle+1)*period)]
            if len(subset) < 8: raise CFDExecutionError("one forced-motion cycle is undersampled")
            theta = float(metadata["frequency_rad_s"])*subset[:, 0]
            design = np.column_stack((np.ones(len(subset)), np.sin(theta), np.cos(theta)))
            metrics.append(np.linalg.lstsq(design, subset[:, 1+axis], rcond=None)[0])
        coeff = np.asarray(metrics); mean = coeff.mean(axis=0)
        scale = max(np.hypot(mean[1], mean[2]), 1e-9)
        spread = float(np.max(np.linalg.norm(coeff[:, 1:]-mean[1:], axis=1))/scale)
        endpoint_change = float(np.linalg.norm(coeff[-1, 1:]-coeff[0, 1:])/scale)
        mean_spread = float(np.ptp(coeff[:, 0])/max(scale, 1.))
        if spread > .25 or endpoint_change > .25 or mean_spread > .25:
            raise CFDExecutionError("forced-motion cycles are not repeatable")
        return selected, {"method": "three_complete_cycles_after_startup", "complete_cycles": complete,
                          "harmonic_spread": spread, "endpoint_harmonic_change": endpoint_change,
                          "mean_spread": mean_spread,
                          "active_channel": axis, "bounds_s": [start, stop]}
    duration = min(max(.04, .25*t[-1]), .5*t[-1])
    report = wrench_stationarity(t, history[:, 1:], force_scale_n=100., moment_scale_nm=20., window_s=duration)
    active = report.channels[axis]
    if not active.accepted:
        raise CFDExecutionError("active steady load is not stationary")
    selected = history[t >= t[-1]-duration]
    for index in range(6):
        if index != axis and np.std(selected[:,1+index]) > (.5*100 if index < 3 else .5*20):
            raise CFDExecutionError("gross cross-axis fluctuation in steady fitting window")
    return selected, {"method": "active_channel_steady_tail", "active_channel": axis,
                      "active": asdict(active), "bounds_s": [float(selected[0, 0]), float(selected[-1, 0])]}


def qualify_case(case_root: str | Path, reference_root: str | Path, output: str | Path) -> Path:
    """Write an artifact only after numerical, reference, and window gates pass."""
    root = Path(case_root).resolve(); ref = Path(reference_root).resolve()
    case = _metadata(root); reference = _metadata(ref); _reference_compatible(case, reference)
    if _sha(root / "constant/polyMesh/points") != _sha(ref / "constant/polyMesh/points"):
        raise CFDExecutionError("motion and static reference do not share the same initial mesh")
    if case["dof"] in {"heave", "roll", "pitch"}:
        raise CFDExecutionError("position-dependent hydrostatic restoring reference is required for this DOF")
    numeric = _numerics(root, case); reference_numeric = _numerics(ref, reference)
    history = _force(root); static = _force(ref)
    selected, quality = _window(history, case)
    static_selected, static_quality = _window(static, reference)
    static_report=wrench_stationarity(static[:, 0], static[:, 1:], force_scale_n=100.,
        moment_scale_nm=20., window_s=static_quality["bounds_s"][1]-static_quality["bounds_s"][0])
    if not static_report.accepted:
        raise CFDExecutionError("static reference wrench is not stationary in all channels")
    static_quality["all_channels"]=[asdict(channel) for channel in static_report.channels]
    baseline = static_selected[:, 1:].mean(axis=0)
    q, v, a = align_motion(selected[:, 0], case)
    corrected = selected[:, 1:]-baseline
    transformed = np.asarray([sum(foam_wrench_to_body(row[:3], row[3:],
        foam_reference=case["solver_moment_reference_point_m"],
        body_reference=case["moment_reference_point_frd_m"],
        waterline_z_m=case["source_waterline_z_m"]), ()) for row in corrected])
    observations = [{"time_s": float(t), "displacement": q[i].tolist(), "velocity": v[i].tolist(),
                     "acceleration": a[i].tolist(), "resisting_wrench": transformed[i].tolist()}
                    for i, t in enumerate(selected[:, 0])]
    window_hash = hashlib.sha256(np.asarray(selected, dtype="<f8").tobytes()).hexdigest()
    artifact = {"schema": SCHEMA, "quality_status": "CFD_QUALIFIED", "parser_version": PARSER,
        "transform_version": CONVENTION, "case_id": case["case_id"], "geometry_sha256": case["geometry_sha256"],
        "case_hash": case["case_hash"], "frame_contract": CONVENTION,
        "openfoam_version": case["openfoam_version"],
        "openfoam_runtime": json.loads((root / "openfoam_runtime.json").read_text()),
        "motion_definition": {k: case[k] for k in ("motion_type", "dof", "magnitude", "frequency_rad_s", "amplitude")},
        "fluid_definition": case["water_properties"], "mesh_definition": case["mesh_settings"],
        "moment_reference_point_frd_m": case["moment_reference_point_frd_m"],
        "waterline_z_m": case["source_waterline_z_m"], "reference_case_id": reference["case_id"],
        "static_reference_wrench_foam": baseline.tolist(), "static_reference_quality": static_quality,
        "case_root": str(root), "reference_root": str(ref), "case_sources": _sources(root),
        "reference_sources": _sources(ref), "numerical_quality": numeric,
        "reference_numerical_quality": reference_numeric, "experiment_quality": quality,
        "fitting_window_s": quality["bounds_s"], "fitting_window_sha256": window_hash,
        "observations": observations}
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, sort_keys=True, indent=2))
    return target


def verify_artifact(path: str | Path) -> dict:
    """Recompute the artifact from immutable source evidence before ingestion."""
    path = Path(path); artifact = json.loads(path.read_text())
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA or artifact.get("quality_status") != "CFD_QUALIFIED" or artifact.get("frame_contract") != CONVENTION or artifact.get("parser_version") != PARSER:
        raise CFDExecutionError("unqualified or unsupported observation artifact")
    for root_key, source_key in (("case_root", "case_sources"), ("reference_root", "reference_sources")):
        root = Path(artifact[root_key])
        if _sources(root) != artifact[source_key]: raise CFDExecutionError("observation source hash mismatch")
    # A hand-edited JSON payload cannot supply observations that differ from
    # those derived from the hashed solved cases.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        regenerated = Path(directory)/"artifact.json"
        qualify_case(artifact["case_root"], artifact["reference_root"], regenerated)
        expected = json.loads(regenerated.read_text())
    if artifact != expected: raise CFDExecutionError("observation artifact differs from solved-case derivation")
    return artifact
