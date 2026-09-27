"""Saved-data harmonic-yaw diagnostics; physical fluid-on-hull loads only."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

FLIP = np.diag([1., -1., -1.])  # body FRD -> OpenFOAM XYZ at zero yaw
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"


def require_compatible(a: dict, b: dict) -> None:
    """Refuse comparisons whose load/reference convention cannot be established."""
    required = ("load_kind", "frame", "moment_origin", "cg_world_m", "phase", "normalization")
    if any(a.get(k) is None or b.get(k) is None for k in required):
        raise ValueError("Unresolved wrench, reference, phase, or normalization metadata")
    if any(a[k] != b[k] for k in ("load_kind", "frame", "moment_origin", "phase", "normalization")):
        raise ValueError("Incompatible wrench/reference metadata")
    if not np.allclose(a["cg_world_m"], b["cg_world_m"], rtol=0, atol=1e-9):
        raise ValueError("Incompatible CG coordinates")


def world_to_body_cg(force_world, moment_world_o, *, origin_world, cg_world, yaw_foam_rad,
                     input_frame="foam_world", load_kind="physical_fluid_on_hull"):
    """Rotate and shift a fixed-origin OpenFOAM world wrench to moving CG/FRD."""
    if input_frame != "foam_world" or load_kind != "physical_fluid_on_hull":
        raise ValueError("Expected untransformed physical OpenFOAM world wrench")
    f = np.asarray(force_world, float)
    m = np.asarray(moment_world_o, float)
    o = np.asarray(origin_world, float)
    cg = np.asarray(cg_world, float)
    c, s = math.cos(yaw_foam_rad), math.sin(yaw_foam_rad)
    rotation_world_body = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ FLIP
    rotation_body_world = rotation_world_body.T
    return rotation_body_world @ f, rotation_body_world @ (m + np.cross(o-cg, f))


def read_forces(case: Path) -> np.ndarray:
    """Read pressure+viscous forces, with newest restart owning overlapping times."""
    paths = sorted(case.glob("postProcessing/forces/*/forces.dat"), key=lambda p: float(p.parent.name))
    samples = {}
    for i, path in enumerate(paths):
        start = float(path.parent.name)
        samples = {t: row for t, row in samples.items() if t < start}
        next_start = float(paths[i+1].parent.name) if i+1 < len(paths) else math.inf
        header = path.read_text().splitlines()[:8]
        if not any("CofR" in line and re.search(rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", line)
                   and all(float(z) == 0 for z in re.search(rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", line).groups())
                   for line in header):
            raise ValueError(f"Unresolved force origin: {path}")
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            t = float(line.split()[0])
            if t >= next_start:
                continue
            vectors = re.findall(rf"\(({NUMBER})\s+({NUMBER})\s+({NUMBER})\)", line)
            if len(vectors) != 4:
                raise ValueError(f"Malformed force sample: {path} at {t}")
            v = np.asarray(vectors, float)
            samples[t] = np.r_[t, v[0]+v[1], v[2]+v[3]]
    if not samples:
        raise ValueError(f"No force samples: {case}")
    out = np.asarray([samples[t] for t in sorted(samples)])
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite force history")
    return out


def read_motion(case: Path) -> np.ndarray:
    rows = []
    for line in (case / "constant/6DoF.dat").read_text().splitlines():
        numbers = re.findall(NUMBER, line)
        if line.lstrip().startswith("(") and len(numbers) == 7:
            rows.append([float(z) for z in numbers])
    data = np.asarray(rows)
    if len(data) < 2 or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Invalid motion table")
    return data


def interpolate_motion(motion: np.ndarray, t: np.ndarray):
    if t.min() < motion[0, 0]-1e-8 or t.max() > motion[-1, 0]+1e-8:
        raise ValueError("Forces outside motion table")
    translation = np.column_stack([np.interp(t, motion[:, 0], motion[:, j]) for j in (1, 2, 3)])
    theta = np.deg2rad(np.interp(t, motion[:, 0], motion[:, 6]))
    # Differentiate the prescribed motion table, then interpolate derivatives.
    vy = np.interp(t, motion[:, 0], np.gradient(motion[:, 2], motion[:, 0]))
    theta_dot = np.interp(t, motion[:, 0], np.gradient(np.deg2rad(motion[:, 6]), motion[:, 0]))
    return translation, theta, vy, theta_dot


def body_kinematics(theta: np.ndarray, vy_world: np.ndarray, theta_dot: np.ndarray,
                    farfield_world=(-1.953, 0., 0.)):
    """Vessel relative-to-water velocity at CG for zero local CG offset."""
    far = np.asarray(farfield_world, float)
    c, s = np.cos(theta), np.sin(theta)
    # minus body transform of fluid velocity relative to moving CG
    u = -c*far[0]-s*far[1]+s*vy_world
    v = -s*far[0]+c*far[1]-c*vy_world
    return u, v, -theta_dot


def harmonic(t, values, frequency_hz=.08, max_harmonic=3):
    """Timestamp-weighted signed Fourier coefficients; rate=cos, accel=-sin."""
    t, values = np.asarray(t, float), np.asarray(values, float)
    if len(t) < 10 or np.any(np.diff(t) <= 0):
        raise ValueError("Irregular timestamps must be strictly increasing")
    phase = 2*math.pi*frequency_hz*t
    columns = [np.ones_like(t)]
    for h in range(1, max_harmonic+1):
        columns.extend((np.cos(h*phase), np.sin(h*phase)))
    design = np.column_stack(columns)
    weights = np.empty(len(t))
    weights[0], weights[-1] = (t[1]-t[0])/2, (t[-1]-t[-2])/2
    weights[1:-1] = (t[2:]-t[:-2])/2
    coef = np.linalg.lstsq(design*np.sqrt(weights[:, None]), values*np.sqrt(weights), rcond=None)[0]
    return {"mean": float(coef[0]), "rate_1": float(coef[1]), "accel_1": float(-coef[2]),
            "amplitude_1": float(math.hypot(coef[1], coef[2])),
            "phase_1_deg": float(math.degrees(math.atan2(coef[2], coef[1]))),
            "rate_3": float(coef[5]), "accel_3": float(-coef[6]),
            "amplitude_3": float(math.hypot(coef[5], coef[6]))}
