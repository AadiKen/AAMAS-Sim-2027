"""Check prescribed hull motion against written OpenFOAM mesh coordinates."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from bcod_sim.vessel_generation.frame_contract import point_to_foam, vector_to_foam


def _foam_list(path: Path, pattern: str) -> list[str]:
    contents = path.read_text()
    match = re.search(r"\n(\d+)\s*\(\s*", contents)
    if match is None:
        raise ValueError(f"OpenFOAM list missing from {path}")
    count = int(match.group(1))
    result = re.findall(pattern, contents[match.end():])[:count]
    if len(result) != count:
        raise ValueError(f"OpenFOAM list truncated in {path}")
    return result


def _points(path: Path) -> np.ndarray:
    return np.asarray([[float(x) for x in row.split()] for row in
                       _foam_list(path, r"\(([-+\d.eE\s]+)\)")])


def _hull_point_ids(mesh: Path) -> np.ndarray:
    boundary = (mesh / "boundary").read_text()
    match = re.search(r"\bhull\s*\{([^}]+)\}", boundary)
    if match is None:
        raise ValueError("hull patch missing")
    body = match.group(1)
    count = int(re.search(r"nFaces\s+(\d+)", body).group(1))
    start = int(re.search(r"startFace\s+(\d+)", body).group(1))
    faces = _foam_list(mesh / "faces", r"\d+\((\d+(?:\s+\d+)*)\)")
    return np.asarray(sorted({int(v) for row in faces[start:start+count]
                              for v in row.split()}))


def verify_motion(case_dir: Path, time_s: float, *, tolerance_m: float=1e-5) -> dict:
    root = Path(case_dir)
    metadata = json.loads((root / "case_metadata.json").read_text())
    mesh = root / "constant/polyMesh"
    ids = _hull_point_ids(mesh)
    original = _points(mesh / "points")[ids]
    moved = _points(root / f"{time_s:g}/polyMesh/points")[ids]
    axis = np.zeros(3)
    axis[("surge", "sway", "heave", "roll", "pitch", "yaw").index(metadata["dof"]) % 3] = 1
    foam_axis = np.asarray(vector_to_foam(axis))
    motion = metadata["motion_type"]
    if motion == "forced_translation":
        displacement = metadata["amplitude"] * math.sin(metadata["frequency_rad_s"] * time_s)
        expected = original + displacement * foam_axis
        velocity = metadata["amplitude"] * metadata["frequency_rad_s"] * math.cos(metadata["frequency_rad_s"] * time_s)
        acceleration = -metadata["amplitude"] * metadata["frequency_rad_s"]**2 * math.sin(metadata["frequency_rad_s"] * time_s)
        commanded = {"displacement_m": displacement, "velocity_mps": velocity,
                     "acceleration_mps2": acceleration}
        observed = {"displacement_m": float(np.mean((moved-original) @ foam_axis))}
    else:
        center = np.asarray(point_to_foam(metadata["moment_reference_point_frd_m"],
                                          metadata["source_waterline_z_m"]))
        if motion == "forced_rotation":
            angle = metadata["amplitude"] * math.sin(metadata["frequency_rad_s"] * time_s)
            rate = metadata["amplitude"] * metadata["frequency_rad_s"] * math.cos(metadata["frequency_rad_s"] * time_s)
            accel = -metadata["amplitude"] * metadata["frequency_rad_s"]**2 * math.sin(metadata["frequency_rad_s"] * time_s)
        elif motion == "steady_rotation":
            angle = metadata["magnitude"] * time_s
            rate = metadata["magnitude"]
            accel = 0.
        else:
            raise ValueError(f"unsupported motion type: {motion}")
        relative = original - center
        expected = center + relative * math.cos(angle) + np.cross(foam_axis, relative) * math.sin(angle) + np.outer(relative @ foam_axis, foam_axis) * (1-math.cos(angle))
        shifted = moved - center
        perpendicular = relative - np.outer(relative @ foam_axis, foam_axis)
        shifted_perp = shifted - np.outer(shifted @ foam_axis, foam_axis)
        sine = np.sum(np.cross(perpendicular, shifted_perp) @ foam_axis)
        cosine = np.sum(perpendicular * shifted_perp)
        observed = {"angle_rad": float(math.atan2(sine, cosine))}
        commanded = {"angle_rad": angle, "angular_rate_radps": rate,
                     "angular_acceleration_radps2": accel}
    errors = np.linalg.norm(expected-moved, axis=1)
    return {"case": str(root), "time_s": time_s, "hull_point_count": len(ids),
            "commanded": commanded, "observed": observed,
            "max_point_error_m": float(errors.max()),
            "mean_point_error_m": float(errors.mean()),
            "accepted": bool(errors.max() <= tolerance_m)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("time_s", type=float)
    args = parser.parse_args()
    print(json.dumps(verify_motion(args.case, args.time_s), indent=2))
