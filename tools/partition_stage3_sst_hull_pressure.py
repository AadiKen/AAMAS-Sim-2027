"""Post-process existing WAM-V hull pressure by geometric region.

No mesh or solution fields are modified. The summed pressure wrench is
cross-checked against the OpenFOAM forces function-object output.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analyze_stage3_sst_diagnostic_probes import CASE, force_file


def nonuniform(path: Path, *, patch: str | None = None) -> np.ndarray:
    text = path.read_text()
    if patch is not None:
        match = re.search(rf"\b{re.escape(patch)}\s*\{{(.*?)\n\s*\}}", text, re.S)
        if match is None:
            raise ValueError(f"missing patch {patch}: {path}")
        text = match.group(1)
    match = re.search(r"nonuniform List<scalar>\s*(\d+)\s*\((.*?)\)", text, re.S)
    if match is None:
        raise ValueError(f"missing scalar list: {path}")
    data = np.fromstring(match.group(2), sep=" ")
    if len(data) != int(match.group(1)):
        raise ValueError(f"incomplete scalar list: {path}")
    return data


def geometry(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh = root / "constant/polyMesh"
    boundary = (mesh / "boundary").read_text()
    patch = re.search(r"\bhull\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);",
                      boundary, re.S)
    if patch is None:
        raise ValueError("missing hull patch")
    count, start = (int(x) for x in patch.groups())
    points_text = (mesh / "points").read_text()
    points_match = re.search(r"\n(\d+)\s*\(\n(.*?)\n\)", points_text, re.S)
    points = np.fromstring(points_match.group(2).replace("(", " ").replace(")", " "),
                           sep=" ").reshape((-1, 3))
    if len(points) != int(points_match.group(1)):
        raise ValueError("incomplete mesh points")
    faces_text = (mesh / "faces").read_text()
    faces_match = re.search(r"\n(\d+)\s*\(\n(.*?)\n\)", faces_text, re.S)
    faces = faces_match.group(2).splitlines()[start:start + count]
    owner_text = (mesh / "owner").read_text()
    owner_match = re.search(r"\n(\d+)\s*\((.*?)\)", owner_text, re.S)
    owners = np.fromstring(owner_match.group(2), sep=" ").astype(int)[start:start + count]
    centers = np.empty((count, 3))
    areas = np.empty((count, 3))
    for i, line in enumerate(faces):
        ids = np.fromstring(line.split("(", 1)[1].split(")", 1)[0], sep=" ").astype(int)
        vertices = points[ids]
        centers[i] = vertices.mean(axis=0)
        areas[i] = .5 * np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)
    return centers, areas, owners, np.arange(count)


def partition(root: Path, times: list[str]) -> dict:
    centers, areas, owners, _ = geometry(root)
    origin = np.asarray((0., 0., -.10601493958702342))
    norm = np.linalg.norm(areas, axis=1)
    lateral = np.divide(areas[:, 1], norm, out=np.zeros(len(norm)), where=norm > 0)
    side = np.where(centers[:, 1] >= 0, 1., -1.)
    masks = {
        "bow": centers[:, 0] < 0,
        "stern": centers[:, 0] >= 0,
        "port": centers[:, 1] >= 0,
        "starboard": centers[:, 1] < 0,
        "inner_lateral": (side * lateral) > .5,
        "outer_lateral": (side * lateral) < -.5,
        "other_facing": np.abs(lateral) <= .5,
    }
    result = {}
    force_time, pressure_force, _ = force_file(root / "postProcessing/forces/60/forces.dat")
    for time in times:
        pressure = nonuniform(root / time / "p", patch="hull")
        alpha = nonuniform(root / time / "alpha.water")[owners]
        if len(pressure) != len(areas):
            raise ValueError(f"hull pressure/mesh length mismatch at {time}")
        force = pressure[:, None] * areas
        moment = np.cross(centers - origin, force)
        masks_at_time = dict(masks)
        masks_at_time["wet_owner"] = alpha > .9
        masks_at_time["interface_owner"] = (alpha >= .1) & (alpha <= .9)
        masks_at_time["dry_owner"] = alpha < .1
        summaries = {}
        for name, mask in masks_at_time.items():
            summaries[name] = {
                "force_N": force[mask].sum(axis=0).tolist(),
                "moment_Nm": moment[mask].sum(axis=0).tolist(),
                "area_m2": float(norm[mask].sum()),
                "face_count": int(mask.sum()),
            }
        point = float(time)
        nearest = int(np.argmin(np.abs(force_time - point)))
        summaries["cross_check"] = {
            "time_s": float(force_time[nearest]),
            "computed_pressure_force_N": force.sum(axis=0).tolist(),
            "computed_pressure_moment_Nm": moment.sum(axis=0).tolist(),
            "forces_output_pressure_force_N": pressure_force[nearest, :3].tolist(),
            "forces_output_pressure_moment_Nm": pressure_force[nearest, 3:].tolist(),
        }
        result[time] = summaries
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--times", nargs="+", default=["60", "65", "70", "75", "80"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = partition(args.case, args.times)
    output = args.output or args.case.parent / "diagnostic_60_80/pressure_partitions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    print(output)


if __name__ == "__main__":
    main()
