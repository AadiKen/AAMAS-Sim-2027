"""Integrate saved +6 degree hull pressure by longitudinal quarter, without CFD."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("stage3_results/kcs-validation/track_b_hmri/static_drift_plus6_outlet_damping")
OUT = ROOT / "saved_pressure_regions.json"
REF_X = -0.0851


def list_body(path: Path) -> str:
    text = path.read_text()
    match = re.search(r"\n\d+\s*\((.*)\)\s*;?\s*(?://.*)?$", text, re.S)
    if not match:
        raise ValueError(f"No OpenFOAM list in {path}")
    return match.group(1)


def hull_geometry(proc: Path) -> tuple[np.ndarray, np.ndarray]:
    poly = proc / "constant/polyMesh"
    boundary = (poly / "boundary").read_text()
    match = re.search(r"\bhull\s*\{([^}]*)\}", boundary, re.S)
    if not match:
        raise ValueError(f"No hull patch in {poly}")
    block = match.group(1)
    n = int(re.search(r"nFaces\s+(\d+)", block).group(1))
    start = int(re.search(r"startFace\s+(\d+)", block).group(1))
    point_rows = re.findall(r"\(([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\)", list_body(poly / "points"))
    points = np.asarray(point_rows, float)
    face_rows = list_body(poly / "faces").splitlines()[1+start:1+start+n]
    centres = np.empty((n, 3)); areas = np.empty((n, 3))
    for i, line in enumerate(face_rows):
        ids = np.fromstring(line[line.index("(")+1:line.index(")")], sep=" ", dtype=int)
        verts = points[ids]
        centres[i] = verts.mean(axis=0)
        areas[i] = .5 * sum((np.cross(verts[j]-verts[0], verts[j+1]-verts[0])
                             for j in range(1, len(verts)-1)), np.zeros(3))
    return centres, areas


def hull_pressure(path: Path) -> np.ndarray:
    text = path.read_text()
    match = re.search(r"\bhull\s*\{.*?value\s+nonuniform List<scalar>\s+(\d+)\s*\((.*?)\)", text, re.S)
    if not match:
        raise ValueError(f"No hull pressure list in {path}")
    values = np.fromstring(match.group(2), sep=" ")
    if len(values) != int(match.group(1)):
        raise ValueError(f"Wrong hull pressure count in {path}")
    return values


def main() -> None:
    procs = sorted(ROOT.glob("processor*"))
    geom = [hull_geometry(proc) for proc in procs]
    all_x = np.concatenate([centre[:, 0] for centre, _ in geom])
    print("hull x bounds", float(all_x.min()), float(all_x.max()))
    # Flow is toward negative x: largest x is bow, smallest x is stern.
    edges = np.linspace(float(all_x.min()), float(all_x.max()), 5)
    labels = ["stern/transom", "aft-midbody", "forward-midbody", "bow"]
    times = sorted({float(p.name) for p in procs[0].iterdir()
                    if p.is_dir() and re.fullmatch(r"(?:10|11|12|13|14|15|16)(?:\.5)?", p.name)})
    records = []
    for t in times:
        name = f"{t:g}"
        force = np.zeros((4, 3)); moment = np.zeros((4, 3)); counts = np.zeros(4, int)
        for proc, (centres, areas) in zip(procs, geom):
            path = proc / name / "p"
            if not path.exists():
                break
            p = hull_pressure(path)
            if len(p) != len(centres):
                raise ValueError(f"Mesh/pressure mismatch at {path}")
            face_force = p[:, None] * areas
            arms = centres - np.array([REF_X, 0, 0])
            face_moment = np.cross(arms, face_force)
            region = np.clip(np.searchsorted(edges, centres[:, 0], side="right")-1, 0, 3)
            for i in range(4):
                mask = region == i
                force[i] += face_force[mask].sum(axis=0)
                moment[i] += face_moment[mask].sum(axis=0)
                counts[i] += int(mask.sum())
        else:
            records.append({"time_s": t, "pressure_force_foam_by_region": force.tolist(),
                            "pressure_moment_foam_by_region": moment.tolist(),
                            "faces_by_region": counts.tolist(),
                            "pressure_force_foam_total": force.sum(axis=0).tolist(),
                            "pressure_moment_foam_total": moment.sum(axis=0).tolist()})
    OUT.write_text(json.dumps({"edges_x_m": edges.tolist(), "regions": labels,
                               "records": records}, indent=2) + "\n")
    print("saved times", [r["time_s"] for r in records])
    print("output", OUT)


if __name__ == "__main__":
    main()
