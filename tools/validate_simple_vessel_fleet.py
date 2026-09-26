"""Small deterministic, synthetic fleet for the CFD-free generation route."""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import trimesh

from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel

ROOT = Path(os.environ.get("BCOD_FLEET_OUTPUT", "stage3_results/simple_hydrodynamics/fleet"))
ROOT.mkdir(parents=True, exist_ok=True)


def box(extents):
    return trimesh.creation.box(extents=extents)


def ellipsoid(extents, subdivisions=2):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    mesh.apply_scale(np.asarray(extents) / 2)
    return mesh


def two_hulls():
    left, right = box((4., .7, 1.)), box((4., .7, 1.))
    left.apply_translation((0., -1., 0.))
    right.apply_translation((0., 1., 0.))
    return trimesh.util.concatenate((left, right))


def vee():
    points = [(x, y, -.5) for x in (-2., 2.) for y in (-1., 1.)]
    points += [(x, 0., .5) for x in (-2., 2.)]
    return trimesh.convex.convex_hull(np.asarray(points))


def with_appendage():
    hull, skeg = box((4., 2., 1.)), box((.7, .08, .25))
    skeg.apply_translation((-.8, 0., .63))
    return trimesh.util.concatenate((hull, skeg))


cases = [
    ("displacement_monohull", box((4., 1.6, 1.)), 1640., 1.),
    ("slender_monohull", ellipsoid((6., .7, .8)), 250., 1.),
    ("catamaran", two_hulls(), 1435., 1.),
    ("flat_bottom_usv", box((3., 1.5, .5)), 576.5625, 1.),
    ("deep_v", vee(), 1025., 1.),
    ("planing_candidate", box((4., 2., 1.)), 2050., 7.),
    ("appendage_candidate", with_appendage(), 2050., 1.),
    ("messy_repairable", box((4., 2., 1.)), 2050., 1.),
]
rows = []
for name, mesh, mass, speed in cases:
    if name == "messy_repairable":
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=np.vstack((mesh.faces, mesh.faces[:2])), process=False)
    path = ROOT / f"{name}.stl"
    mesh.export(path)
    start = perf_counter()
    try:
        root = generate_simple_vessel(geometry=path, output=ROOT / name,
            mass_kg=mass, cg_frd_m=(0., 0., 0.), units="m", speed_range_mps=(0., speed),
            disable_bem=False, lut_samples=9)
        provenance = json.loads((root / "provenance.json").read_text())
        confidence = json.loads((root / "confidence.json").read_text())
        validation = json.loads((root / "validation.json").read_text())
        rows.append({"hull": name, "classification": provenance["classification"]["classification"],
                     "bem": provenance["bem"]["status"],
                     "fallback": "strip" if provenance["bem"]["status"] != "accepted" else "not used",
                     "runtime_s": perf_counter() - start,
                     "overall_confidence": confidence["overall_passive_model"],
                     "result": "PASS" if validation["passed"] else "FAIL",
                     "faces": int(provenance["hydrodynamic_mesh_reduction"]["hydrodynamic_faces"])})
    except Exception as exc:
        rows.append({"hull": name, "classification": "unavailable", "bem": "not run",
                     "fallback": "none", "runtime_s": perf_counter() - start,
                     "overall_confidence": "unavailable", "result": f"EXPLICIT FAILURE: {exc}"})
(ROOT / "fleet_results.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
