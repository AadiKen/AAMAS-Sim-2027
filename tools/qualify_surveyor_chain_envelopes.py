"""Assess source fidelity and inferred reconstructed wetted area for chain lofts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import trimesh

from qualify_surveyor_envelopes import _sample_source, _surface_distances, _to_frd


def run(source_path: Path, root: Path) -> dict:
    source_mm = trimesh.load_mesh(source_path, process=True)
    sources = {"port": next(part for part in source_mm.split(only_watertight=False)
                            if part.bounds[1, 0] < 0),
               "starboard": next(part for part in source_mm.split(only_watertight=False)
                                 if part.bounds[0, 0] > 0)}
    points = {side: _sample_source(part, 50, 1750) for side, part in sources.items()}
    original_frd = source_mm.copy()
    original_frd.vertices = _to_frd(original_frd.vertices)
    hydro = json.loads((root/"hydrostatics_52_3kg.json").read_text())
    result = {"schema": "surveyor-chain-envelope-qualification-v1",
              "source_point_sample": "12,000 vertices and triangle centroids per side, deterministic random subsample",
              "explicit_reconstruction_review_zones_source_Z_mm": [[325, 350], [900, 1200]],
              "distance_gate_outside_review_zones_mm": {"p95": 2., "max": 5.},
              "wetted_area_estimate": "area-weighted 30,000 wet triangle-centroid samples; distance >2 mm to selected original mixed-group faces",
              "hypotheses": {}}
    for name in ("smooth", "inset", "outset"):
        mesh = trimesh.load_mesh(root/f"surveyor_{name}_FRD_m.stl", process=True)
        waterline = hydro[name]["equilibrium_52_3kg"]["waterline_z_frd_m"]
        side_reports = {}
        all_distances, all_points = [], []
        for side, pts in points.items():
            distances = _surface_distances(pts, mesh)
            source_z = pts[:, 0]*1000+915
            outside = ((source_z<900)|(source_z>1200))&((source_z<325)|(source_z>350))
            underwater = pts[:, 2] >= waterline
            assessed = distances[outside & underwater]
            side_reports[side] = {"source_point_count": len(pts),
                "all_distance_mm": {"p95": float(np.percentile(distances,95)*1000),
                                    "max": float(distances.max()*1000)},
                "outside_review_underwater_count": len(assessed),
                "outside_review_underwater_distance_mm": {
                    "p95": float(np.percentile(assessed,95)*1000),
                    "max": float(assessed.max()*1000)}}
            all_distances.append(distances); all_points.append(pts)
        all_distances = np.concatenate(all_distances)
        all_points = np.vstack(all_points)
        source_z = all_points[:, 0]*1000+915
        outside = ((source_z<900)|(source_z>1200))&((source_z<325)|(source_z>350))
        underwater = all_points[:, 2] >= waterline
        assessed = all_distances[outside & underwater]
        wet_faces = np.flatnonzero(mesh.triangles_center[:, 2]>=waterline)
        area = mesh.area_faces[wet_faces]
        rng = np.random.default_rng(43)
        sampled_faces = rng.choice(wet_faces, size=30000, replace=True, p=area/area.sum())
        wet_distances = _surface_distances(mesh.triangles_center[sampled_faces], original_frd)
        unsupported_fraction = float(np.mean(wet_distances>.002))
        wetted_area = hydro[name]["equilibrium_52_3kg"]["wetted_area_m2"]
        result["hypotheses"][name] = {
            "sides": side_reports,
            "all_original_surface_distance_mm": {"p95": float(np.percentile(all_distances,95)*1000),
                                                  "max": float(all_distances.max()*1000)},
            "all_original_underwater_distance_mm": {
                "p95": float(np.percentile(all_distances[underwater],95)*1000),
                "max": float(all_distances[underwater].max()*1000)},
            "outside_review_underwater_distance_mm": {
                "p95": float(np.percentile(assessed,95)*1000),
                "max": float(assessed.max()*1000)},
            "outside_review_distance_gate_pass": bool(np.percentile(assessed,95)<.002 and
                                                      assessed.max()<.005),
            "estimated_reconstructed_wetted_fraction_gt2mm": unsupported_fraction,
            "estimated_reconstructed_wetted_area_m2_gt2mm": unsupported_fraction*wetted_area,
            "wetted_area_m2": wetted_area,
        }
    (root/"source_fidelity.json").write_text(json.dumps(result, indent=2) + "\n")
    print({name: {"gate": row["outside_review_distance_gate_pass"],
                   "area_fraction": row["estimated_reconstructed_wetted_fraction_gt2mm"]}
           for name, row in result["hypotheses"].items()})
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/qualify_surveyor_chain_envelopes.py SOURCE_STL HYPOTHESES_DIR")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
