"""Measure source-surface fidelity, reconstructed area and displacement capacity."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def _to_frd(points_mm: np.ndarray) -> np.ndarray:
    return np.column_stack(((points_mm[:, 2]-915)/1000,
                            -points_mm[:, 0]/1000, -points_mm[:, 1]/1000))


def _sample_source(source: trimesh.Trimesh, min_z: float = 330,
                   max_z: float = 1740, limit: int = 12000) -> np.ndarray:
    points = np.vstack((source.vertices, source.triangles_center))
    points = points[(points[:, 2] >= min_z) & (points[:, 2] <= max_z)]
    if len(points) > limit:
        indices = np.random.default_rng(11).choice(len(points), size=limit, replace=False)
        points = points[indices]
    return _to_frd(points)


def _surface_distances(points: np.ndarray, mesh: trimesh.Trimesh, neighbours: int = 16) -> np.ndarray:
    triangles = mesh.triangles
    tree = cKDTree(triangles.mean(axis=1))
    distances = np.empty(len(points))
    for start in range(0, len(points), 500):
        stop = min(len(points), start+500)
        chunk = points[start:stop]
        _, indices = tree.query(chunk, k=neighbours)
        candidates = triangles[indices].reshape(-1, 3, 3)
        repeated = np.repeat(chunk, neighbours, axis=0)
        closest = trimesh.triangles.closest_point(candidates, repeated)
        distances[start:stop] = np.linalg.norm(repeated-closest, axis=1).reshape(-1, neighbours).min(axis=1)
    return distances


def run(source_path: Path, hypothesis_dir: Path) -> dict:
    source = trimesh.load_mesh(source_path, process=True)
    components = sorted(source.split(only_watertight=False), key=lambda part: part.area, reverse=True)
    constraints = np.load(hypothesis_dir / "section_constraints.npz")
    stations = constraints["stations_mm"]
    angles = constraints["angles_rad"]
    centre_x = float(constraints["centre_x_mm"]) if "centre_x_mm" in constraints else -338.
    centre_y = float(constraints["centre_y_mm"]) if "centre_y_mm" in constraints else -190.
    origin_z = float(constraints["longitudinal_origin_mm"]) if "longitudinal_origin_mm" in constraints else 915.
    source_points = np.vstack([_sample_source(part, stations[0], stations[-1])
                               for part in components[:3]])
    observed = constraints["source_observed"]
    station_count, angular_count = observed.shape
    result = {"schema": "surveyor-envelope-qualification-v1",
              "source_distance_sampling": "original selected main-hull STEP-face vertices and triangle centroids; 330-1740 mm source Z",
              "source_point_count": len(source_points),
              "density_kg_m3": 1025., "published_mass_kg": 52.3,
              "required_displacement_m3": 52.3/1025.,
              "published_draft_m": .17, "hypotheses": {}}
    for name in ("smooth", "inset", "outset"):
        mesh = trimesh.load_mesh(hypothesis_dir / f"surveyor_{name}_FRD_m.stl", process=True)
        distances = _surface_distances(source_points, mesh)
        centroids = mesh.triangles_center
        station_index = np.clip(np.rint((centroids[:, 0]*1000+origin_z-stations[0])/
                                       (stations[1]-stations[0])).astype(int),
                                0, station_count-1)
        source_x = -np.abs(centroids[:, 1])*1000
        source_y = -centroids[:, 2]*1000
        theta = np.mod(np.arctan2(source_y-centre_y, source_x-centre_x), 2*np.pi)
        angle_index = np.mod(np.rint(theta/(2*np.pi)*angular_count).astype(int), angular_count)
        face_reconstructed = ~observed[station_index, angle_index]
        face_reconstructed |= (station_index == 0) | (station_index == station_count-1)
        reconstructed_area = float(mesh.area_faces[face_reconstructed].sum())
        keel = float(mesh.bounds[1, 2])
        published_waterline = keel-.17
        full_submerged = bool(published_waterline < mesh.bounds[0, 2])
        result["hypotheses"][name] = {
            "source_distance_mm": {"p50": float(np.percentile(distances, 50)*1000),
                                    "p95": float(np.percentile(distances, 95)*1000),
                                    "max": float(distances.max()*1000)},
            "source_distance_p95_gate_pass": bool(np.percentile(distances, 95) < .002),
            "source_distance_max_gate_pass": bool(distances.max() < .005),
            "reconstructed_surface_area_m2_upper_bound": reconstructed_area,
            "reconstructed_surface_fraction_upper_bound": reconstructed_area/mesh.area,
            "total_surface_area_m2": float(mesh.area),
            "total_volume_m3": float(mesh.volume),
            "maximum_displacement_kg_in_seawater": float(mesh.volume*1025),
            "nominal_displacement_margin_m3": float(mesh.volume-52.3/1025),
            "keel_z_frd_m": keel,
            "waterline_at_published_0_17m_draft_z_frd_m": published_waterline,
            "entire_reconstructed_envelope_submerged_at_published_draft": full_submerged,
            "nominal_mass_hydrostatic_capacity": (
                "PASS_CAPACITY_ONLY" if mesh.volume > 52.3/1025
                else "FAIL_MASS_EXCEEDS_DISPLACEMENT_CAPACITY")
        }
    (hypothesis_dir / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/qualify_surveyor_envelopes.py SOURCE_STL HYPOTHESIS_DIR")
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2]))["hypotheses"], indent=2))
