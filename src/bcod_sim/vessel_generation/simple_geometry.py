"""Geometry boundary for the CFD-free vessel generator (metres, body FRD)."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
import trimesh


@dataclass(frozen=True)
class PreparedGeometry:
    mesh: trimesh.Trimesh
    source_hash: str
    processed_hash: str
    report: dict


def _remove_degenerate_faces_preserving_closure(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, bool]:
    """Drop degenerate facets only when doing so preserves a closed surface.

    Some authoritative hull grids encode collapsed stern/keel facets with zero
    area but use their indexed edges to close the volume. Removing those facets
    blindly turns a qualified watertight hull into an open surface.
    """
    candidate = mesh.copy()
    candidate.update_faces(candidate.nondegenerate_faces())
    candidate.update_faces(candidate.unique_faces())
    candidate.remove_unreferenced_vertices()
    removed = len(mesh.faces) - len(candidate.faces)
    if candidate.is_watertight and candidate.is_winding_consistent:
        return candidate, removed, False
    # Preserve topological closure. The retained zero-area faces have no area
    # or volume contribution; their edge incidence is still needed by the mesh.
    fallback = mesh.copy()
    fallback.update_faces(fallback.unique_faces())
    fallback.remove_unreferenced_vertices()
    return fallback, 0, removed > 0


def prepare_geometry(path: str | Path, *, units: str | None = None,
                     known_length_m: float | None = None,
                     source_frame: str = "FRD") -> PreparedGeometry:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".stl", ".obj", ".step", ".stp"}:
        raise ValueError("Supported geometry formats: STL, OBJ, STEP/STP")
    if suffix in {".step", ".stp"}:
        raise ValueError("STEP/STP requires an installed CAD converter; export a triangulated STL/OBJ first")
    if (units is None) == (known_length_m is None):
        raise ValueError("Supply exactly one of units or known_length_m")
    source_hash = sha256(path.read_bytes()).hexdigest()
    loaded = trimesh.load(path, force="scene", process=False)
    parts = [g.copy() for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not parts:
        raise ValueError("No triangle surface in geometry")
    mesh = trimesh.util.concatenate(parts)
    raw_faces = len(mesh.faces)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Non-finite geometry coordinates")
    scale = {"m": 1., "cm": .01, "mm": .001, "ft": .3048, "in": .0254}.get(units) if units else None
    if units and scale is None:
        raise ValueError("Unsupported units")
    if known_length_m is not None:
        if not np.isfinite(known_length_m) or known_length_m <= 0:
            raise ValueError("known_length_m must be positive")
        extent = float(mesh.extents[0])
        if extent <= 0:
            raise ValueError("Zero X extent")
        scale = known_length_m / extent
    assert scale is not None
    mesh.apply_scale(scale)
    if np.max(mesh.extents) > 1000. or np.min(mesh.extents) < .001:
        raise ValueError("INVALID_SCALE: normalized mesh dimensions outside [0.001, 1000] m")
    if source_frame == "FPU":
        mesh.vertices[:, 1:] *= -1
        mesh.invert()
    elif source_frame != "FRD":
        raise ValueError("source_frame must be FRD or FPU")
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=10)
    mesh, removed_degenerate_faces, retained_degenerate_for_closure = _remove_degenerate_faces_preserving_closure(mesh)
    mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Submerged geometry must be a closed orientable surface; no underwater reconstruction is attempted")
    if mesh.volume <= 0:
        mesh.invert()
    if mesh.volume <= 0:
        raise ValueError("No positive enclosed volume")
    components = mesh.split(only_watertight=True)
    edge_vectors = mesh.vertices[mesh.edges_unique[:, 0]] - mesh.vertices[mesh.edges_unique[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    largest = max(float(x.volume) for x in components)
    component_reports = []
    for component in components:
        role = "hull_candidate" if component.volume >= .1 * largest else "unknown_appendage_candidate"
        component_reports.append({"volume_m3": float(component.volume),
                                  "centroid_frd_m": component.center_mass.tolist(),
                                  "bounds_frd_m": component.bounds.tolist(),
                                  "role": role,
                                  "passive_model_treatment": "included in hydrostatics; empirical effects uncalibrated"})
    possible_intersections = []
    for i, first in enumerate(components):
        for j in range(i+1, len(components)):
            second = components[j]
            if np.all(first.bounds[0] < second.bounds[1]) and np.all(second.bounds[0] < first.bounds[1]):
                possible_intersections.append([i, j])
    bounds = np.asarray(mesh.bounds)
    report = {"format": suffix[1:], "source_frame": source_frame, "canonical_frame": "FRD",
              "frame_transform_3x3": ([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
                                      if source_frame == "FPU" else np.eye(3).tolist()),
              "scale_to_m": scale, "input_faces": raw_faces, "processed_faces": len(mesh.faces),
              "component_count": len(components), "watertight": True,
              "bounds_frd_m": bounds.tolist(), "dimensions_m": mesh.extents.tolist(),
              "component_volumes_m3": [float(x.volume) for x in components],
              "components": component_reports,
              "feature_scale_m": {"min_edge": float(edge_lengths.min()),
                                  "p10_edge": float(np.quantile(edge_lengths, .1)),
                                  "median_edge": float(np.median(edge_lengths))},
              "degenerate_face_cleanup": {"removed_count": removed_degenerate_faces,
                                           "retained_to_preserve_watertightness": retained_degenerate_for_closure},
              "triangle_density_faces_per_m2": float(len(mesh.faces) / mesh.area),
              "symmetry": "not_asserted_without_surface_reflection_test",
              "possible_component_intersections_by_bbox": possible_intersections,
              "warnings": ["Global triangle self-intersection testing is unavailable; mesh topology checks do not prove collision-free geometry"]}
    if mesh.extents[0] < mesh.extents[1]:
        report["warnings"].append("Longitudinal X extent is shorter than beam; verify source axes for this hull")
    canonical = np.asarray(mesh.vertices, "<f8").tobytes() + np.asarray(mesh.faces, "<i8").tobytes()
    return PreparedGeometry(mesh, source_hash, sha256(canonical).hexdigest(), report)
