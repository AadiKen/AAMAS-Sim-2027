"""Deterministic, preservation-gated hydrodynamic mesh reduction."""
from __future__ import annotations

from hashlib import sha256
import numpy as np
import trimesh

from .simple_sections import hydrostatic_state

DEFAULT_TOLERANCES = {"volume": .01, "cb": .01, "waterplane_area": .02,
                      "waterplane_moments": .03, "wetted_area": .03,
                      "dimensions": .005}


def _errors(candidate: dict, reference: dict, mesh: trimesh.Trimesh,
            original: trimesh.Trimesh) -> dict:
    length = max(float(original.extents.max()), 1e-9)
    relative = lambda a, b: abs(a - b) / max(abs(b), 1e-9)
    return {"volume": relative(candidate["volume_m3"], reference["volume_m3"]),
            "cb": float(np.linalg.norm(np.asarray(candidate["center_buoyancy_frd_m"]) -
                                        np.asarray(reference["center_buoyancy_frd_m"])) / length),
            "waterplane_area": relative(candidate["waterplane_area_m2"], reference["waterplane_area_m2"]),
            "waterplane_moments": max(relative(candidate[k], reference[k]) for k in
                                      ("waterplane_i_roll_m4", "waterplane_i_pitch_m4")),
            "wetted_area": relative(candidate["wetted_area_m2"], reference["wetted_area_m2"]),
            "dimensions": float(np.max(np.abs(mesh.extents - original.extents)) / length)}


def reduce_hydrodynamic_mesh(mesh: trimesh.Trimesh, waterline: float, reference: dict,
                             *, target_faces: int = 3000,
                             tolerances: dict | None = None) -> tuple[trimesh.Trimesh, dict]:
    limits = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    if len(mesh.faces) <= target_faces:
        return mesh, {"status": "original_below_target", "original_faces": len(mesh.faces),
                      "hydrodynamic_faces": len(mesh.faces), "errors": {k: 0. for k in limits},
                      "tolerances": limits}
    characteristic = float(mesh.extents.max())
    attempts = []
    try:
        import fast_simplification  # noqa: F401 - used by trimesh's decimator
    except ImportError:
        pass
    else:
        for faces in (target_faces, min(len(mesh.faces)-1, int(target_faces*1.5)),
                      min(len(mesh.faces)-1, target_faces*2)):
            if faces <= 0 or faces >= len(mesh.faces):
                continue
            try:
                candidate = mesh.simplify_quadric_decimation(face_count=faces)
                if not candidate.is_watertight or candidate.volume <= 0:
                    attempts.append({"method": "quadric_decimation", "target_faces": faces,
                                     "status": "topology_rejected"})
                    continue
                state = hydrostatic_state(candidate, waterline)
                errors = _errors(state, reference, candidate, mesh)
                accepted = all(errors[k] <= limits[k] for k in limits)
                attempts.append({"method": "quadric_decimation", "target_faces": faces,
                                 "faces": len(candidate.faces), "errors": errors,
                                 "status": "accepted" if accepted else "tolerance_rejected"})
                if accepted:
                    return candidate, {"status": "reduced", "method": "quadric_decimation",
                        "original_faces": len(mesh.faces), "hydrodynamic_faces": len(candidate.faces),
                        "errors": errors, "tolerances": limits, "attempts": attempts}
            except ValueError:
                attempts.append({"method": "quadric_decimation", "target_faces": faces,
                                 "status": "geometry_rejected"})
    components = mesh.split(only_watertight=True)
    if components and all(part.is_convex for part in components):
        for stride in (4, 3, 2):
            try:
                candidate = trimesh.util.concatenate([
                    trimesh.convex.convex_hull(part.vertices[::stride]) for part in components])
                if len(candidate.faces) >= len(mesh.faces):
                    continue
                state = hydrostatic_state(candidate, waterline)
                errors = _errors(state, reference, candidate, mesh)
                accepted = all(errors[k] <= limits[k] for k in limits)
                attempts.append({"method": "convex_component_subsample", "stride": stride,
                                 "faces": len(candidate.faces), "errors": errors,
                                 "status": "accepted" if accepted else "tolerance_rejected"})
                if accepted:
                    return candidate, {"status": "reduced", "method": "convex_component_subsample",
                        "original_faces": len(mesh.faces), "hydrodynamic_faces": len(candidate.faces),
                        "errors": errors, "tolerances": limits, "attempts": attempts}
            except ValueError:
                attempts.append({"method": "convex_component_subsample", "stride": stride,
                                 "status": "geometry_rejected"})
    for divisor in (80., 160., 320., 640.):
        spacing = characteristic / divisor
        candidate = mesh.copy()
        candidate.vertices = np.round(candidate.vertices / spacing) * spacing
        candidate.merge_vertices(digits_vertex=10)
        candidate.update_faces(candidate.nondegenerate_faces())
        candidate.update_faces(candidate.unique_faces())
        candidate.remove_unreferenced_vertices()
        if not candidate.is_watertight or candidate.volume <= 0:
            attempts.append({"grid_spacing_m": spacing, "status": "topology_rejected"})
            continue
        if len(candidate.faces) >= len(mesh.faces):
            attempts.append({"grid_spacing_m": spacing, "status": "no_reduction"})
            continue
        state = hydrostatic_state(candidate, waterline, samples=reference["integration_samples"])
        errors = _errors(state, reference, candidate, mesh)
        accepted = all(errors[k] <= limits[k] for k in limits)
        attempts.append({"grid_spacing_m": spacing, "faces": len(candidate.faces),
                         "errors": errors, "status": "accepted" if accepted else "tolerance_rejected"})
        if accepted:
            return candidate, {"status": "reduced", "method": "vertex_clustering", "original_faces": len(mesh.faces),
                "hydrodynamic_faces": len(candidate.faces), "errors": errors,
                "tolerances": limits, "attempts": attempts}
    return mesh, {"status": "original_retained_after_rejection", "original_faces": len(mesh.faces),
                  "hydrodynamic_faces": len(mesh.faces), "errors": {k: 0. for k in limits},
                  "tolerances": limits, "attempts": attempts}


def mesh_hash(mesh: trimesh.Trimesh) -> str:
    payload = np.asarray(mesh.vertices, "<f8").tobytes() + np.asarray(mesh.faces, "<i8").tobytes()
    return sha256(payload).hexdigest()
