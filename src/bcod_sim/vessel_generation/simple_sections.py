"""Deterministic sectional geometry and hydrostatic quadrature in body FRD."""
from __future__ import annotations

import numpy as np
import trimesh
from .volume_clip import submerged_volume_centroid


def plane_section(mesh: trimesh.Trimesh, axis: int, coordinate: float) -> list[np.ndarray]:
    normal = np.eye(3)[axis]
    path = mesh.section(plane_origin=normal * coordinate, plane_normal=normal)
    if path is None:
        return []
    return [np.asarray(line, dtype=float) for line in path.discrete if len(line) >= 3]


def polygon_properties(poly: np.ndarray, a: int, b: int) -> tuple[float, float, float, float, float]:
    p, q = poly[:, a], poly[:, b]
    if np.linalg.norm(poly[0] - poly[-1]) > 1e-8:
        p, q = np.r_[p, p[0]], np.r_[q, q[0]]
    cross = p[:-1] * q[1:] - p[1:] * q[:-1]
    signed = .5 * cross.sum()
    if abs(signed) < 1e-12:
        return (0., 0., 0., 0., 0.)
    cx = ((p[:-1] + p[1:]) * cross).sum() / (6 * signed)
    cy = ((q[:-1] + q[1:]) * cross).sum() / (6 * signed)
    ix = abs((((q[:-1] ** 2 + q[:-1] * q[1:] + q[1:] ** 2) * cross).sum() / 12) - signed * cy ** 2)
    iy = abs((((p[:-1] ** 2 + p[:-1] * p[1:] + p[1:] ** 2) * cross).sum() / 12) - signed * cx ** 2)
    return abs(signed), float(cx), float(cy), float(ix), float(iy)


def section_properties(mesh: trimesh.Trimesh, axis: int, coordinate: float) -> dict:
    other = tuple(i for i in range(3) if i != axis)
    loops = plane_section(mesh, axis, coordinate)
    parts = [polygon_properties(loop, *other) for loop in loops]
    area = sum(v[0] for v in parts)
    if area <= 0:
        return {"area": 0., "centroid": (0., 0.), "i_first": 0., "i_second": 0., "loops": loops}
    ca = sum(v[0] * v[1] for v in parts) / area
    cb = sum(v[0] * v[2] for v in parts) / area
    ia = sum(v[3] + v[0] * (v[2] - cb) ** 2 for v in parts)
    ib = sum(v[4] + v[0] * (v[1] - ca) ** 2 for v in parts)
    return {"area": float(area), "centroid": (float(ca), float(cb)),
            "i_first": float(ia), "i_second": float(ib), "loops": loops}


def hydrostatic_state(mesh: trimesh.Trimesh, waterline_z: float, *, density: float = 1025.,
                      gravity: float = 9.80665, samples: int = 81,
                      include_wetted: bool = True,
                      include_waterplane: bool = True) -> dict:
    """Clip triangles exactly below a FRD waterline (z increases downward)."""
    zmax = float(mesh.bounds[1, 2])
    if not mesh.bounds[0, 2] < waterline_z < zmax:
        raise ValueError("Waterline must cut the hull between its vertical bounds")
    volume, cb_array = submerged_volume_centroid(mesh, waterline_z)
    cb = cb_array.tolist()
    wp = section_properties(mesh, 2, waterline_z) if include_waterplane else None
    wetted = 0.
    if include_wetted:
        for triangle in mesh.triangles:
            poly = []
            points = list(triangle)
            for p, q in zip(points, points[1:] + points[:1]):
                in_p, in_q = p[2] >= waterline_z, q[2] >= waterline_z
                if in_p:
                    poly.append(p)
                if in_p != in_q:
                    poly.append(p + (waterline_z - p[2]) / (q[2] - p[2]) * (q - p))
            if len(poly) >= 3:
                for j in range(1, len(poly) - 1):
                    wetted += .5 * float(np.linalg.norm(np.cross(poly[j] - poly[0], poly[j+1] - poly[0])))
    return {"waterline_z_frd_m": float(waterline_z), "volume_m3": volume,
            "displacement_kg": volume * density, "center_buoyancy_frd_m": cb,
            "waterplane_area_m2": wp["area"] if wp else None,
            "waterplane_centroid_frd_m": [*wp["centroid"], waterline_z] if wp else None,
            "waterplane_i_roll_m4": wp["i_first"] if wp else None,
            "waterplane_i_pitch_m4": wp["i_second"] if wp else None,
            "wetted_area_m2": wetted if include_wetted else None,
            "integration_method": "exact_triangle_clip_v1", "integration_samples": 0}


def solve_waterline(mesh: trimesh.Trimesh, mass: float, density: float, *, samples: int = 81) -> dict:
    lo, hi = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    target = mass / density
    if target <= 0 or target >= mesh.volume:
        raise ValueError("Mass is outside displacement capacity")
    lo += 1e-6 * (hi - lo)
    hi -= 1e-6 * (hi - lo)
    for _ in range(20):
        mid = (lo + hi) / 2
        state = hydrostatic_state(mesh, mid, density=density, include_wetted=False,
                                  include_waterplane=False)
        # More submerged volume at smaller FRD z waterline.
        if state["volume_m3"] > target:
            lo = mid
        else:
            hi = mid
    return hydrostatic_state(mesh, (lo + hi) / 2, density=density, samples=samples)
