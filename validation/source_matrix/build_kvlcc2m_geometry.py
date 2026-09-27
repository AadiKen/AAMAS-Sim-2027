"""Convert NMRI's KVLCC2M half-hull surface grid to a closed Spec A hull.

The only added surface is the flat design-waterline cap. Source points are
scaled by the published 4.970 m Lpp; no fairing or hull-shape modification.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "kvlcc2m_dat.zip"
OUTPUT = ROOT / "kvlcc2m_underwater.stl"
LPP = 4.970


def build() -> dict:
    with zipfile.ZipFile(SOURCE) as archive:
        lines = archive.read("kvlcc2m_whole.dat").decode().splitlines()
    if "I=401, J=101" not in lines[5]:
        raise ValueError("Unexpected official KVLCC2M grid dimensions")
    half = np.loadtxt(io.StringIO("\n".join(lines[7:]))).reshape(101, 401, 3)
    # NMRI grid: x aft, y port (negative), z upward, normalized by Lpp.
    # BCOD input: x forward, y starboard, z downward.
    starboard = half * np.array([-LPP, -LPP, -LPP])
    starboard[-1, :, 2] = 0.
    port = starboard.copy()
    port[:, :, 1] *= -1
    nk, ni = starboard.shape[:2]
    vertices = np.concatenate((starboard.reshape(-1, 3), port.reshape(-1, 3)))

    def idx(side: int, k: int, i: int) -> int:
        return side * nk * ni + k * ni + i

    faces = []
    for side in (0, 1):
        for k in range(nk - 1):
            for i in range(ni - 1):
                a, b, c, d = (idx(side, k, i), idx(side, k, i + 1),
                              idx(side, k + 1, i + 1), idx(side, k + 1, i))
                faces.extend(((a, b, c), (a, c, d)) if side == 0 else
                             ((a, c, b), (a, d, c)))
    source_face_count = len(faces)
    for i in range(ni - 1):
        a, b = idx(0, nk - 1, i), idx(0, nk - 1, i + 1)
        c, d = idx(1, nk - 1, i + 1), idx(1, nk - 1, i)
        faces.extend(((a, b, c), (a, c, d)))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices(digits_vertex=7)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    components = sorted(mesh.split(only_watertight=False), key=lambda p: len(p.faces), reverse=True)
    removed_seam_fragments = [dict(faces=len(p.faces), area_m2=float(p.area)) for p in components[1:]]
    if any(item["area_m2"] > 1e-5 for item in removed_seam_fragments):
        raise ValueError(f"Material disconnected surface: {removed_seam_fragments}")
    mesh = components[0]
    # The source grid stops at the flat stern lip. Close its small waterplane
    # polygon using only points already present in the official grid.
    counts = np.bincount(mesh.edges_unique_inverse)
    open_edges = mesh.edges_unique[counts == 1]
    stern_edges = [edge for edge in open_edges if
                   np.all(np.abs(mesh.vertices[edge, 2]) < 1e-7) and
                   np.all(np.abs(mesh.vertices[edge, 0] - mesh.vertices[edge[0], 0]) < 1e-7)]
    if stern_edges:
        adjacency = {}
        for a, b in stern_edges:
            adjacency.setdefault(int(a), []).append(int(b))
            adjacency.setdefault(int(b), []).append(int(a))
        if any(len(v) != 2 for v in adjacency.values()):
            raise ValueError("Stern lip does not form a simple closed polygon")
        start = min(adjacency)
        polygon = [start, adjacency[start][0]]
        while polygon[-1] != start:
            options = [v for v in adjacency[polygon[-1]] if v != polygon[-2]]
            polygon.append(options[0])
            if len(polygon) > len(adjacency) + 1:
                raise ValueError("Stern lip polygon failed to close")
        patch_faces = [(polygon[0], polygon[i], polygon[i+1])
                       for i in range(1, len(polygon)-2)]
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=np.vstack((mesh.faces, patch_faces)), process=True)
        mesh.fix_normals()
    counts = np.bincount(mesh.edges_unique_inverse)
    report = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_surface": "kvlcc2m_whole.dat",
        "source_face_count": source_face_count,
        "closure": "flat cap along official design-waterline coordinates",
        "transform": "(x,y,z)_BCOD_FRD = (-x,-y,-z)_NMRI * 4.970 m",
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "open_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "connected_components": len(mesh.split(only_watertight=False)),
        "bounds_frd_m": mesh.bounds.tolist(),
        "volume_m3": float(mesh.volume),
        "area_m2": float(mesh.area),
        "removed_zero_area_seam_fragments": removed_seam_fragments,
        "component_face_counts": [len(p.faces) for p in mesh.split(only_watertight=False)],
        "open_edge_endpoints_m": mesh.vertices[mesh.edges_unique[counts == 1]].tolist()[:30],
        "nonmanifold_edge_endpoints_m": mesh.vertices[mesh.edges_unique[counts > 2]].tolist()[:30],
    }
    if not mesh.is_watertight or report["nonmanifold_edges"]:
        raise ValueError(f"Reconstructed hull fails topology: {report}")
    mesh.export(OUTPUT)
    report["stl_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    (ROOT / "kvlcc2m_geometry_qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
