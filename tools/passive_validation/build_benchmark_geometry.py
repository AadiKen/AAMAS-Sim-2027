"""Build topology-preserving OBJ from official NMRI KVLCC2M surface grid."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/passive_hull_validation"
ZIP = CAMPAIGN / "raw_sources/kvlcc2m_dat.zip"
OUT = CAMPAIGN / "benchmarks/kvlcc2m/geometry/official_grid_underwater.obj"


def main() -> None:
    with zipfile.ZipFile(ZIP) as archive:
        lines = archive.read("kvlcc2m_whole.dat").decode().splitlines()
    if "I=401, J=101" not in lines[5]:
        raise ValueError("Unexpected official NMRI KVLCC2M grid dimensions")
    half = np.loadtxt(io.StringIO("\n".join(lines[7:]))).reshape(101, 401, 3)
    starboard = half * np.array([-4.970, -4.970, -4.970])
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
    for i in range(ni - 1):
        a, b = idx(0, nk - 1, i), idx(0, nk - 1, i + 1)
        c, d = idx(1, nk - 1, i + 1), idx(1, nk - 1, i)
        faces.extend(((a, b, c), (a, c, d)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices(digits_vertex=7)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    components = sorted(mesh.split(only_watertight=False), key=lambda part: len(part.faces), reverse=True)
    mesh = components[0]
    counts = np.bincount(mesh.edges_unique_inverse)
    boundary = mesh.edges_unique[counts == 1]
    stern = [edge for edge in boundary if
             np.all(np.abs(mesh.vertices[edge, 2]) < 1e-7) and
             np.all(np.abs(mesh.vertices[edge, 0] - mesh.vertices[edge[0], 0]) < 1e-7)]
    if stern:
        adjacency = {}
        for a, b in stern:
            adjacency.setdefault(int(a), []).append(int(b))
            adjacency.setdefault(int(b), []).append(int(a))
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise ValueError("Official grid stern boundary is not a simple loop")
        start = min(adjacency)
        polygon = [start, adjacency[start][0]]
        while polygon[-1] != start:
            polygon.append(next(v for v in adjacency[polygon[-1]] if v != polygon[-2]))
            if len(polygon) > len(adjacency) + 1:
                raise ValueError("Stern cap boundary did not close")
        cap = [(polygon[0], polygon[i], polygon[i + 1])
               for i in range(1, len(polygon) - 2)]
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=np.vstack((mesh.faces, cap)), process=True)
        mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Official-grid benchmark geometry failed closed-surface validation")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(OUT)
    report = {"source_archive_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
              "source_surface": "kvlcc2m_whole.dat", "format": "OBJ (preserves shared vertex topology)",
              "transform": "BCOD_FRD=(-x,-y,-z)_NMRI * 4.970 m", "closure": "flat waterline cap and stern lip",
              "watertight": mesh.is_watertight, "winding_consistent": mesh.is_winding_consistent,
              "bounds_frd_m": mesh.bounds.tolist(), "volume_m3": float(mesh.volume),
              "face_count": len(mesh.faces), "geometry_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
              "production_import_note": "OBJ retains vertex index topology; STL duplicates triangle vertices and fails the generator's strict merge tolerance."}
    (OUT.parent / "geometry_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
