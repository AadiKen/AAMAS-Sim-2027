"""Close official NMRI full-profile KCS grids above the design waterline.

Only the deck and transom are artificial meshing closures. All below-water
triangles come directly from the four official surface-grid panels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from inspect_kcs_full_grid import zones


def build(grid_dir: Path, output: Path) -> dict:
    parts = [*zones(grid_dir / "kcs_bow2.dat"), *zones(grid_dir / "kcs_stn2.dat")]
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    ids: dict[tuple[int, int], np.ndarray] = {}
    source_faces = 0

    for part_id, part in enumerate(parts):
        nj, ni = part.shape[:2]
        for side in (1, -1):
            p = part.copy() * np.array([-5.75, side * 5.75, -5.75])
            offset = len(vertices)
            vertices.extend(p.reshape(-1, 3))
            index = np.arange(offset, offset + nj * ni).reshape(nj, ni)
            ids[part_id, side] = index
            for j in range(nj - 1):
                for i in range(ni - 1):
                    a, b, c, d = index[j, i], index[j, i+1], index[j+1, i+1], index[j+1, i]
                    faces.extend(((a, b, c), (a, c, d)) if side == 1 else ((a, c, b), (a, d, c)))
            source_faces += 2 * (nj-1) * (ni-1)

    # Official top rows have a constant z above DLWL. Bridge port/starboard
    # corresponding source stations, without changing any source hull vertex.
    deck_faces = 0
    for part_id in (1, 3, 2):
        a, b = ids[part_id, 1][-1], ids[part_id, -1][-1]
        for i in range(len(a)-1):
            faces.extend(((a[i], b[i], b[i+1]), (a[i], b[i+1], a[i+1])))
            deck_faces += 2

    # The source overhang panel ends on a constant-x transom plane, entirely
    # above DLWL; bridge the matching starboard/port transom curves.
    a, b = ids[2, 1][:, -1], ids[2, -1][:, -1]
    transom_faces = 0
    for j in range(len(a)-1):
        faces.extend(((a[j], b[j], b[j+1]), (a[j], b[j+1], a[j+1])))
        transom_faces += 2

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices(digits_vertex=6)
    mesh.fix_normals()
    pieces = sorted(mesh.split(only_watertight=False), key=lambda p: len(p.faces), reverse=True)
    seam_fragments = [{"faces": len(p.faces), "area_m2": float(p.area)} for p in pieces[1:]]
    if any(p["faces"] > 2 or p["area_m2"] > 1e-4 for p in seam_fragments):
        raise ValueError("material disconnected panel or seam fragment")
    mesh = pieces[0]
    mesh.fix_normals()
    edge_count = np.bincount(mesh.edges_unique_inverse)
    open_edges = int(np.sum(edge_count == 1))
    nonmanifold_edges = int(np.sum(edge_count > 2))
    bbox = mesh.bounds
    # Divergence theorem with F=(0,0,z): the z=0 cutting plane contributes
    # zero. Clip each source triangle to z>=0 (FRD underwater) and integrate.
    submerged_volume = 0.0
    wetted_area = 0.0
    for tri in mesh.triangles:
        clipped = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            inside_a, inside_b = a[2] >= 0, b[2] >= 0
            if inside_a:
                clipped.append(a)
            if inside_a != inside_b:
                clipped.append(a + (b-a) * (-a[2] / (b[2]-a[2])))
        for i in range(1, len(clipped)-1):
            a, b, c = clipped[0], clipped[i], clipped[i+1]
            normal_area = np.cross(b-a, c-a) / 2
            submerged_volume += np.mean([a[2], b[2], c[2]]) * normal_area[2]
            wetted_area += np.linalg.norm(normal_area)
    report = {
        "source": "NMRI official 2005 KCS full-profile surface grids bow2/stn2",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (grid_dir / "kcs_bow2.dat", grid_dir / "kcs_stn2.dat")},
        "transform": "(x,y,z)_FRD = (-x,+y,-z)_NMRI * 5.75",
        "artificial_closures": ["top deck at source z=+0.018261 Lpp", "transom at source x=+0.52609 Lpp"],
        "source_hull_faces": source_faces,
        "deck_faces_before_welding": deck_faces,
        "transom_faces_before_welding": transom_faces,
        "discarded_zero_volume_seam_fragments": seam_fragments,
        "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "open_edges": open_edges, "nonmanifold_edges": nonmanifold_edges,
        "connected_components": len(mesh.split(only_watertight=False)),
        "bbox_frd_m": bbox.tolist(),
        "max_beam_m": float(bbox[1,1]-bbox[0,1]),
        "deck_height_above_water_m": float(-bbox[0,2]),
        "design_draft_m": float(bbox[1,2]),
        "submerged_volume_m3": float(abs(submerged_volume)),
        "wetted_surface_area_m2": float(wetted_area),
        "displacement_target_m3": 0.813,
        "displacement_relative_error": float((abs(submerged_volume)-0.813)/0.813),
        "cfd_ready_topology": bool(mesh.is_watertight and mesh.is_winding_consistent and open_edges==0 and nonmanifold_edges==0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)
    report["stl_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir / "kcs_hmri_full_hull.stl"
    report = build(args.grid_dir, output)
    (args.output_dir / "full_hull_geometry.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
