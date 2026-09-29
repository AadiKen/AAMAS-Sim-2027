"""Inventory original OCC faces near the starboard Surveyor propulsion pocket.

This is an evidence collector. OCC tags are import-session identifiers, not
STEP entity numbers; no unsupported STEP entity mapping is inferred.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import gmsh
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def run(step: Path, audit: Path, mirror: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads(audit.read_text())["groups"]["13380"]["faces"]
    # Do not include broad faces merely because their bounding boxes cross the
    # pocket. The later tessellated-patch filter localizes each face.
    candidate_tags = {r["tag"] for r in records if r["near_starboard"]
                      and r["bounds_mm"][2] < 1200 and r["bounds_mm"][5] > 900}
    source_bounds = {r["tag"]: r for r in records}
    # STL stores unshared triangle vertices; process=True is essential before
    # split, otherwise each triangle appears as a separate component.
    hull = trimesh.load_mesh(mirror, process=True)
    starboard_hull = next(p for p in hull.split(only_watertight=False)
                          if p.bounds[0, 1] < 0)
    mirror_xyz = np.column_stack((-starboard_hull.vertices[:, 1],
                                   -starboard_hull.vertices[:, 2],
                                   starboard_hull.vertices[:, 0]*1000+915))
    mirror_xyz[:, :2] *= 1000
    tree = cKDTree(mirror_xyz)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("surveyor-pocket-original-faces")
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        present = {t for _, t in gmsh.model.getEntities(2)}
        if not candidate_tags <= present:
            raise RuntimeError("OCC face tags changed on STEP import")
        boundary_by_face = {}
        face_by_edge = {}
        original_types, colors, centroids, areas = {}, {}, {}, {}
        for tag in sorted(candidate_tags):
            edges = {t for dim, t in gmsh.model.getBoundary([(2, tag)], oriented=False)
                     if dim == 1}
            boundary_by_face[tag] = edges
            for edge in edges:
                face_by_edge.setdefault(edge, []).append(tag)
            original_types[tag] = gmsh.model.getType(2, tag)
            colors[tag] = gmsh.model.getColor(2, tag)
            centroids[tag] = gmsh.model.occ.getCenterOfMass(2, tag)
            areas[tag] = gmsh.model.occ.getMass(2, tag)
        # Mesh only the candidate original faces, preserving face entity tags.
        gmsh.model.occ.remove([(2, t) for t in present-candidate_tags], recursive=True)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 3)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 8)
        gmsh.model.mesh.generate(2)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords).reshape(-1, 3)
        node_map = {int(tag): coords[i] for i, tag in enumerate(node_tags)}
        inventory = []
        meshes = {}
        for tag in sorted(candidate_tags):
            _, _, elem_nodes = gmsh.model.mesh.getElements(2, tag)
            elements = []
            for block in elem_nodes:
                flat = np.asarray(block, dtype=np.int64)
                if len(flat) % 3 == 0:
                    elements.extend(flat.reshape(-1, 3))
            if not elements:
                continue
            triangles = np.asarray(elements)
            used, remapped = np.unique(triangles, return_inverse=True)
            vertices = np.asarray([node_map[int(t)] for t in used])
            faces = remapped.reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            centres = mesh.triangles_center
            z = centres[:, 2]
            y = centres[:, 1]
            x = centres[:, 0]
            local = (x > 0) & (z >= 900) & (z <= 1200) & (y <= 0)
            if not np.any(local):
                continue
            local_pts = centres[local]
            local_area = mesh.area_faces[local]
            distances = tree.query(local_pts)[0]
            normals = mesh.face_normals[local]
            principal = np.average(normals, axis=0, weights=local_area)
            principal /= max(np.linalg.norm(principal), 1e-12)
            mean_normal_alignment = float(np.average(normals @ principal, weights=local_area))
            normal_angles = np.degrees(np.arccos(np.clip(normals @ principal, -1, 1)))
            below = -local_pts[:, 1]/1000 >= 0.07826950089242604
            adjacent = sorted({other for edge in boundary_by_face[tag]
                               for other in face_by_edge[edge] if other != tag})
            surface_id = f"SF{len(inventory)+1:03d}"
            inventory.append({
                "face_id": surface_id, "occ_face_tag": tag,
                "step_entity_id": "UNRESOLVED_OCC_IMPORT_MAP",
                "area_mm2": areas[tag], "pocket_patch_area_mm2": float(local_area.sum()),
                "centroid_source_mm": list(centroids[tag]),
                "bounds_source_mm": source_bounds[tag]["bounds_mm"],
                "surface_type": original_types[tag],
                "principal_normal_source_xyz": principal.tolist(),
                "normal_alignment_mean_cosine": mean_normal_alignment,
                "normal_angle_p95_deg": float(np.percentile(normal_angles, 95)),
                "normal_angle_max_deg": float(normal_angles.max()),
                "adjacent_occ_face_tags": adjacent,
                "source_connected_group_root": 13380,
                "mirror_vertex_distance_p50_mm": float(np.percentile(distances, 50)),
                "mirror_vertex_distance_p95_mm": float(np.percentile(distances, 95)),
                "mirror_vertex_distance_max_mm": float(distances.max()),
                "pocket_patch_below_52_3kg_waterline_area_fraction": float(local_area[below].sum()/local_area.sum()),
                "occ_import_rgba": list(colors[tag]),
                "ap214_layer": "UNRESOLVED",
                "pocket_patch_triangle_count": int(local.sum()),
            })
            meshes[surface_id] = (vertices, faces)
        tag_to_id = {r["occ_face_tag"]: r["face_id"] for r in inventory}
        for row in inventory:
            row["adjacent_face_ids"] = [tag_to_id[t] for t in row.pop("adjacent_occ_face_tags") if t in tag_to_id]
        with (output/"face_inventory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        (output/"face_inventory.json").write_text(json.dumps({
            "schema": "surveyor-pocket-original-face-inventory-v1",
            "source_step_sha256": hashlib.sha256(step.read_bytes()).hexdigest(),
            "distance_method": "triangle centroid to nearest nominal mirrored-hull vertex; approximate and one-sided",
            "waterline_z_frd_m": 0.07826950089242604,
            "equilibrium_draft_m": 0.1690547274,
            "face_count": len(inventory), "faces": inventory}, indent=2)+"\n")
        np.savez_compressed(output/"face_triangles.npz", **{
            f"{face_id}_{kind}": arr for face_id, pair in meshes.items()
            for kind, arr in zip(("vertices", "triangles"), pair)})
        print(f"inventoried {len(inventory)} original faces; {sum(r['pocket_patch_area_mm2'] for r in inventory)/1e6:.3f} m2 pocket-patch area")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: python tools/attribute_surveyor_pocket_faces.py STEP AUDIT_JSON MIRRORED_STL OUTPUT_DIR")
    run(*(Path(arg) for arg in sys.argv[1:]))
