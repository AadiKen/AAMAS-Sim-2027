"""Restore only the source-constrained ordinary bridge in Surveyor v2.

The v2 builder replaced FRD x=-625..-545 mm with a fair loft. That operation
removed valid source-supported detail. The earlier source-constrained B-rep is
restored through a 5 mm buffer on each side. The cut planes lie inside unchanged
source-constrained loft spans, so the original surface tangent crosses each
patch boundary. Everything outside remains v2 CAD.
This script is intentionally a local patch, not a reconstruction pipeline.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import cadquery as cq
import numpy as np
from OCP.BRepCheck import BRepCheck_Analyzer
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/cad'
HANDOFF = Path('/private/tmp/surveyor-v2-handoff/surveyor_surrogate_v2_fixed/cad')
OUT = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v2_targeted_patch'
LEFT_MM, RIGHT_MM = -630.0, -540.0


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def region_box() -> cq.Solid:
    return cq.Solid.makeBox(RIGHT_MM-LEFT_MM, 2000, 2000,
                            cq.Vector(LEFT_MM, -1000, -1000))


def replace_segment(old: cq.Solid, current: cq.Solid, box: cq.Solid) -> tuple[cq.Solid, dict]:
    outside = current.cut(box)
    inside = old.intersect(box)
    if len(outside.Solids()) != 2 or len(inside.Solids()) != 1:
        raise RuntimeError('Patch decomposition has unexpected topology')
    updated = outside.fuse(inside).clean()
    if len(updated.Solids()) != 1:
        raise RuntimeError('Patch did not fuse to one solid')
    solid = updated.Solids()[0]
    if not solid.isValid() or not BRepCheck_Analyzer(solid.wrapped).IsValid():
        raise RuntimeError('Patched OpenCascade solid is invalid')
    # Set-theoretic comparison outside the cut planes: both differences must
    # vanish. This also verifies unchanged source pocket and tip geometry.
    after_outside = solid.cut(box)
    outside_added = max(0., after_outside.cut(outside).Volume())
    outside_removed = max(0., outside.cut(after_outside).Volume())
    return solid, {'before_mm3': current.Volume(), 'after_mm3': solid.Volume(),
                   'outside_added_mm3': outside_added,
                   'outside_removed_mm3': outside_removed,
                   'inside_source_brep_mm3': inside.Volume(),
                   'brepcheck_pass': True}


def welded_tessellation(solid: cq.Solid, tol_mm: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = solid.tessellate(tol_mm, .20)
    coordinates = np.array([[p.x, p.y, p.z] for p in vertices], dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    quantized = np.round(coordinates / 1e-6).astype(np.int64)
    _, first, inverse = np.unique(quantized, axis=0, return_index=True,
                                  return_inverse=True)
    coordinates = coordinates[first]
    triangles = inverse[triangles]
    keep = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & (
        triangles[:, 2] != triangles[:, 0])
    triangles = triangles[keep]
    _, unique = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    triangles = triangles[np.sort(unique)]
    signed = np.einsum('ij,ij->i', coordinates[triangles[:, 0]],
                       np.cross(coordinates[triangles[:, 1]],
                                coordinates[triangles[:, 2]])).sum()/6
    if signed < 0:
        triangles = triangles[:, [0, 2, 1]]
    return coordinates/1000, triangles


def run() -> dict:
    (OUT / 'cad').mkdir(parents=True, exist_ok=True)
    (OUT / 'meshes').mkdir(parents=True, exist_ok=True)
    result = {'schema': 'surveyor-v2-targeted-source-bridge-v1',
              'patch_frd_x_mm': [LEFT_MM, RIGHT_MM],
              'original_step_face_tags_under_patch': [12991, 12992, 13028, 13119,
                                                       13141, 13145],
              'method': 'restore earlier source-constrained B-rep through 5 mm unchanged-source buffers on each side of v2 bridge',
              'variants': {}}
    box = region_box()
    for name in ('nominal', 'fuller', 'finer'):
        old_path = OLD / f'surveyor_{name}.step'
        current_path = HANDOFF / f'surveyor_{name}_v2.step'
        old = cq.importers.importStep(str(old_path)).val()
        current = cq.importers.importStep(str(current_path)).val()
        old_sides = sorted(old.Solids(), key=lambda s: s.Center().y, reverse=True)
        current_sides = sorted(current.Solids(), key=lambda s: s.Center().y, reverse=True)
        if len(old_sides) != 2 or len(current_sides) != 2:
            raise RuntimeError('Expected two pontoon solids in each STEP')
        solids, audits = [], []
        for original, now in zip(old_sides, current_sides):
            repaired, audit = replace_segment(original, now, box)
            solids.append(repaired)
            audits.append(audit)
        step_path = OUT / 'cad' / f'surveyor_{name}_v2_bridge_refit.step'
        cq.exporters.export(cq.Compound.makeCompound(solids), str(step_path))
        all_vertices, all_faces = [], []
        offset = 0
        for solid in solids:
            v, f = welded_tessellation(solid)
            all_vertices.append(v)
            all_faces.append(f+offset)
            offset += len(v)
        mesh = trimesh.Trimesh(vertices=np.vstack(all_vertices),
                               faces=np.vstack(all_faces), process=False)
        stl_path = OUT / 'meshes' / f'surveyor_{name}_v2_bridge_refit.stl'
        mesh.export(stl_path)
        loaded = trimesh.load_mesh(stl_path, process=True)
        row = {'old_source_constrained_step_sha256': digest(old_path),
               'input_v2_step_sha256': digest(current_path),
               'output_step_sha256': digest(step_path),
               'output_stl_sha256': digest(stl_path),
               'sides': audits, 'watertight': bool(loaded.is_watertight),
               'winding_consistent': bool(loaded.is_winding_consistent),
               'component_count': len(loaded.split(only_watertight=False)),
               'stl_volume_m3': float(loaded.volume),
               'step_volume_m3': sum(s.Volume() for s in solids)/1e9,
               'bounds_frd_m': loaded.bounds.tolist()}
        result['variants'][name] = row
        print(name, 'outside delta mm3',
              [(round(a['outside_added_mm3'], 6),
                round(a['outside_removed_mm3'], 6)) for a in audits],
              'watertight', row['watertight'], flush=True)
    (OUT / 'patch_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    run()
