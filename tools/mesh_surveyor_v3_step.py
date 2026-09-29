"""Create conformal metre-unit STL from the v3 STEP solids with Gmsh/OCC."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import gmsh
import trimesh

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def mesh_one(name: str) -> dict:
    step = BASE / 'cad' / f'surveyor_{name}_v3_aft_repair.step'
    mesh_dir = BASE / 'meshes'
    mesh_dir.mkdir(exist_ok=True)
    temporary = mesh_dir / f'surveyor_{name}_v3_aft_repair_mm.stl'
    output = mesh_dir / f'surveyor_{name}_v3_aft_repair.stl'
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add(f'surveyor_{name}_v3')
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        if len(gmsh.model.getEntities(3)) != 2:
            raise RuntimeError('Expected exactly two STEP pontoon solids')
        gmsh.option.setNumber('Mesh.MeshSizeMin', 1.)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 4.)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(temporary))
    finally:
        gmsh.finalize()
    mesh = trimesh.load_mesh(temporary, process=True)
    if not mesh.is_watertight:
        raise RuntimeError(f'{name}: Gmsh STL is not watertight')
    mesh.fix_normals()
    if not mesh.is_winding_consistent or mesh.volume <= 0:
        raise RuntimeError(f'{name}: mesh normals or volume invalid')
    if len(mesh.split(only_watertight=False)) != 2:
        raise RuntimeError(f'{name}: mesh does not have exactly two components')
    mesh.apply_scale(.001)
    mesh.export(output)
    temporary.unlink()
    return {'step_sha256': digest(step), 'stl_sha256': digest(output),
            'stl_path': str(output), 'face_count': len(mesh.faces),
            'watertight': bool(mesh.is_watertight),
            'winding_consistent': bool(mesh.is_winding_consistent),
            'component_count': 2, 'volume_m3': float(mesh.volume),
            'center_of_volume_frd_m': mesh.center_mass.tolist(),
            'bounds_frd_m': mesh.bounds.tolist()}


def run() -> None:
    results = {}
    for name in ('nominal', 'fuller', 'finer'):
        results[name] = mesh_one(name)
        print(name, results[name]['volume_m3'], flush=True)
    (BASE / 'mesh_manifest.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    run()
