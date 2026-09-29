"""Tessellate mixed-group low faces as source constraints, without CAD healing."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import gmsh


def run(step: Path, audit: Path, output: Path) -> None:
    records = json.loads(audit.read_text())["groups"]["13380"]["faces"]
    tags = {row["tag"] for row in records}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("surveyor-mixed-low-faces")
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        faces = {tag for _, tag in gmsh.model.getEntities(2)}
        if not tags <= faces:
            raise ValueError("STEP face tags changed")
        gmsh.model.occ.remove([(2, tag) for tag in faces-tags], recursive=True)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 3.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 8.0)
        gmsh.model.mesh.generate(2)
        output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output))
        print(f"exported {len(tags)} mixed-group faces")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python tools/export_surveyor_mixed_low_faces.py STEP AUDIT_JSON OUTPUT_STL")
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
