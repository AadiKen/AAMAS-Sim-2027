"""Tessellate selected low STEP surface groups for section constraints only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import gmsh


GROUPS = {
    "negative_x_pontoon": (1669, 1963, 609, 13555),
    "positive_x_pontoon": (1197, 1441, 297, 13451, 13463),
}


def run(step: Path, groups_path: Path, output: Path) -> None:
    inventory = {g["root"]: g for g in json.loads(groups_path.read_text())}
    output.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("surveyor-section-sources")
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        chosen = {tag for roots in GROUPS.values() for root in roots
                  for tag in inventory[root]["surface_tags"]}
        available = {tag for _, tag in gmsh.model.getEntities(2)}
        if not chosen <= available:
            raise ValueError("source face tags changed during STEP import")
        gmsh.model.occ.remove([(2, tag) for tag in available-chosen], recursive=True)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 3.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 7.0)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output / "selected_low_surfaces_source_mm.stl"))
        (output / "group_selection.json").write_text(json.dumps({
            "schema": "surveyor-section-source-selection-v1", "groups": GROUPS,
            "face_count": len(chosen), "status": "source_face_tessellation_not_closed_hull"
        }, indent=2) + "\n")
        print("exported", len(chosen), "CAD faces", flush=True)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python tools/export_surveyor_section_sources.py STEP GROUPS_JSON OUTPUT_DIR")
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
