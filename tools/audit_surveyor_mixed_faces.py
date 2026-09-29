"""Locate low pontoon-region faces hidden in large mixed STEP surface groups."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import gmsh


def run(step: Path, inventory: Path, output: Path) -> None:
    groups = {g["root"]: g for g in json.loads(inventory.read_text())}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("surveyor-mixed-face-audit")
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        result = {"schema": "surveyor-mixed-low-face-audit-v1", "groups": {}}
        for root in (13380, 13817):
            records = []
            for tag in groups[root]["surface_tags"]:
                bounds = gmsh.model.getBoundingBox(2, tag)
                x0, y0, z0, x1, y1, z1 = bounds
                near_port = x0 < -260 and x1 > -440
                near_starboard = x1 > 260 and x0 < 440
                if not (near_port or near_starboard) or y0 > 0 or y1 < -260 or z0 > 1750 or z1 < 300:
                    continue
                records.append({"tag": tag, "bounds_mm": list(bounds),
                                "area_mm2": gmsh.model.occ.getMass(2, tag),
                                "near_port": near_port, "near_starboard": near_starboard})
            result["groups"][str(root)] = {"source_face_count": len(groups[root]["surface_tags"]),
                                             "low_pontoon_region_face_count": len(records),
                                             "faces": records}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print({root: row["low_pontoon_region_face_count"] for root, row in result["groups"].items()})
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python tools/audit_surveyor_mixed_faces.py STEP GROUPS_JSON OUTPUT_JSON")
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
