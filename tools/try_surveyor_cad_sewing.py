"""Conservative OCC healing sweep on copied pontoon faces; no mesh export."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import gmsh


GROUPS = {"port": (13555,), "starboard": (13451, 13463)}
TOLERANCES_MM = (0.01, 0.1, 1.0)


def _free_curve_count(tags: list[int]) -> int:
    counts = Counter()
    for tag in tags:
        counts.update(abs(edge) for dim, edge in gmsh.model.getBoundary([(2, tag)], oriented=False)
                      if dim == 1)
    return sum(count == 1 for count in counts.values())


def run(step: Path, groups_path: Path, output: Path) -> None:
    groups = {g["root"]: g for g in json.loads(groups_path.read_text())}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    result = {"schema": "surveyor-occ-heal-sweep-v1", "tolerance_unit": "mm", "attempts": []}
    try:
        gmsh.model.add("surveyor-sewing-sweep")
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        for side, roots in GROUPS.items():
            tags = [tag for root in roots for tag in groups[root]["surface_tags"]]
            before_free = _free_curve_count(tags)
            for tolerance in TOLERANCES_MM:
                record = {"side": side, "tolerance_mm": tolerance,
                          "before_face_count": len(tags), "before_free_curve_count": before_free}
                try:
                    copied = gmsh.model.occ.copy([(2, tag) for tag in tags])
                    healed = gmsh.model.occ.healShapes(copied, tolerance=tolerance,
                        fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
                        sewFaces=True, makeSolids=True)
                    gmsh.model.occ.synchronize()
                    surfaces = [tag for dim, tag in healed if dim == 2]
                    record.update({"result": "returned", "after_face_count": len(surfaces),
                                   "after_volume_count": sum(dim == 3 for dim, _ in healed),
                                   "after_free_curve_count": _free_curve_count(surfaces)})
                    gmsh.model.occ.remove(healed, recursive=True)
                    gmsh.model.occ.synchronize()
                except Exception as exc:
                    record.update({"result": "error", "error": str(exc)})
                result["attempts"].append(record)
                print(record, flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python tools/try_surveyor_cad_sewing.py STEP GROUPS_JSON OUTPUT_JSON")
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
