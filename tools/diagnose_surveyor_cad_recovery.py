"""Audit CAD-kernel free edges on the selected Surveyor pontoon faces.

This is a read-only diagnostic of the STEP; it never constructs or exports a hull.
The root group IDs come from the surface adjacency inventory in the prior gate.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import gmsh


GROUPS = {"port": (13555,), "starboard": (13451, 13463)}


def run(step: Path, groups_path: Path, output: Path) -> None:
    groups = {g["root"]: g for g in json.loads(groups_path.read_text())}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("surveyor-cad-topology")
        imported = gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        result = {"schema": "surveyor-cad-boundary-audit-v1",
                  "source_sha256": sha256(step.read_bytes()).hexdigest(),
                  "imported_entity_counts": {str(dim): len(gmsh.model.getEntities(dim)) for dim in range(4)},
                  "classification": {}, "pontoons": {}}
        for side, roots in GROUPS.items():
            tags = [tag for root in roots for tag in groups[root]["surface_tags"]]
            available = {tag for _, tag in gmsh.model.getEntities(2)}
            missing = sorted(set(tags) - available)
            if missing:
                raise ValueError(f"{side}: {len(missing)} source face tags changed on import")
            result["classification"][side] = {"group_roots": list(roots), "surface_tags": tags,
                                                "decision": "retain_candidate_pontoon_exterior"}
            edges = Counter()
            for tag in tags:
                edges.update(abs(edge) for dim, edge in gmsh.model.getBoundary([(2, tag)], oriented=False)
                             if dim == 1)
            free = [tag for tag, count in edges.items() if count == 1]
            lengths = {str(tag): gmsh.model.occ.getMass(1, tag) for tag in free}
            boxes = {str(tag): list(gmsh.model.getBoundingBox(1, tag)) for tag in free}
            face_areas = [gmsh.model.occ.getMass(2, tag) for tag in tags]
            result["pontoons"][side] = {
                "cad_face_count": len(tags), "source_connected_surface_groups": len(roots),
                "cad_surface_area_mm2": sum(face_areas),
                "free_curve_count": len(free), "free_curve_total_length_mm": sum(lengths.values()),
                "free_curve_lengths_mm": lengths, "free_curve_bounding_boxes_mm": boxes,
                "source_group_bounds_mm": [groups[root]["bounds"] for root in roots],
                "status": "OPEN_CAD_SURFACES" if free else "NO_CAD_FREE_CURVES",
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({side: {k: v for k, v in record.items() if k in
              ("cad_face_count", "cad_surface_area_mm2", "free_curve_count",
               "free_curve_total_length_mm", "status")}
              for side, record in result["pontoons"].items()}, indent=2))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python tools/diagnose_surveyor_cad_recovery.py STEP GROUPS_JSON OUTPUT_JSON")
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
