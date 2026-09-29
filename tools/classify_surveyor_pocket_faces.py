"""Conservative, auditable geometric classification of pocket face inventory."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import networkx as nx
import yaml


def run(root: Path) -> None:
    rows=json.loads((root/"face_inventory.json").read_text())["faces"]
    rows_by_id={r["face_id"]:r for r in rows}
    classes={}
    for row in rows:
        sid=row["face_id"]
        kind=row["surface_type"].lower()
        d=row["mirror_vertex_distance_p95_mm"]
        wet=row["pocket_patch_below_52_3kg_waterline_area_fraction"]
        area=row["pocket_patch_area_mm2"]
        if sid in ("SF057","SF058"):
            cls,confidence="UNKNOWN","low"
            evidence=["large conical patch forming the paired pocket feature",
                      "connected through original CAD edges to each other and multiple surrounding faces",
                      "50+ mm from mirrored port envelope; location and cone shape alone cannot tell wet recess wall from propulsion insert"]
        elif d<=5 and area>=1000:
            cls,confidence="HULL_SKIN","high" if wet>.1 else "medium"
            evidence=["broad B-spline or planar continuation within 5 mm of mirrored port envelope",
                      "shares original CAD edges with neighboring pontoon faces"]
        elif "cone" in kind or ("plane" in kind and d>5):
            cls,confidence="INTERFACE","low"
            evidence=["small manufactured conical or planar transition near pocket feature",
                      "local position alone does not establish whether wetted or internal"]
        elif d>10 and wet>.05:
            cls,confidence="UNKNOWN","low"
            evidence=["underwater source patch differs materially from mirrored port envelope",
                      "original face remains connected to mixed hull and pocket network"]
        elif d>10:
            cls,confidence="INTERFACE","low"
            evidence=["source patch differs materially from mirror but lies mainly above nominal waterline",
                      "could be a pocket rim, mounting transition, or hull detail"]
        elif d<=5 and area>=100:
            cls,confidence="HULL_SKIN","medium"
            evidence=["source face lies within 5 mm of mirrored port envelope",
                      "small patch requires neighboring topology for certainty"]
        else:
            cls,confidence="UNKNOWN","low"
            evidence=["small face with insufficient geometric evidence for a displaced-skin or hardware assignment"]
        classes[sid]={"class":cls,"confidence":confidence,
                      "occ_face_tag":row["occ_face_tag"],
                      "evidence":evidence,
                      "waterline_relation":"underwater_patch" if wet>.5 else "crosses_waterline" if wet>0 else "above_waterline"}
    output={"schema":"surveyor-pocket-face-classification-v1",
            "scope":"93 original faces with tessellated material in source X>0, Z900..1200 mm, Y<=0",
            "status":"AMBIGUOUS; classification is geometric evidence, not a complete wet-skin attribution",
            "external_fact":"published 2x 1 kW BLDC pocket thrusters; does not identify individual CAD faces",
            "face_classification":classes}
    (root/"classification.yaml").write_text(yaml.safe_dump(output,sort_keys=False))
    graph=nx.Graph();graph.add_nodes_from(classes)
    for row in rows:
        for neighbor in row["adjacent_face_ids"]:
            graph.add_edge(row["face_id"],neighbor)
    cone=("SF057","SF058")
    reduced=graph.copy();reduced.remove_nodes_from(cone)
    topology={"schema":"surveyor-pocket-candidate-adjacency-v1",
              "scope":"only inventoried faces; edges to faces outside window are not represented",
              "node_count":graph.number_of_nodes(),"edge_count":graph.number_of_edges(),
              "connected_component_sizes":sorted((len(c) for c in nx.connected_components(graph)),reverse=True),
              "after_removing_large_cone_faces":{
                  "removed_face_ids":list(cone),
                  "connected_component_sizes":sorted((len(c) for c in nx.connected_components(reduced)),reverse=True),
                  "candidate_shared_edge_seams_exposed":sum(len(rows_by_id[s]["adjacent_face_ids"]) for s in cone)-2,
                  "interpretation":"cone removal leaves the candidate graph connected but creates open seams; original assembly was already an open surface shell, so enclosed volume cannot be inferred"},
              "adjacency":{row["face_id"]:row["adjacent_face_ids"] for row in rows}}
    (root/"topology_graph.json").write_text(json.dumps(topology,indent=2)+"\n")
    from collections import Counter
    print(Counter(v["class"] for v in classes.values()))


if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: python tools/classify_surveyor_pocket_faces.py ROOT")
    run(Path(sys.argv[1]))
