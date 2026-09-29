"""Recover raw AP214 hierarchy and map conical CAD faces without kernel flattening.

The source has a single PRODUCT and a shell-based surface model. This parser
uses STEP entity references directly, then matches selected OCC faces to raw
ADVANCED_FACE geometry by source-coordinate boundary vertex bounds.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


START = re.compile(r"^#(\d+)\s*=\s*(.*)")
TYPE = re.compile(r"^([A-Z][A-Z_0-9]*)\s*\(")
REF = re.compile(r"#(\d+)")
STRING = re.compile(r"'((?:''|[^'])*)'")
NUMBER_TRIPLE = re.compile(r"\(\s*([-+\d.Ee]+)\s*,\s*([-+\d.Ee]+)\s*,\s*([-+\d.Ee]+)\s*\)")
KEYWORDS = re.compile(r"thruster|thrust|motor|prop|propeller|pocket|drive|pod|hull|pontoon|port|starboard|left|right", re.I)
META_TYPES = {"PRODUCT", "PRODUCT_DEFINITION", "PRODUCT_DEFINITION_SHAPE",
              "SHAPE_DEFINITION_REPRESENTATION", "SHAPE_REPRESENTATION",
              "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
              "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION", "MAPPED_ITEM",
              "STYLED_ITEM", "PRESENTATION_LAYER_ASSIGNMENT", "PRESENTATION_STYLE_ASSIGNMENT",
              "MANIFOLD_SURFACE_SHAPE_REPRESENTATION", "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION",
              "SHELL_BASED_SURFACE_MODEL", "SURFACE_SIDE_STYLE", "SURFACE_STYLE_USAGE"}


def entities(path: Path):
    current_id=None
    parts=[]
    with path.open(errors="replace") as stream:
        for line in stream:
            match=START.match(line)
            if match:
                if current_id is not None:
                    yield current_id,"".join(parts).split(";",1)[0].strip()
                current_id=int(match.group(1));parts=[match.group(2)]
            elif current_id is not None:
                parts.append(line)
            if current_id is not None and ";" in line:
                yield current_id,"".join(parts).split(";",1)[0].strip()
                current_id=None;parts=[]
    if current_id is not None:
        yield current_id,"".join(parts).strip()


def refs(text: str) -> tuple[int,...]:
    return tuple(map(int,REF.findall(text)))


def triplet(text: str) -> tuple[float,float,float] | None:
    matches=NUMBER_TRIPLE.findall(text)
    return tuple(map(float,matches[-1])) if matches else None


def run(step: Path, inventory_path: Path, out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    counts={}
    meta={}
    points={};directions={};vertices={};edges={};oriented={};loops={};bounds={}
    face_surfaces={};face_bounds={};shells={};shell_type={};cones={};placements={};surface_types={}
    strings=[]
    for eid,body in entities(step):
        m=TYPE.match(body)
        if not m:continue
        typ=m.group(1);counts[typ]=counts.get(typ,0)+1
        if typ in META_TYPES:
            meta[eid]={"type":typ,"references":refs(body),
                       "strings":[x.replace("''", "'") for x in STRING.findall(body)],
                       "raw":body[:1200]}
        if typ=="CARTESIAN_POINT":
            v=triplet(body)
            if v:points[eid]=v
        elif typ=="DIRECTION":
            v=triplet(body)
            if v:directions[eid]=v
        elif typ=="VERTEX_POINT":vertices[eid]=refs(body)[-1]
        elif typ=="EDGE_CURVE":edges[eid]=refs(body)[:2]
        elif typ=="ORIENTED_EDGE":oriented[eid]=refs(body)[-1]
        elif typ=="EDGE_LOOP":loops[eid]=refs(body)
        elif typ in ("FACE_BOUND","FACE_OUTER_BOUND"):bounds[eid]=refs(body)[0]
        elif typ=="ADVANCED_FACE":
            r=refs(body);face_surfaces[eid]=r[-1];face_bounds[eid]=r[:-1]
        elif typ in ("CLOSED_SHELL","OPEN_SHELL"):
            shells[eid]=refs(body);shell_type[eid]=typ
        elif typ=="CONICAL_SURFACE":
            rr=refs(body)
            nums=re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][-+]?\d+)?",body.split("#"+str(rr[0]),1)[-1])
            cones[eid]={"placement_id":rr[0],"radius_in":float(nums[0]) if nums else None,
                        "half_angle_rad":float(nums[1]) if len(nums)>1 else None}
        elif typ=="AXIS2_PLACEMENT_3D":placements[eid]=refs(body)
        elif typ in ("PLANE","CYLINDRICAL_SURFACE","SPHERICAL_SURFACE","TOROIDAL_SURFACE",
                     "B_SPLINE_SURFACE_WITH_KNOTS","B_SPLINE_SURFACE","SURFACE_OF_REVOLUTION"):
            surface_types[eid]=typ
        if typ in META_TYPES or KEYWORDS.search(body) and typ not in {"CARTESIAN_POINT","DIRECTION"}:
            for value in STRING.findall(body):
                value=value.replace("''", "'").strip()
                if value:strings.append((eid,typ,value,bool(KEYWORDS.search(value))))
    # Raw STEP product hierarchy. No kernel entity tags are used in this tree.
    representation_types={k for k in counts if "REPRESENTATION" in k}
    hierarchy={"schema":"surveyor-raw-step-assembly-v1",
               "source_step_sha256":hashlib.sha256(step.read_bytes()).hexdigest(),
               "entity_counts":{k:counts.get(k,0) for k in sorted(set(META_TYPES)|{"ADVANCED_FACE","CLOSED_SHELL","OPEN_SHELL","CONICAL_SURFACE"})},
               "representation_types":sorted(representation_types),
               "products":[{"entity_id":i,**v} for i,v in meta.items() if v["type"]=="PRODUCT"],
               "assembly_occurrences":[{"entity_id":i,**v} for i,v in meta.items() if v["type"]=="NEXT_ASSEMBLY_USAGE_OCCURRENCE"],
               "shape_definition_representations":[{"entity_id":i,**v} for i,v in meta.items() if v["type"]=="SHAPE_DEFINITION_REPRESENTATION"],
               "shell_based_surface_models":[{"entity_id":i,"shell_ids":list(v["references"]),"strings":v["strings"]}
                                             for i,v in meta.items() if v["type"]=="SHELL_BASED_SURFACE_MODEL"],
               "shells":[{"entity_id":i,"type":shell_type[i],"face_count":len(f),
                          "face_ids":list(f)} for i,f in shells.items()],
               "instance_transforms":[],"mapping_status":"single_product_no_ap214_assembly_instances_or_transforms"}
    product_nodes=[(i,v) for i,v in meta.items() if v["type"]=="PRODUCT"]
    definitions=[(i,v) for i,v in meta.items() if v["type"]=="PRODUCT_DEFINITION"]
    shape_links=[(i,v) for i,v in meta.items() if v["type"]=="SHAPE_DEFINITION_REPRESENTATION"]
    models=[(i,v) for i,v in meta.items() if v["type"]=="SHELL_BASED_SURFACE_MODEL"]
    if len(product_nodes)==len(definitions)==len(shape_links)==len(models)==1:
        sid,shape=shape_links[0]
        hierarchy["assembly_nodes"]=[{
            "product_id":product_nodes[0][0],"product_name":product_nodes[0][1]["strings"][0],
            "product_definition_id":definitions[0][0],
            "shape_definition_representation_id":sid,
            "shape_representation_id":shape["references"][-1],
            "surface_model_id":models[0][0],
            "instance_name":None,"instance_transform":None,
            "shell_ids":list(models[0][1]["references"]),
            "face_ids_by_shell":{str(k):list(v) for k,v in shells.items()}}]
    (out/"assembly_tree.json").write_text(json.dumps(hierarchy,indent=2)+"\n")
    with (out/"products.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["entity_id","product_name","product_description","all_strings","references"])
        for i,v in meta.items():
            if v["type"]=="PRODUCT":w.writerow([i,*((v["strings"]+["",""])[:2]),json.dumps(v["strings"]),json.dumps(v["references"])])
    with (out/"representations.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["entity_id","type","name","references"])
        for i,v in meta.items():
            if "REPRESENTATION" in v["type"] or v["type"]=="SHELL_BASED_SURFACE_MODEL":
                w.writerow([i,v["type"],v["strings"][0] if v["strings"] else "",json.dumps(v["references"])])
    (out/"metadata_strings.txt").write_text("\n".join(f"#{i} {typ} {'MATCH ' if hit else ''}{value}"
                                                  for i,typ,value,hit in strings)+"\n")
    # Inverse face->shell map. STEP may put a face in more than one shell, so
    # store all memberships rather than choosing one.
    face_to_shell={}
    for shell,faces in shells.items():
        for face in faces:face_to_shell.setdefault(face,[]).append(shell)
    @lru_cache(maxsize=20000)
    def face_vertex_coords(face: int) -> np.ndarray:
        coords=[]
        for bound in face_bounds.get(face,()):
            loop=bounds.get(bound)
            for orient in loops.get(loop,()):
                edge=oriented.get(orient)
                for vertex in edges.get(edge,()):
                    point=points.get(vertices.get(vertex))
                    if point:coords.append(point)
        return np.asarray(coords,dtype=float)*25.4 if coords else np.empty((0,3))
    cone_faces=[]
    raw_faces=[]
    for face,surface in face_surfaces.items():
        pts=face_vertex_coords(face)
        if len(pts):
            entry={"step_face_id":face,"surface_id":surface,
                   "surface_type":"CONICAL_SURFACE" if surface in cones else surface_types.get(surface,"UNKNOWN_COMPLEX"),
                   "bounds_mm":np.r_[pts.min(axis=0),pts.max(axis=0)].tolist(),
                   "shell_ids":face_to_shell.get(face,[]),
                   "boundary_vertex_count":len(pts)}
            raw_faces.append(entry)
            if surface in cones:cone_faces.append(entry)
    (out/"raw_face_bounds.json").write_text(json.dumps({
        "schema":"surveyor-raw-step-face-boundary-bounds-v1",
        "method":"min/max of STEP boundary vertex points in millimetres; curved extrema may extend beyond bounds",
        "faces":raw_faces},indent=2)+"\n")
    raw_cone_data=[]
    for row in cone_faces:
        cone=cones[row["surface_id"]]
        placement=placements.get(cone["placement_id"],())
        origin=np.asarray(points.get(placement[0]))*25.4 if len(placement)>0 and placement[0] in points else None
        axis=np.asarray(directions.get(placement[1])) if len(placement)>1 and placement[1] in directions else None
        if axis is not None:axis=axis/max(np.linalg.norm(axis),1e-12)
        pts=face_vertex_coords(row["step_face_id"])
        if origin is not None and axis is not None and len(pts):
            rel=pts-origin
            axial=rel@axis
            radial=np.linalg.norm(rel-axial[:,None]*axis,axis=1)
            apex=origin-axis*(cone["radius_in"]*25.4)/math.tan(cone["half_angle_rad"])
            radial_range=[float(radial.min()),float(radial.max())]
            axial_range=[float(axial.min()),float(axial.max())]
        else:apex=None;radial_range=None;axial_range=None
        raw_cone_data.append({"step_face_id":row["step_face_id"],
                              "surface_id":row["surface_id"],
                              "shell_ids":row["shell_ids"],
                              "placement_origin_mm":origin.tolist() if origin is not None else None,
                              "axis":axis.tolist() if axis is not None else None,
                              "reference_radius_mm":cone["radius_in"]*25.4,
                              "half_angle_deg":math.degrees(cone["half_angle_rad"]),
                              "apex_mm":apex.tolist() if apex is not None else None,
                              "boundary_radial_range_mm":radial_range,
                              "boundary_axial_range_mm":axial_range,
                              "boundary_bbox_mm":row["bounds_mm"],
                              "boundary_vertex_count":row["boundary_vertex_count"],
                              "trim_loop_count":len(face_bounds[row["step_face_id"]])})
    (out/"raw_cones.json").write_text(json.dumps({
        "schema":"surveyor-raw-step-cones-v1","cones":raw_cone_data},indent=2)+"\n")
    inventory=json.loads(inventory_path.read_text())["faces"]
    raw_boxes=np.asarray([r["bounds_mm"] for r in raw_faces])
    for row in inventory:
        target=np.asarray(row["bounds_source_mm"])
        scores=np.mean(np.abs(raw_boxes-target),axis=1)
        order=np.argsort(scores)[:5]
        row["step_face_match_candidates"]=[{
            "step_face_id":raw_faces[i]["step_face_id"],
            "shell_ids":raw_faces[i]["shell_ids"],
            "surface_type":raw_faces[i]["surface_type"],
            "mean_bbox_extrema_difference_mm":float(scores[i])} for i in order]
        best=row["step_face_match_candidates"][0]
        second=row["step_face_match_candidates"][1]
        row["step_face_match"]=(best if best["mean_bbox_extrema_difference_mm"]<5
                                 and second["mean_bbox_extrema_difference_mm"]-best["mean_bbox_extrema_difference_mm"]>2
                                 else None)
    (out/"face_match_candidates.json").write_text(json.dumps({
        "schema":"surveyor-occ-to-raw-step-face-candidates-v1",
        "method":"unique nearest six-extrema boundary-vertex bbox; <5 mm mean difference and >2 mm separation",
        "matches":[{"face_id":r["face_id"],"occ_face_tag":r["occ_face_tag"],
                    "accepted":r["step_face_match"],
                    "candidates":r["step_face_match_candidates"]} for r in inventory]},indent=2)+"\n")
    # Compare only conical candidates. A small corner/trim difference is okay,
    # but do not force a mapping when more than one STEP cone is plausible.
    for row in inventory:
        if row["surface_type"]!="Cone":continue
        target=np.array(row["bounds_source_mm"])
        ranked=[]
        for candidate in cone_faces:
            candidate_box=np.array(candidate["bounds_mm"])
            # Vertex extrema can under-bound curved surfaces. Mean absolute
            # extrema difference is a ranking metric, not a hard exact match.
            score=float(np.mean(np.abs(target-candidate_box)))
            ranked.append((score,candidate))
        ranked.sort(key=lambda x:x[0])
        row["step_cone_candidates"]=[{"step_face_id":v["step_face_id"],
                                      "shell_ids":v["shell_ids"],
                                      "mean_bbox_extrema_difference_mm":score}
                                     for score,v in ranked[:5]]
    (out/"cone_face_match_candidates.json").write_text(json.dumps({
        "schema":"surveyor-raw-step-cone-face-candidates-v1",
        "raw_conical_face_count":len(cone_faces),
        "candidate_method":"OCC face bounds ranked against raw STEP cone face boundary-vertex bounds; curved extrema may differ",
        "matches":[{"face_id":r["face_id"],"occ_face_tag":r["occ_face_tag"],
                    "candidates":r["step_cone_candidates"]} for r in inventory if r["surface_type"]=="Cone"]},indent=2)+"\n")
    # Save compact evidence needed for follow-on cluster and symmetry analysis.
    shell_descriptors=[]
    for sid,faces in shells.items():
        clouds=[p for face in faces if len(p:=face_vertex_coords(face))]
        if clouds:
            all_points=np.vstack(clouds)
            bbox=np.r_[all_points.min(axis=0),all_points.max(axis=0)].tolist()
            unique_points=np.unique(np.round(all_points,3),axis=0)
            centered=unique_points-unique_points.mean(axis=0)
            eigen=np.linalg.eigvalsh(centered.T@centered/max(len(centered),1))
            radii=np.linalg.norm(centered,axis=1)
            rigid_signature={"covariance_eigenvalues_mm2_0p1":np.round(eigen,1).tolist(),
                             "radial_quantiles_mm_0p1":np.round(np.percentile(radii,[0,10,25,50,75,90,100]),1).tolist(),
                             "unique_boundary_vertex_count":len(unique_points)}
        else:bbox=None;rigid_signature=None
        edge_to_faces={}
        for face in faces:
            for bound in face_bounds.get(face,()):
                for orient in loops.get(bounds.get(bound),()):
                    edge=oriented.get(orient)
                    if edge is not None:edge_to_faces.setdefault(edge,set()).add(face)
        neighbor={face:set() for face in faces}
        for touching in edge_to_faces.values():
            for face in touching:neighbor[face].update(touching-{face})
        degree_hist={}
        for nearby in neighbor.values():
            degree=len(nearby);degree_hist[str(degree)]=degree_hist.get(str(degree),0)+1
        type_hist={}
        for face in faces:
            surface=face_surfaces.get(face)
            kind="CONICAL_SURFACE" if surface in cones else surface_types.get(surface,"UNKNOWN_COMPLEX")
            type_hist[kind]=type_hist.get(kind,0)+1
        shell_descriptors.append({"shell_id":sid,"type":shell_type[sid],
                                  "face_count":len(faces),
                                  "cone_face_count":sum(face_surfaces.get(face) in cones for face in faces),
                                  "boundary_vertex_bounds_mm":bbox,
                                  "rigid_invariant_boundary_signature":rigid_signature,
                                  "surface_type_histogram":type_hist,
                                  "adjacency_degree_histogram":degree_hist,
                                  "shared_topological_edge_count":sum(len(v)>1 for v in edge_to_faces.values()),
                                  "surface_area_mm2":None,
                                  "volume_mm3":None})
    (out/"raw_shell_descriptors.json").write_text(json.dumps({
        "schema":"surveyor-raw-step-shell-descriptors-v1",
        "shells":shell_descriptors},indent=2)+"\n")
    print({"products":counts.get("PRODUCT",0),"assembly_occurrences":counts.get("NEXT_ASSEMBLY_USAGE_OCCURRENCE",0),
           "shells":len(shells),"conical_faces":len(cone_faces)})


if __name__=="__main__":
    if len(sys.argv)!=4:raise SystemExit("usage: python tools/parse_surveyor_step_provenance.py STEP INVENTORY_JSON OUTPUT_DIR")
    run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
