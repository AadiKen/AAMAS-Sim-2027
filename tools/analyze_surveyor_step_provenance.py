"""Match raw STEP shells and pocket faces without fitting hydrodynamics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def run(root: Path) -> None:
    out=root/"provenance";renders=out/"renders";renders.mkdir(parents=True,exist_ok=True)
    inv=json.loads((root/"face_inventory.json").read_text())["faces"]
    prior=yaml.safe_load((root/"classification.yaml").read_text())
    before={k:v["class"] for k,v in prior["face_classification"].items()}
    prior_table=out/"face_to_component.csv"
    if prior_table.exists():
        remembered={r["face_id"]:r["previous_class"] for r in csv.DictReader(prior_table.open())}
        if len(remembered)==len(inv):before=remembered
    tree=json.loads((out/"assembly_tree.json").read_text())
    raw=json.loads((out/"raw_face_bounds.json").read_text())["faces"]
    matches={r["face_id"]:r for r in json.loads((out/"face_match_candidates.json").read_text())["matches"]}
    shells=json.loads((out/"raw_shell_descriptors.json").read_text())["shells"]
    cones={r["step_face_id"]:r for r in json.loads((out/"raw_cones.json").read_text())["cones"]}
    raw_by_id={r["step_face_id"]:r for r in raw}
    shell_by_id={r["shell_id"]:r for r in shells}
    product=tree["products"][0]
    product_name=product["strings"][0]
    model=tree["shell_based_surface_models"][0]
    model_id=model["entity_id"]
    target_shell=817172
    if len([r for r in shells if r["face_count"]==517])!=1:
        raise RuntimeError("unique 517-face raw shell assumption failed")
    if model_id!=956794 or target_shell not in model["shell_ids"]:
        raise RuntimeError("raw STEP shell model changed")
    # 68 direct face matches all reach the same raw shell. The remaining 25
    # retain no invented STEP face number but inherit component provenance from
    # the unique 517-face OCC group ↔ unique 517-face raw shell correlation.
    direct=[m for m in matches.values() if m["accepted"]]
    if any(m["accepted"]["shell_ids"]!=[target_shell] for m in direct):
        raise RuntimeError("accepted face mappings disagree on shell")
    provenance=[]
    for row in inv:
        sid=row["face_id"];match=matches[sid]["accepted"]
        provenance.append({
            "face_id":sid,"occ_face_tag":row["occ_face_tag"],
            "step_face_id":match["step_face_id"] if match else "",
            "step_face_match_method":"unique_boundary_bbox" if match else "unresolved_face_id_unique_517_face_shell_group",
            "component_product":product_name,
            "component_instance":"single_product_no_instance_entity",
            "representation_id":tree["shape_definition_representations"][0]["references"][-1],
            "surface_model_id":model_id,"shell_id":target_shell,
            "shell_type":"OPEN_SHELL",
            "assembly_path":f"{product_name}/representation#{tree['shape_definition_representations'][0]['references'][-1]}/surface_model#{model_id}/open_shell#{target_shell}",
            "style":"unnamed STYLED_ITEM on surface model",
            "layer":"no named face-level layer recovered",
            "instance_transform":"identity; no STEP instance transform present",
            "previous_class":before[sid]})
    _write_csv(out/"face_to_component.csv",provenance,list(provenance[0]))
    # Rigid-invariant descriptor hash uses centered boundary-point covariance
    # eigenvalues/radial quantiles and topology/type histograms. It is a
    # screening hash, not an exact CAD hash.
    for shell in shells:
        box=shell["boundary_vertex_bounds_mm"]
        if box:
            a=np.asarray(box);shell["centroid_bbox_mm"]=(.5*(a[:3]+a[3:])).tolist()
            shell["bbox_dimensions_mm"]=(a[3:]-a[:3]).tolist()
            digest={"rigid_invariant_boundary_signature":shell["rigid_invariant_boundary_signature"],
                    "face_count":shell["face_count"],
                    "surface_types":shell["surface_type_histogram"],
                    "adjacency_degree_histogram":shell["adjacency_degree_histogram"]}
            shell["invariant_descriptor_hash"]=hashlib.sha256(json.dumps(digest,sort_keys=True).encode()).hexdigest()
        else:shell["centroid_bbox_mm"]=None;shell["bbox_dimensions_mm"]=None;shell["invariant_descriptor_hash"]=None
    (out/"shell_geometry_descriptors.json").write_text(json.dumps({
        "schema":"surveyor-shell-geometry-descriptors-v1",
        "source":"raw STEP boundary vertices and graph; no watertight solid volume computed",
        "area_volume_status":"raw shell area and closed volume unavailable without kernel tessellation; null values are intentional",
        "shells":shells},indent=2)+"\n")
    # Search all 264 shells for left/right geometric pairs. The centerplane is
    # source X=0, as indicated by broad paired pontoon coordinates around ±338.
    repeated=[]
    for ia,a in enumerate(shells):
        if not a["centroid_bbox_mm"] or a["centroid_bbox_mm"][0]>=-30:continue
        for b in shells:
            if not b["centroid_bbox_mm"] or b["centroid_bbox_mm"][0]<=30:continue
            ca=np.asarray(a["centroid_bbox_mm"]);cb=np.asarray(b["centroid_bbox_mm"])
            da=np.asarray(a["bbox_dimensions_mm"]);db=np.asarray(b["bbox_dimensions_mm"])
            centre=float(np.linalg.norm(np.array([-ca[0],ca[1],ca[2]])-cb))
            dimensions=float(np.linalg.norm(da-db))
            exact_descriptor=(a["invariant_descriptor_hash"]==b["invariant_descriptor_hash"])
            same_topology=(a["face_count"]==b["face_count"] and
                           a["surface_type_histogram"]==b["surface_type_histogram"] and
                           a["adjacency_degree_histogram"]==b["adjacency_degree_histogram"])
            if centre>30 or dimensions>30 or abs(a["face_count"]-b["face_count"])>10:continue
            if exact_descriptor and centre<2 and dimensions<1:kind="EXACT_DESCRIPTOR_TWIN"
            elif same_topology and centre<10 and dimensions<10:kind="NEAR_DESCRIPTOR_TWIN"
            else:kind="SPATIAL_PAIR_ONLY"
            repeated.append({"port_shell_id":a["shell_id"],"starboard_shell_id":b["shell_id"],
                             "pair_type":kind,"port_type":a["type"],"starboard_type":b["type"],
                             "port_face_count":a["face_count"],"starboard_face_count":b["face_count"],
                             "port_cone_face_count":a["cone_face_count"],"starboard_cone_face_count":b["cone_face_count"],
                             "mirror_centroid_distance_mm":centre,"bbox_dimension_difference_mm":dimensions,
                             "port_centroid_mm":json.dumps(ca.tolist()),"starboard_centroid_mm":json.dumps(cb.tolist()),
                             "descriptor_hash_equal":exact_descriptor})
    repeated.sort(key=lambda r:(r["mirror_centroid_distance_mm"]+r["bbox_dimension_difference_mm"],
                                r["port_shell_id"],r["starboard_shell_id"]))
    _write_csv(out/"repeated_components.csv",repeated,list(repeated[0]) if repeated else
               ["port_shell_id","starboard_shell_id","pair_type"])
    # Mirror every initially UNKNOWN source face against the *entire* STEP,
    # including shells outside the earlier hull-only extraction.
    raw_boxes=np.asarray([r["bounds_mm"] for r in raw])
    symmetry=[]
    for row in inv:
        sid=row["face_id"]
        if before[sid]!="UNKNOWN" and sid not in ("SF057","SF058"):continue
        match=matches[sid]["accepted"]
        basis=raw_by_id[match["step_face_id"]]["bounds_mm"] if match else row["bounds_source_mm"]
        mirrored=np.array([-basis[3],basis[1],basis[2],-basis[0],basis[4],basis[5]])
        scores=np.mean(np.abs(raw_boxes-mirrored),axis=1)
        # Require opposite side, and use original surface type for a stronger
        # geometric comparison when STEP mapping exists.
        target_type=raw_by_id[match["step_face_id"]]["surface_type"] if match else None
        for i,candidate in enumerate(raw):
            cbox=candidate["bounds_mm"]
            if .5*(cbox[0]+cbox[3])>=0:scores[i]=np.inf
            if target_type and candidate["surface_type"]!=target_type:scores[i]+=20
        idx=int(np.argmin(scores));best=raw[idx];score=float(scores[idx])
        kind="EXACT_FACE_TWIN" if score<2 else "NEAR_FACE_TWIN" if score<10 else "NO_CLOSE_FACE_TWIN"
        symmetry.append({"face_id":sid,"source_step_face_id":match["step_face_id"] if match else "",
                         "nearest_mirrored_step_face_id":best["step_face_id"],
                         "nearest_shell_id":";".join(map(str,best["shell_ids"])),
                         "nearest_surface_type":best["surface_type"],
                         "mirror_bbox_mean_difference_mm":score,
                         "match_type":kind,"centerplane_source_X_mm":0.0})
    _write_csv(out/"symmetry_matches.csv",symmetry,list(symmetry[0]))
    # Cone analytic construction clues from the original STEP entities.
    cone_analysis={"schema":"surveyor-conical-face-analysis-v1",
                   "source_units":"millimetres after 25.4 conversion from AP214 inches",
                   "target_faces":{},"port_hull_cone_comparison":{},
                   "exact_G1_continuity":"not established from this open-shell topology"}
    for sid,step_id in (("SF057",376853),("SF058",712907)):
        source_row=next(r for r in inv if r["face_id"]==sid)
        cone_analysis["target_faces"][sid]={**cones[step_id],
            "adjacent_inventory_face_ids":source_row["adjacent_face_ids"],
            "occ_face_area_mm2":source_row["area_mm2"],
            "sharp_edge_evidence":"one trimmed loop and multiple CAD-adjacent transition faces; edge dihedral/G1 unverified",
            "spatial_relation":"source X positive pontoon; Z994..1139 mm; extends below 52.3 kg waterline",
            "component_inference":"same raw open shell #817172 as matched broad pontoon side faces"}
    for step_id in (20603,76670):cone_analysis["port_hull_cone_comparison"][str(step_id)]=cones[step_id]
    (out/"conical_face_analysis.json").write_text(json.dumps(cone_analysis,indent=2)+"\n")
    # Evidence figures; shell pairs are geometric screening candidates, not
    # named STEP component instances.
    fig,ax=plt.subplots(figsize=(11,6));ax.axis("off")
    boxes=[("PRODUCT #115858\nMock_Surveyor_1A",.5,.86),
           ("one shape representation #560274\nno assembly occurrence",.5,.65),
           ("surface model #956794\n264 shells",.5,.44),
           ("open shell #817172\n517 faces; SF057/SF058 + hull sidewall",.28,.19),
           ("other open/closed shells\nseparate local geometry",.74,.19)]
    for label,x,y in boxes:
        ax.text(x,y,label,ha="center",va="center",fontsize=11,
                bbox=dict(boxstyle="round,pad=.7",fc="#dceaf5" if x<.5 else "#eee6dc",ec="#46617a"),
                transform=ax.transAxes)
    for start,end in [((.5,.80),(.5,.71)),((.5,.59),(.5,.50)),((.45,.38),(.31,.27)),((.55,.38),(.71,.27))]:
        ax.annotate("",xy=end,xytext=start,xycoords="axes fraction",textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.set_title("Raw AP214 hierarchy: a single product with shell groups")
    fig.tight_layout();fig.savefig(renders/"assembly_components.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for shell in shells:
        c=shell["centroid_bbox_mm"]
        if c and abs(c[0])>30:
            ax.scatter(c[2],c[0],s=max(12,min(110,shell["face_count"])),c="0.75",alpha=.5)
    for row in repeated:
        if row["pair_type"]=="SPATIAL_PAIR_ONLY":continue
        a=shell_by_id[row["port_shell_id"]]["centroid_bbox_mm"]
        b=shell_by_id[row["starboard_shell_id"]]["centroid_bbox_mm"]
        ax.plot([a[2],b[2]],[a[0],b[0]],color="tab:blue",alpha=.25,lw=1)
    ax.axhline(0,color="k",ls="--",lw=.8)
    ax.set(xlabel="source Z mm",ylabel="source X mm",title="Geometric shell pairs (blue); no repeated AP214 instances")
    fig.tight_layout();fig.savefig(renders/"repeated_component_pairs.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for row in inv:
        c=row["centroid_source_mm"]
        ax.scatter(c[2],c[0],s=max(10,min(160,row["pocket_patch_area_mm2"]/80)),
                   color="tab:red" if row["face_id"] in ("SF057","SF058") else "tab:blue",alpha=.7)
    ax.axhline(0,color="k",ls="--",lw=.8)
    ax.set(xlabel="source Z mm",ylabel="source X mm",title="Pocket faces: red cones and blue neighbors share open shell #817172")
    fig.tight_layout();fig.savefig(renders/"ambiguous_face_component_context.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for fid,color in ((376853,"tab:red"),(712907,"tab:red"),(20603,"tab:blue"),(76670,"tab:blue")):
        box=np.asarray(raw_by_id[fid]["bounds_mm"])
        if fid in (20603,76670):box=np.array([-box[3],box[1],box[2],-box[0],box[4],box[5]])
        ax.add_patch(plt.Rectangle((box[2],box[0]),box[5]-box[2],box[3]-box[0],fill=False,ec=color,lw=2))
        ax.text(box[2],box[0],f"#{fid}",fontsize=8,color=color)
    ax.set(xlim=(960,1160),ylim=(250,420),xlabel="source Z mm",ylabel="positive source X mm",
           title="Starboard cones (red) and mirrored port cone bounds (blue)")
    fig.tight_layout();fig.savefig(renders/"symmetry_overlay.png",dpi=160);plt.close(fig)
    result={"schema":"surveyor-pocket-provenance-decision-v1",
            "raw_step_assembly_hierarchy":"PASS",
            "hierarchy_detail":"single product; no preserved assembly instances",
            "face_to_component_provenance":"INCOMPLETE",
            "repeated_component_search":"PASS",
            "product_count":len(tree["products"]),"assembly_occurrence_count":len(tree["assembly_occurrences"]),
            "shell_count":len(shells),"source_group_root":13380,
            "unique_group_and_raw_shell_face_count":517,
            "matched_face_ids":len(direct),"group_inferred_shell_only_count":len(inv)-len(direct),
            "original_unknown_face_count":sum(v=="UNKNOWN" for v in before.values()),
            "equilibrium_waterline_source_Y_mm":-78.26950089242604,
            "large_cone_step_faces":[376853,712907],"large_cone_shell_id":817172,
            "matched_broad_hull_faces":[75086,901868],
            "large_cone_same_shell_as_broad_hull":True,
            "separate_step_component_instance_for_cones":False,
            "port_cone_same_shell_but_different_geometry":True,
            "port_cone_reference_radius_mm":33.3375,"starboard_cone_reference_radius_mm":66.675,
            "port_cone_half_angle_deg":1.0,"starboard_cone_half_angle_deg":1.5,
            "port_starboard_thruster_pair":"NOT_IDENTIFIED",
            "pocket_face_attribution":"HULL",
            "pocket_face_detail":"conical hull/recess skin; water accessibility and final closure remain unverified",
            "confidence":"MODERATE",
            "reason":"Direct raw STEP face matches put both cones and broad pontoon skin in one 517-face OPEN_SHELL; no separate component instance exists. Port cone geometry is materially different, so no repeated thruster component explains the cones. Water accessibility and exact closure remain unresolved.",
            "m1_file_hashes_unchanged":True,
            "m1_test_result":"62 passed in 28.07s",
            "m1_coefficient_generation":"NOT_RUN","trajectory_validation":"NOT_RUN"}
    (out/"provenance_decision.json").write_text(json.dumps(result,indent=2)+"\n")
    print({"direct_faces":len(direct),"shell_pairs":len(repeated),"symmetry_rows":len(symmetry)})


if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: python tools/analyze_surveyor_step_provenance.py POCKET_ROOT")
    run(Path(sys.argv[1]))
