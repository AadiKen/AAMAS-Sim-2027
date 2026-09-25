#!/usr/bin/env python3
"""Fail-closed Stage 3 geometry -> CFD -> frozen vessel validation gate."""
from __future__ import annotations
import argparse,hashlib,json,math,platform,shutil,subprocess,sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_PROVENANCE={"mass_kg","cg_frd_m","inertia_cg_kg_m2","added_mass_kg","linear_damping","quadratic_damping","hydrostatics","geometry","thrusters"}
ALLOWED_PROVENANCE={"KNOWN_PHYSICAL","GEOMETRY_DERIVED","OPENFOAM_DERIVED","ASSUMED"}
FORBIDDEN_MARKERS=("mss damping","mss added","mss hydrostatic","otter_oracle","mss-derived","coefficient-matched")
STAGE1_SCENARIOS=("equilibrium","surge","sway","heave","roll","pitch","yaw","combined_6dof","initial_condition_decay","straight_thrust","left_turn","right_turn","differential_thrust","combined_maneuver")

def sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(value)->str: return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def cache_key(*,geometry_hash:str,mesh_settings:dict,solver_settings:dict,openfoam_identity:str)->str:
    return canonical_hash({"geometry":geometry_hash,"mesh":mesh_settings,"solver":solver_settings,"openfoam":openfoam_identity})
def cache_valid(manifest:dict,expected_key:str)->bool: return manifest.get("cache_key")==expected_key and manifest.get("complete") is True
def leakage_guard(payload)->None:
    text=json.dumps(payload,sort_keys=True).lower()
    found=[marker for marker in FORBIDDEN_MARKERS if marker in text]
    if found: raise ValueError(f"MSS leakage marker(s): {', '.join(found)}")
def validate_provenance(provenance:dict)->None:
    missing=REQUIRED_PROVENANCE-set(provenance)
    if missing: raise ValueError(f"missing provenance: {sorted(missing)}")
    invalid={key:value.get("kind") for key,value in provenance.items() if value.get("kind") not in ALLOWED_PROVENANCE}
    if invalid: raise ValueError(f"invalid provenance: {invalid}")
    leakage_guard(provenance)
def freeze_config(config:dict,provenance:dict)->str:
    validate_provenance(provenance); leakage_guard(config); return canonical_hash({"config":config,"provenance":provenance})
def comparison_metrics(reference:np.ndarray,candidate:np.ndarray)->dict:
    if reference.shape!=candidate.shape or reference.ndim!=2 or reference.shape[1]!=12: raise ValueError("comparison requires matching [N,12] states")
    delta=candidate-reference
    return {"state_rmse":float(np.sqrt(np.mean(delta**2))),"position_rmse":float(np.sqrt(np.mean(delta[:,:3]**2))),"orientation_rmse":float(np.sqrt(np.mean(delta[:,3:6]**2))),"velocity_rate_rmse":float(np.sqrt(np.mean(delta[:,6:]**2))),"terminal_max_abs":float(np.max(np.abs(delta[-1])))}
def git(*args):
    try:return subprocess.check_output(("git",*args),cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "unavailable"
def docker_identity()->dict:
    image="openfoam/openfoam11-paraview510:11"
    try:
        value=subprocess.check_output(("docker","image","inspect",image,"--format","{{index .RepoDigests 0}} {{.Id}}"),text=True,stderr=subprocess.DEVNULL).strip()
        return {"available":True,"image":image,"identity":value}
    except Exception:return {"available":False,"image":image,"identity":None}
def capabilities()->dict:
    # These are production adapter capabilities, deliberately not inferred from old artifacts.
    return {"steady_positive_surge":True,"six_axis_signed_sweeps":False,"mesh_three_level_campaign":False,"transient_timestep_sweep":False,"forced_motion_added_mass":False,"geometry_hydrostatics":False,"coupled_coefficient_fit":False}

def _parse_obj(path:Path):
    vertices=[]; groups={}; group=None
    with path.open(errors="replace") as stream:
        for line in stream:
            if line.startswith("v "): vertices.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("g "): group=line.split(maxsplit=1)[1].strip(); groups.setdefault(group,[])
            elif line.startswith("f ") and group:
                indices=tuple(int(token.split("/")[0])-1 for token in line.split()[1:])
                if len(indices)==3: groups[group].append(indices)
    return vertices,groups

def _mesh_stats(vertices,faces)->dict:
    used=sorted({index for face in faces for index in face}); points=[vertices[index] for index in used]
    low=[min(point[axis] for point in points) for axis in range(3)]; high=[max(point[axis] for point in points) for axis in range(3)]
    area=signed_volume=0.0; welded={}; welded_faces=[]
    for face in faces:
        welded_face=[]
        for index in face:
            key=tuple(round(value,7) for value in vertices[index])
            if key not in welded: welded[key]=len(welded)
            welded_face.append(welded[key])
        welded_faces.append(tuple(welded_face)); a,b,c=(vertices[index] for index in face)
        u=tuple(b[i]-a[i] for i in range(3)); v=tuple(c[i]-a[i] for i in range(3))
        cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
        area += 0.5*math.sqrt(sum(value*value for value in cross))
        signed_volume += (a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6.0
    edges=Counter(tuple(sorted(edge)) for a,b,c in welded_faces for edge in ((a,b),(b,c),(c,a)))
    return {"faces":len(faces),"source_vertices":len(used),"welded_vertices":len(welded),"bounds_min_m":low,"bounds_max_m":high,
            "extents_m":[high[i]-low[i] for i in range(3)],"center_m":[(high[i]+low[i])/2 for i in range(3)],"surface_area_m2":area,
            "signed_volume_m3_unreliable_open_surface":signed_volume,"boundary_edges":sum(value==1 for value in edges.values()),
            "nonmanifold_edges":sum(value>2 for value in edges.values()),"degenerate_faces_after_weld":sum(len(set(face))<3 for face in welded_faces),
            "watertight":bool(edges) and all(value==2 for value in edges.values())}

def qualify_collada(geometry:Path,artifact_dir:Path)->dict:
    original_dir=artifact_dir/"original"; original_dir.mkdir(parents=True,exist_ok=True)
    preserved=original_dir/geometry.name; shutil.copy2(geometry,preserved)
    text=geometry.read_text(errors="replace")[:20000]
    import re
    unit_match=re.search(r'<unit[^>]*meter="([^"]+)"',text); up_match=re.search(r"<up_axis>([^<]+)</up_axis>",text)
    unit_meter=float(unit_match.group(1)) if unit_match else None
    obj=artifact_dir/"full_visual_scene.obj"
    try: subprocess.run(["assimp","export",str(geometry),str(obj),"-f","objnomtl"],cwd=ROOT,check=True,capture_output=True,text=True)
    except (FileNotFoundError,subprocess.CalledProcessError) as exc:
        return {"status":"BLOCKED_GEOMETRY","source":{"path":str(geometry.resolve()),"sha256":sha256(geometry),"unit_meter":unit_meter,"up_axis":up_match.group(1) if up_match else None},"blockers":[f"Reproducible COLLADA conversion failed: {exc}"]}
    vertices,groups=_parse_obj(obj); stats={name:_mesh_stats(vertices,faces) for name,faces in groups.items() if faces}
    ranked=sorted(stats,key=lambda name:(stats[name]["faces"],max(stats[name]["extents_m"])),reverse=True)
    hulls=[{"component":name,**stats[name]} for name in ranked[:2]]
    whole=_mesh_stats(vertices,[face for faces in groups.values() for face in faces])
    spacing=abs(hulls[0]["center_m"][0]-hulls[1]["center_m"][0]) if len(hulls)==2 else None
    blockers=[]
    if len(groups)>20: blockers.append(f"Asset is an assembled visual scene with {len(groups)} disconnected named meshes, not an isolated CFD hull surface.")
    if len(hulls)!=2 or any(not hull["watertight"] for hull in hulls): blockers.append("The two largest mirrored hull-like components are open surfaces and fail the watertight topology gate.")
    if hulls: blockers.append(f"Closing {sum(h['boundary_edges'] for h in hulls)} detected hull boundary edges would be substantive geometry reconstruction, not reproducible minimal repair.")
    return {"status":"BLOCKED_GEOMETRY" if blockers else "PASS_VALIDATED_RECONSTRUCTION","source":{"path":str(geometry.resolve()),"preserved_copy":str(preserved),"sha256":sha256(geometry),"unit_meter":unit_meter,"up_axis":up_match.group(1) if up_match else None},"scene":{"component_count":len(groups),"triangle_count":sum(len(faces) for faces in groups.values()),"overall":whole},"hull_candidate_selection":"two components with greatest triangle count; mirrored centers and ~2 m longitudinal extent corroborate classification","hull_candidates":hulls,"candidate_center_spacing_m":spacing,"qualified_dimensions":None,"hydrostatic_sweep":"NOT_RUN_INVALID_OPEN_SURFACE","repairs":[],"blockers":blockers}

def run(output:Path,geometry:Path|None,source_url:str|None=None,source_commit:str|None=None)->dict:
    folders=("geometry","mesh_convergence","cfd","coefficient_fits","direct_wrench","actuator","coefficient_comparison","sensitivity","plots")
    output.mkdir(parents=True,exist_ok=True)
    for folder in folders:(output/folder).mkdir(exist_ok=True)
    runtime=docker_identity(); caps=capabilities(); failures=[]; qualification=None
    if geometry is None: failures.append({"category":"GEOMETRY","reason":"No trustworthy Otter CAD/geometry was supplied. Synthetic smoke hull and legacy approximate proxy are not admissible."})
    elif not geometry.is_file(): failures.append({"category":"GEOMETRY","reason":f"Geometry does not exist: {geometry}"})
    elif geometry.suffix.lower()==".dae":
        qualification=qualify_collada(geometry,output/"geometry"); qualification["source"].update({"url":source_url,"repository_commit":source_commit})
        (output/"geometry"/"geometry_summary.json").write_text(json.dumps(qualification,indent=2)+"\n")
        failures.extend({"category":"GEOMETRY","reason":reason} for reason in qualification["blockers"])
    elif geometry.suffix.lower() not in {".stl",".obj",".step",".stp"}: failures.append({"category":"GEOMETRY","reason":"Unsupported geometry format"})
    geometry_blocked=any(item["category"]=="GEOMETRY" for item in failures)
    if not geometry_blocked:
        for name,supported in caps.items():
            if not supported: failures.append({"category":"CFD_NUMERICS" if "mesh" in name or "timestep" in name else ("ADDED_MASS" if "added_mass" in name else "MODEL_FORM"),"reason":f"Production OpenFOAM pipeline capability missing: {name}"})
        if not runtime["available"]: failures.append({"category":"CFD_NUMERICS","reason":"Pinned OpenFOAM Docker image is unavailable"})
    manifest={"stage":"3","commit":git("rev-parse","HEAD"),"branch":git("branch","--show-current"),"python":sys.version,"platform":platform.platform(),"mss_reference":"pinned Stage 1 reference; coefficients embargoed until freeze","openfoam":runtime,"geometry":None if geometry is None else {"path":str(geometry.resolve()),"sha256":sha256(geometry) if geometry.is_file() else None,"source_url":source_url,"source_commit":source_commit},"capabilities":caps,"stage1_scenarios":STAGE1_SCENARIOS}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    provenance={"status":"NOT_GENERATED","reason":"Model freeze prohibited because geometry/CFD prerequisites failed","allowed_kinds":sorted(ALLOWED_PROVENANCE),"mss_leakage_guard":"PASS"}
    leakage_guard(provenance);(output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    status="BLOCKED_GEOMETRY" if geometry_blocked else "FAIL"
    summary={"stage3_status":status,"geometry_qualification":None if qualification is None else qualification["status"],"surface_check":"NOT_RUN_GEOMETRY_GATE","cfd_cases":{"completed":0,"failed":0,"not_run":True},"mesh_convergence":"NOT_RUN","model_frozen":False,"direct_wrench_comparison":"NOT_RUN","actuator_comparison":"NOT_RUN","coefficient_comparison":"NOT_RUN","sensitivity":"NOT_RUN","failures":failures}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# BCOD Stage 3 OpenFOAM validation","","## Executive result","",f"**STAGE 3: {status}**","","No blind model was frozen and no MSS comparison was performed.","","## Geometry qualification",""]
    if qualification:
        whole=qualification["scene"]["overall"]; lines += [f"- Source: `{source_url}` at commit `{source_commit}`",f"- SHA256: `{qualification['source']['sha256']}`",f"- Native declaration: unit scale `{qualification['source']['unit_meter']}` metre, `{qualification['source']['up_axis']}`",f"- Scene: {qualification['scene']['component_count']} meshes, {qualification['scene']['triangle_count']} triangles",f"- Full visual bounds (native XYZ): `{whole['extents_m']}` m",f"- Candidate hull spacing: `{qualification['candidate_center_spacing_m']}` m",f"- Candidate hull extents: `{[h['extents_m'] for h in qualification['hull_candidates']]}` m",f"- Candidate hull boundary edges: `{[h['boundary_edges'] for h in qualification['hull_candidates']]}`",f"- Hydrostatic sweep: **{qualification['hydrostatic_sweep']}**",""]
    lines += ["## Gate failures",""]
    lines += [f"- **{item['category']}** — {item['reason']}" for item in failures]
    lines += ["","## CFD result","","0 cases completed; CFD was not launched because pre-freeze prerequisites failed.","","## Integrity decision","","The verified smoke campaign cannot substitute for this stage: it uses a synthetic ellipsoidal hull, positive surge only, one mesh size, and no forced-motion added-mass or geometry-derived hydrostatics. The legacy approximate Otter proxy is likewise inadmissible. No `otter_openfoam_derived` config has been created.",""]
    (output/"report.md").write_text("\n".join(lines))
    return summary

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--run-id",default="latest");p.add_argument("--geometry",type=Path);p.add_argument("--source-url");p.add_argument("--source-commit");p.add_argument("--output",type=Path);a=p.parse_args();output=a.output.resolve() if a.output else ROOT/"stage3_results"/a.run_id
    summary=run(output,a.geometry,a.source_url,a.source_commit);print(output);return 0 if summary["stage3_status"] in {"PASS","PASS_WITH_MODEL_FORM_LIMITATIONS"} else 1
if __name__=="__main__":raise SystemExit(main())
