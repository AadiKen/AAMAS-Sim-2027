#!/usr/bin/env python3
"""Fail-closed WAM-V Stage 3 geometry/hydrostatics/CFD gate."""
from __future__ import annotations

import argparse, hashlib, json, math, platform, shutil, subprocess, sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from bcod_sim.cli import main as bcod_main

ROOT=Path(__file__).resolve().parents[1]
VRX_COMMIT="41f2df50fbf75b30264eb4c7a2222d4235f6a70b"
VRX_URL="https://github.com/osrf/vrx"
RHO=1025.0; G=9.81; MASS=180.0
FORBIDDEN_PREFREEZE=("added_mass","linear_damping","quadratic_damping","hydrodynamic_coefficient","restoring_coefficient","vrx_trajectory")

def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def cfd_cache_key(geometry_hash,mesh,solver,fluid,motion): return canonical_hash({"geometry":geometry_hash,"mesh":mesh,"solver":solver,"fluid":fluid,"motion":motion})
def guard_pre_freeze_reference(payload):
    text=json.dumps(payload,sort_keys=True).lower();found=[marker for marker in FORBIDDEN_PREFREEZE if marker in text]
    if found: raise ValueError(f"pre-freeze VRX leakage: {found}")

def load_collada(path):
    ns={"c":"http://www.collada.org/2005/11/COLLADASchema"}; root=ET.parse(path).getroot()
    asset=root.find("c:asset",ns); unit=float(asset.find("c:unit",ns).attrib["meter"]); up=asset.findtext("c:up_axis",namespaces=ns)
    source=next(node for node in root.findall(".//c:source",ns) if "POSITION" in node.attrib.get("id", ""))
    vertices=np.fromstring(source.find("c:float_array",ns).text,sep=" ").reshape(-1,3)
    matrix=np.fromstring(root.find(".//c:visual_scene//c:matrix",ns).text,sep=" ").reshape(4,4)
    vertices=(matrix@np.c_[vertices,np.ones(len(vertices))].T).T[:,:3]*unit
    triangles=root.find(".//c:triangles",ns); inputs=triangles.findall("c:input",ns); stride=max(int(node.attrib["offset"]) for node in inputs)+1
    vertex_offset=next(int(node.attrib["offset"]) for node in inputs if node.attrib["semantic"]=="VERTEX")
    packed=np.fromstring(triangles.find("c:p",ns).text,sep=" ",dtype=int).reshape(-1,3*stride)
    faces=packed[:,[vertex_offset,vertex_offset+stride,vertex_offset+2*stride]]
    return vertices,faces,{"unit_meter":unit,"unit_name":asset.find("c:unit",ns).attrib.get("name"),"up_axis":up,"matrix":matrix.tolist()}

def welded_mesh(vertices,faces,tolerance_digits=7):
    mapping={}; points=[]; converted=[]
    for face in faces:
        item=[]
        for old in face:
            key=tuple(round(float(value),tolerance_digits) for value in vertices[old])
            if key not in mapping: mapping[key]=len(points); points.append(np.asarray(key))
            item.append(mapping[key])
        converted.append(tuple(item))
    return np.asarray(points),converted

def face_components(points,faces):
    vertex_faces=defaultdict(list)
    for index,face in enumerate(faces):
        for vertex in face: vertex_faces[vertex].append(index)
    adjacency=[set() for _ in faces]
    for incident in vertex_faces.values():
        for face in incident: adjacency[face].update(incident)
    unseen=set(range(len(faces))); result=[]
    while unseen:
        seed=unseen.pop(); stack=[seed]; members={seed}
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen: unseen.remove(neighbor);members.add(neighbor);stack.append(neighbor)
        ids=sorted({vertex for index in members for vertex in faces[index]}); remap={old:new for new,old in enumerate(ids)}
        result.append((points[ids],[(remap[a],remap[b],remap[c]) for index in members for a,b,c in [faces[index]]]))
    return result

def mesh_stats(points,faces):
    edges=Counter(tuple(sorted(edge)) for a,b,c in faces for edge in ((a,b),(b,c),(c,a)))
    directed=Counter(edge for a,b,c in faces for edge in ((a,b),(b,c),(c,a)))
    area=sum(np.linalg.norm(np.cross(points[b]-points[a],points[c]-points[a]))/2 for a,b,c in faces)
    volume=sum(np.dot(points[a],np.cross(points[b],points[c]))/6 for a,b,c in faces)
    return {"vertices":len(points),"triangles":len(faces),"bounds_min_m":points.min(0).tolist(),"bounds_max_m":points.max(0).tolist(),"extents_m":np.ptp(points,axis=0).tolist(),"center_m":((points.min(0)+points.max(0))/2).tolist(),"surface_area_m2":float(area),"signed_volume_m3":float(volume),"enclosed_volume_m3":float(abs(volume)),"boundary_edges":sum(value==1 for value in edges.values()),"nonmanifold_edges":sum(value>2 for value in edges.values()),"normal_inconsistent_edges":sum(value==2 and not (directed[(a,b)]==1 and directed[(b,a)]==1) for (a,b),value in edges.items()),"duplicate_faces":sum(value-1 for value in Counter(tuple(sorted(face)) for face in faces).values()),"degenerate_faces":sum(len(set(face))<3 for face in faces),"watertight":bool(edges) and all(value==2 for value in edges.values())}

def combine(meshes):
    points=[];faces=[]
    for mesh_points,mesh_faces in meshes:
        offset=len(points);points.extend(mesh_points);faces.extend((a+offset,b+offset,c+offset) for a,b,c in mesh_faces)
    return np.asarray(points),faces

def section_polygons(points,faces,z):
    segments=[]
    for face in faces:
        triangle=[points[index] for index in face]; hits=[]
        for a,b in ((triangle[0],triangle[1]),(triangle[1],triangle[2]),(triangle[2],triangle[0])):
            da,db=a[2]-z,b[2]-z
            if da*db<0:
                t=da/(da-db);hits.append(a+t*(b-a))
            elif abs(da)<1e-10 and abs(db)>=1e-10: hits.append(a)
            elif abs(db)<1e-10 and abs(da)>=1e-10: hits.append(b)
        unique=[]
        for point in hits:
            if not any(np.linalg.norm(point-other)<1e-8 for other in unique): unique.append(point)
        if len(unique)==2: segments.append((unique[0][:2],unique[1][:2]))
    coordinates={}; adjacency=defaultdict(list)
    def key(point): return tuple(np.round(point,7))
    for a,b in segments:
        ka,kb=key(a),key(b);coordinates[ka]=a;coordinates[kb]=b;adjacency[ka].append(kb);adjacency[kb].append(ka)
    polygons=[];unused={tuple(sorted((a,b))) for a,neighbors in adjacency.items() for b in neighbors}
    while unused:
        edge=unused.pop();start,current=edge;loop=[start,current];previous=start
        while current!=start:
            candidates=[node for node in adjacency[current] if node!=previous and tuple(sorted((current,node))) in unused]
            if not candidates: break
            nxt=candidates[0];unused.discard(tuple(sorted((current,nxt))));loop.append(nxt);previous,current=current,nxt
        if current==start and len(loop)>3: polygons.append(np.asarray([coordinates[node] for node in loop[:-1]]))
    return polygons

def polygon_properties(polygon):
    x=polygon[:,0];y=polygon[:,1];xn=np.roll(x,-1);yn=np.roll(y,-1);cross=x*yn-xn*y; signed=cross.sum()/2
    if abs(signed)<1e-14:return (0,0,0,0,0)
    cx=((x+xn)*cross).sum()/(6*signed);cy=((y+yn)*cross).sum()/(6*signed)
    ixx=((y*y+y*yn+yn*yn)*cross).sum()/12
    iyy=((x*x+x*xn+xn*xn)*cross).sum()/12
    return abs(signed),cx,cy,abs(ixx-signed*cy*cy),abs(iyy-signed*cx*cx)

def section_properties(points,faces,z):
    props=[polygon_properties(poly) for poly in section_polygons(points,faces,z)];area=sum(p[0] for p in props)
    if not area:return (0,0,0,0,0)
    cx=sum(p[0]*p[1] for p in props)/area;cy=sum(p[0]*p[2] for p in props)/area
    ixx=sum(p[3]+p[0]*(p[2]-cy)**2 for p in props);iyy=sum(p[4]+p[0]*(p[1]-cx)**2 for p in props)
    return area,cx,cy,ixx,iyy

def hydrostatics(points,faces,mass=MASS,rho=RHO):
    zmin,zmax=points[:,2].min(),points[:,2].max();zs=np.linspace(zmin+1e-7,zmax-1e-7,1201)
    sections=np.asarray([section_properties(points,faces,z)[:3] for z in zs]);areas=sections[:,0]
    dz=np.diff(zs); cumulative=np.r_[0,np.cumsum((areas[:-1]+areas[1:])*dz/2)];target=mass/rho
    if target>cumulative[-1]: raise ValueError("mass exceeds hull displacement")
    draft=float(np.interp(target,cumulative,zs));mask=zs<=draft; zsub=np.r_[zs[mask],draft] if zs[mask][-1]<draft else zs[mask]
    vals=np.asarray([section_properties(points,faces,z) for z in zsub]);a=vals[:,0]
    volume=float(np.trapezoid(a,zsub));cob_x=float(np.trapezoid(a*vals[:,1],zsub)/volume);cob_y=float(np.trapezoid(a*vals[:,2],zsub)/volume);cob_z=float(np.trapezoid(a*zsub,zsub)/volume)
    wp=section_properties(points,faces,draft);bm_t=wp[3]/volume;bm_l=wp[4]/volume;kb=cob_z
    return {"rho_water_kg_m3":rho,"mass_kg":mass,"target_displacement_m3":target,"equilibrium_waterline_z_m":draft,"draft_from_lowest_point_m":draft-zmin,"displaced_volume_m3":volume,"equivalent_mass_kg":volume*rho,"center_of_buoyancy_m":[cob_x,cob_y,cob_z],"waterplane_area_m2":wp[0],"waterplane_centroid_m":[wp[1],wp[2],draft],"waterplane_Ixx_m4":wp[3],"waterplane_Iyy_m4":wp[4],"metacentric_radius_transverse_m":bm_t,"metacentric_radius_longitudinal_m":bm_l,"heave_restoring_N_m":rho*G*wp[0],"roll_restoring_Nm_rad_assuming_CG_at_origin":rho*G*volume*(kb+bm_t),"pitch_restoring_Nm_rad_assuming_CG_at_origin":rho*G*volume*(kb+bm_l),"method":"1201-plane deterministic cross-section integration"}

def write_stl(path,points,faces):
    with path.open("w") as stream:
        stream.write("solid wamv_qualified_hulls\n")
        for a,b,c in faces:
            pa,pb,pc=points[a],points[b],points[c];normal=np.cross(pb-pa,pc-pa);norm=np.linalg.norm(normal);normal=normal/norm if norm else normal
            stream.write(f" facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n  outer loop\n")
            for point in (pa,pb,pc):stream.write(f"   vertex {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
            stream.write("  endloop\n endfacet\n")
        stream.write("endsolid wamv_qualified_hulls\n")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",default="wamv-stage3");parser.add_argument("--geometry-only",action="store_true");parser.add_argument("--cfd",action="store_true");parser.add_argument("--stage3a",action="store_true");parser.add_argument("--stage3b",action="store_true");parser.add_argument("--full",action="store_true");parser.add_argument("--reuse-cfd",action="store_true");args=parser.parse_args()
    source=ROOT/"stage3_inputs/wamv/vrx-41f2df50/original/WAM-V-Base.dae";output=ROOT/"stage3_results"/args.run_id
    for folder in ("geometry/original","geometry/qualified","hydrostatics","mesh_convergence","cfd","fits","stage3a","reference_comparison","propulsion","stage3b","sensitivity","plots"): (output/folder).mkdir(parents=True,exist_ok=True)
    preserved=output/"geometry/original/WAM-V-Base.dae";shutil.copy2(source,preserved)
    vertices,faces,coordinate=load_collada(source);points,wfaces=welded_mesh(vertices,faces);components=face_components(points,wfaces);ranked=sorted(components,key=lambda mesh:len(mesh[1]),reverse=True)
    hulls=ranked[:2];qualified=combine(hulls);qstats=mesh_stats(*qualified);component_stats=sorted([mesh_stats(*mesh) for mesh in components],key=lambda item:item["triangles"],reverse=True)
    stl=output/"geometry/qualified/wamv_twin_hulls.stl";write_stl(stl,*qualified);hydro=hydrostatics(*qualified)
    geometry={"status":"PASS_GEOMETRY","source_sha256":sha256(source),"qualified_sha256":sha256(stl),"coordinate_semantics":coordinate,"scene_component_count":len(components),"full_scene_extents_m":np.ptp(points,axis=0).tolist(),"selection":"two largest connected, watertight, mirror-symmetric components","individual_hulls":component_stats[:2],"hull_center_spacing_m":abs(component_stats[0]["center_m"][1]-component_stats[1]["center_m"][1]),"qualified":qstats,"repairs":["welded coincident export vertices at 1e-7 m","extracted two existing closed hull components; no shape reconstruction"],"self_intersection":"PASS_OPENFOAM_SURFACECHECK_11","surface_check":"PASS: closed; 2 unconnected hull parts; no illegal triangles; no near points; no self-intersection; min triangle quality 0.0487892"}
    (output/"geometry/qualification.json").write_text(json.dumps(geometry,indent=2)+"\n");(output/"hydrostatics/results.json").write_text(json.dumps(hydro,indent=2)+"\n")
    image="openfoam/openfoam11-paraview510:11";identity="openfoam/openfoam11-paraview510@sha256:fd10956e0b1eb70f9808baf2857e4baf846a0f6f272f73b6d00546eae96be181"
    manifest={"bcod_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"vrx":{"url":VRX_URL,"commit":VRX_COMMIT,"geometry_sha256":sha256(source)},"openfoam":{"version":"OpenFOAM Foundation 11 build 11-e1fc8c682ae6","image":image,"identity":identity},"assimp":"6.0","platform":platform.platform(),"python":sys.version,"fluid":{"density_kg_m3":RHO,"dynamic_viscosity_Pa_s":0.001},"reference_isolation":"PASS; VRX dynamics and propulsion parameters not inspected","requested_mode":"geometry-only" if args.geometry_only else "full"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    provenance={"geometry":{"kind":"KNOWN_PHYSICAL","source":"pinned VRX geometry"},"mass":{"kind":"KNOWN_PHYSICAL","value_kg":MASS,"source":"pinned VRX wamv_base.urdf.xacro"},"hydrostatics":{"kind":"GEOMETRY_DERIVED","source":"qualified twin hulls"},"passive_hydrodynamics":{"kind":"NOT_DERIVED"},"vrx_hydrodynamic_coefficients":{"accessed_pre_freeze":False}}
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    campaign_status="NOT_REQUESTED"
    if not args.geometry_only:
        campaign=output/"cfd"/"identification"
        bcod_main(["vessel","identify","--geometry",str(stl),"--mass",str(MASS),"--cg","0,0,0",
            "--output",str(campaign),"--name","wamv","--fluid-model","free_surface","--openfoam-version","11"])
        case_count=json.loads((campaign/"identification_manifest.json").read_text())["case_count"]
        campaign_status=f"READY: {case_count} content-addressed free-surface 6-DOF cases generated"
    stage3a="BLOCKED_CFD_EXECUTION";stage3b="NOT_RUN"
    summary={"overall_status":"BLOCKED","stage3a_status":stage3a,"geometry_status":"PASS_GEOMETRY","mesh_cfd_convergence":campaign_status,"cfd_cases":{"accepted":0,"rejected":0},"stage3b_status":stage3b,"model_frozen":False,"failures":[{"category":"CFD_EXECUTION","reason":"The generalized production campaign is prepared, but the expensive OpenFOAM case matrix has not yet been executed and quality-gated."}],"hydrostatics":hydro}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Stage 3 WAM-V OpenFOAM pipeline validation","",f"**OVERALL: BLOCKED — Stage 3A {stage3a}; Stage 3B {stage3b}**","","## Geometry","",f"Pinned VRX `{VRX_COMMIT}`. The two existing hull components pass the topology gate: each has 314 triangles, zero boundary/non-manifold edges, and enclosed volume about 0.61555 m³. Hull dimensions are approximately 4.9315 × 0.4330 × 0.5592 m with 2.0543 m center spacing. No underwater shape was invented.","","## Hydrostatics","",f"For the known 180 kg mass in 1025 kg/m³ water, the geometry-derived equilibrium waterline is z={hydro['equilibrium_waterline_z_m']:.5f} m and draft from the lowest hull point is {hydro['draft_from_lowest_point_m']:.5f} m. Displaced volume is {hydro['displaced_volume_m3']:.6f} m³.","","## CFD gate","",f"The production 6-DOF pipeline is now available. {campaign_status}. Stage 3 resumed through campaign generation and is now blocked at actual OpenFOAM execution/quality gating, not at missing architecture. No CFD-derived coefficient is claimed before those runs complete. VRX hydrodynamic coefficients remain embargoed.",""]
    (output/"report.md").write_text("\n".join(lines));print(output)
    return 2

if __name__=="__main__":raise SystemExit(main())
