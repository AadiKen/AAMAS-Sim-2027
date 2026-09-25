"""Reconstruct an underwater KCS hydrostatic shell from official NMRI grids.

This is a geometry *qualification* tool. Its planar waterplane closure is not
an above-water CFD hull, so the exported STL must not be passed to snappyHexMesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def grid(path: Path) -> np.ndarray:
    header=path.read_text().splitlines()[:2]
    import re
    match=re.search(r"Zone\s+i=\s*(\d+)\s+j=\s*(\d+)\s+f=point",header[1],re.I)
    if not match: raise ValueError(f"unsupported NMRI surface grid header: {path}")
    ni,nj=map(int,match.groups())
    points=np.loadtxt(path,skiprows=2)
    if points.shape!=(ni*nj,3): raise ValueError(f"incomplete NMRI grid: {path}")
    return points.reshape(nj,ni,3)


def reconstruct(bow_path: Path, stern_path: Path, out: Path) -> dict:
    import trimesh

    bow,stern=grid(bow_path),grid(stern_path)
    if bow.shape!=stern.shape or not np.allclose(bow[:,-1],stern[:,0],atol=1e-10):
        raise ValueError("official fore/aft grids do not share the midship seam")
    half=np.concatenate((bow,stern[:,1:]),axis=1)
    if np.max(np.abs(half[-1,:,2]))>1e-10 or np.min(half[:,:,1]) < -1e-10:
        raise ValueError("NMRI grid waterline or starboard sign is unexpected")
    # Official coordinates: nondimensional x downstream, y starboard, z up;
    # BCOD FRD: x forward, y starboard, z down. Scale by model Lpp=5.75 m.
    star=half.copy()*np.array([-5.75,5.75,-5.75])
    port=star.copy();port[:,:,1]*=-1
    nk,ni=star.shape[:2]
    vertices=np.concatenate((star.reshape(-1,3),port.reshape(-1,3)))
    def idx(side: int,k: int,i: int) -> int: return side*nk*ni+k*ni+i
    faces=[]
    for side in (0,1):
        for k in range(nk-1):
            for i in range(ni-1):
                a,b,c,d=idx(side,k,i),idx(side,k,i+1),idx(side,k+1,i+1),idx(side,k+1,i)
                faces.extend(((a,b,c),(a,c,d)) if side==0 else ((a,c,b),(a,d,c)))
    hull_face_count=len(faces)
    # Fill the official z=0 waterplane outline by strips across corresponding
    # port/starboard grid stations. This cap is only for hydrostatic integration.
    for i in range(ni-1):
        a,b=idx(0,nk-1,i),idx(0,nk-1,i+1)
        c,d=idx(1,nk-1,i+1),idx(1,nk-1,i)
        faces.extend(((a,b,c),(a,c,d)))
    mesh=trimesh.Trimesh(vertices=vertices,faces=np.asarray(faces),process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices(digits_vertex=9)
    mesh.fix_normals()
    pieces=sorted(mesh.split(only_watertight=False),key=lambda item:len(item.faces),reverse=True)
    seam_fragments=[{"faces":len(item.faces),"area_m2":float(item.area),
                     "volume_m3":float(item.volume)} for item in pieces[1:]]
    if any(abs(item["volume_m3"])>1e-9 or item["area_m2"]>1e-4 for item in seam_fragments):
        raise ValueError("official grid reconstruction has a material disconnected piece")
    mesh=pieces[0]
    boundary_count=int(np.sum(np.bincount(mesh.edges_unique_inverse)==1))
    bbox=np.asarray(mesh.bounds)
    # Waterplane area uses the source stations, independent of cap tessellation.
    waterline=star[-1]
    x=waterline[:,0];beam=2*waterline[:,1]
    area=float(abs(np.trapezoid(beam[np.argsort(x)],np.sort(x))))
    volume=float(abs(mesh.volume))
    # Triangles from the two official grids comprise the wetted surface.
    wetted=float(mesh.area-area)
    report={
        "purpose":"underwater hydrostatic geometry qualification only; not CFD-ready freeboard",
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (bow_path,stern_path)},
        "source":"NMRI 2005 official KCS surface-grid archive, bow1/stn1 designed-load-waterline blocks",
        "model_Lpp_m":5.75,"transform":"(x,y,z)_FRD = (-x,+y,-z)_NMRI * 5.75",
        "artificial_closure":"planar cap at z=0 from official waterline stations; hydrostatics only",
        "vertices":len(mesh.vertices),"faces":len(mesh.faces),"source_hull_faces_before_dedup":hull_face_count,
        "discarded_zero_volume_seam_fragments":seam_fragments,
        "watertight":bool(mesh.is_watertight),"winding_consistent":bool(mesh.is_winding_consistent),
        "open_edges":boundary_count,"euler_number":int(mesh.euler_number),
        "bbox_frd_m":bbox.tolist(),"max_beam_m":float(bbox[1,1]-bbox[0,1]),
        "design_draft_m":float(bbox[1,2]),"submerged_volume_m3":volume,
        "waterplane_area_m2":area,"wetted_surface_area_m2":wetted,
        "center_of_buoyancy_frd_m":mesh.center_mass.tolist(),
        "displacement_target_m3":0.813,"displacement_relative_error":(volume-.813)/.813,
        "self_intersection_verified":False,
        "cfd_ready":False,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    mesh.export(out)
    report["stl_sha256"]=hashlib.sha256(out.read_bytes()).hexdigest()
    return report


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--grid-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir/"kcs_hmri_underwater_hydrostatic_only.stl"
    report=reconstruct(args.grid_dir/"kcs_bow1.dat",args.grid_dir/"kcs_stn1.dat",out)
    (args.output_dir/"underwater_hydrostatics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
