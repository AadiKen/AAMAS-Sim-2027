"""Reconstruct official NMRI KVLCC2 underwater grid for independent 2D+t test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from tools.build_kcs_hmri_geometry import grid

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'stage3_inputs/twodt/kvlcc2/surface'
OUT=ROOT/'stage3_results/twodt/kvlcc2_independent'
LPP=5.5172


def main():
    bow_file=SOURCE/'kvlcc_bow1.dat';stern_file=SOURCE/'kvlcc2_stn1.dat'
    bow,stern=grid(bow_file),grid(stern_file)
    if bow.shape!=stern.shape or np.max(np.abs(bow[:,-1]-stern[:,0]))>1e-5:
        raise ValueError('Official fore/aft grid seam does not match')
    common=(bow[:,-1]+stern[:,0])/2
    bow[:,-1]=common;stern[:,0]=common
    half=np.concatenate((bow,stern[:,1:]),axis=1)
    if np.max(np.abs(half[-1,:,2]))>1e-8 or np.min(half[:,:,1]) < -1e-8:
        raise ValueError('Unexpected official waterline/side convention')
    star=half*np.array([-LPP,LPP,-LPP]);port=star.copy();port[:,:,1]*=-1
    nk,ni=star.shape[:2]
    vertices=np.concatenate((star.reshape(-1,3),port.reshape(-1,3)))
    idx=lambda side,k,i:side*nk*ni+k*ni+i
    faces=[]
    for side in (0,1):
        for k in range(nk-1):
            for i in range(ni-1):
                a,b,c,d=idx(side,k,i),idx(side,k,i+1),idx(side,k+1,i+1),idx(side,k+1,i)
                faces.extend(((a,b,c),(a,c,d)) if side==0 else ((a,c,b),(a,d,c)))
    for i in range(ni-1):
        a,b=idx(0,nk-1,i),idx(0,nk-1,i+1)
        c,d=idx(1,nk-1,i+1),idx(1,nk-1,i)
        faces.extend(((a,b,c),(a,c,d)))
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    mesh.update_faces(mesh.nondegenerate_faces());mesh.merge_vertices(digits_vertex=8);mesh.fix_normals()
    pieces=sorted(mesh.split(only_watertight=False),key=lambda m:len(m.faces),reverse=True)
    fragments=[{'faces':len(m.faces),'area_m2':float(m.area),'volume_m3':float(m.volume)} for m in pieces[1:]]
    if any(x['area_m2']>1e-4 or abs(x['volume_m3'])>1e-9 for x in fragments):
        raise ValueError('Material disconnected grid fragments')
    mesh=pieces[0]
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('KVLCC2 reconstruction failed closure')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'kvlcc2_underwater.stl';mesh.export(path)
    waterline=star[-1];order=np.argsort(waterline[:,0]);waterplane=float(np.trapezoid(2*waterline[order,1],waterline[order,0]))
    report={'source':'official NMRI KVLCC2 *_bow1/stn1 underwater grids',
            'source_url':'https://www.nmri.go.jp/study/research_organization/fluid_performance/cfd/cfdws05/gothenburg2000/KVLCC/kvlcc_g%26c.htm',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (bow_file,stern_file)},
            'scale_Lpp_m':LPP,'transform':'FRD=(-x,+y,-z)*Lpp from NMRI',
            'artificial_closure':'planar waterline cap; hydrostatics/sectioning only, not 3D CFD',
            'watertight':bool(mesh.is_watertight),'faces':len(mesh.faces),'volume_m3':float(mesh.volume),
            'volume_target_m3':1.6029,'volume_relative_error':float((mesh.volume-1.6029)/1.6029),
            'beam_m':float(mesh.extents[1]),'draft_m':float(mesh.bounds[1,2]),
            'wetted_area_m2':float(mesh.area-abs(waterplane)),'bbox_frd_m':mesh.bounds.tolist(),
            'stl_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'discarded_fragments':fragments}
    (OUT/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
