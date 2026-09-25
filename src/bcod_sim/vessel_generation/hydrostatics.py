"""Geometry-derived equilibrium hydrostatics for closed triangulated hulls."""

from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import numpy as np


@dataclass(frozen=True)
class HydrostaticResult:
    submerged_volume_m3: float
    equilibrium_waterline_z_m: float
    draft_m: float
    center_of_buoyancy_frd_m: tuple[float,float,float]
    waterplane_area_m2: float
    waterplane_centroid_frd_m: tuple[float,float,float]
    waterplane_i_roll_m4: float
    waterplane_i_pitch_m4: float
    heave_restoring_n_m: float
    roll_restoring_nm_rad: float
    pitch_restoring_nm_rad: float
    stiffness_6x6: tuple[tuple[float,...],...]

    def to_dict(self): return asdict(self)


def load_ascii_stl(path: str|Path) -> tuple[np.ndarray,list[tuple[int,int,int]]]:
    text=Path(path).read_text(errors="strict")
    raw=[tuple(map(float,m)) for m in re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",text)]
    if not raw or len(raw)%3: raise ValueError("ASCII STL has no complete triangles")
    points=[];lookup={};faces=[]
    for start in range(0,len(raw),3):
        face=[]
        for value in raw[start:start+3]:
            key=tuple(round(x,10) for x in value)
            if key not in lookup: lookup[key]=len(points);points.append(value)
            face.append(lookup[key])
        faces.append(tuple(face))
    return np.asarray(points,float),faces


def _polygons(points,faces,z):
    segments=[]
    for face in faces:
        tri=[points[i] for i in face];hits=[]
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            da,db=a[2]-z,b[2]-z
            if da*db<0: hits.append(a+da/(da-db)*(b-a))
        if len(hits)==2: segments.append((hits[0][:2],hits[1][:2]))
    coordinates={};adjacency=defaultdict(list)
    key=lambda p:tuple(np.round(p,7))
    for a,b in segments:
        ka,kb=key(a),key(b);coordinates[ka]=a;coordinates[kb]=b;adjacency[ka].append(kb);adjacency[kb].append(ka)
    unused={tuple(sorted((a,b))) for a,neighbors in adjacency.items() for b in neighbors}; result=[]
    while unused:
        start,current=unused.pop();loop=[start,current];previous=start
        while current!=start:
            candidates=[n for n in adjacency[current] if n!=previous and tuple(sorted((current,n))) in unused]
            if not candidates: break
            nxt=candidates[0];unused.discard(tuple(sorted((current,nxt))));loop.append(nxt);previous,current=current,nxt
        if current==start and len(loop)>3: result.append(np.asarray([coordinates[n] for n in loop[:-1]]))
    return result


def _section(points,faces,z):
    values=[]
    for polygon in _polygons(points,faces,z):
        x,y=polygon[:,0],polygon[:,1];xn,yn=np.roll(x,-1),np.roll(y,-1);cross=x*yn-xn*y;signed=cross.sum()/2
        if abs(signed)<1e-14: continue
        cx=((x+xn)*cross).sum()/(6*signed);cy=((y+yn)*cross).sum()/(6*signed)
        ixx=abs(((y*y+y*yn+yn*yn)*cross).sum()/12-signed*cy*cy)
        iyy=abs(((x*x+x*xn+xn*xn)*cross).sum()/12-signed*cx*cx)
        values.append((abs(signed),cx,cy,ixx,iyy))
    area=sum(v[0] for v in values)
    if not area:return (0.,0.,0.,0.,0.)
    cx=sum(v[0]*v[1] for v in values)/area;cy=sum(v[0]*v[2] for v in values)/area
    return area,cx,cy,sum(v[3]+v[0]*(v[2]-cy)**2 for v in values),sum(v[4]+v[0]*(v[1]-cx)**2 for v in values)


def derive_hydrostatics(points, faces, *, mass_kg: float, cg_frd_m=(0.,0.,0.), density_kg_m3=1025.,
                        gravity_mps2=9.80665, sections=1201) -> HydrostaticResult:
    points=np.asarray(points,float);zmin,zmax=float(points[:,2].min()),float(points[:,2].max())
    zs=np.linspace(zmin+1e-7,zmax-1e-7,sections);values=np.asarray([_section(points,faces,z) for z in zs]);areas=values[:,0]
    cumulative=np.r_[0,np.cumsum((areas[:-1]+areas[1:])*np.diff(zs)/2)];target=mass_kg/density_kg_m3
    if target<=0 or target>cumulative[-1]: raise ValueError("mass is outside geometry displacement capacity")
    waterline=float(np.interp(target,cumulative,zs));sub=zs[zs<=waterline]
    if sub[-1]<waterline: sub=np.r_[sub,waterline]
    vals=np.asarray([_section(points,faces,z) for z in sub]);a=vals[:,0];volume=float(np.trapezoid(a,sub))
    cob=(float(np.trapezoid(a*vals[:,1],sub)/volume),float(np.trapezoid(a*vals[:,2],sub)/volume),float(np.trapezoid(a*sub,sub)/volume))
    wp=_section(points,faces,waterline);bm_roll=wp[3]/volume;bm_pitch=wp[4]/volume
    # Source STL z is positive up while body FRD z is positive down. Therefore
    # source KG = -cg_frd_z and GM = KB + BM - KG.
    gm_roll=cob[2]+bm_roll+cg_frd_m[2];gm_pitch=cob[2]+bm_pitch+cg_frd_m[2]
    heave=density_kg_m3*gravity_mps2*wp[0];roll=density_kg_m3*gravity_mps2*volume*gm_roll;pitch=density_kg_m3*gravity_mps2*volume*gm_pitch
    stiffness=np.zeros((6,6));stiffness[2,2]=heave;stiffness[3,3]=roll;stiffness[4,4]=pitch
    cob_frd=(cob[0],-cob[1],-cob[2]);wp_frd=(wp[1],-wp[2],-waterline)
    return HydrostaticResult(volume,waterline,waterline-zmin,cob_frd,(wp[0]),wp_frd,wp[3],wp[4],heave,roll,pitch,tuple(map(tuple,stiffness)))
