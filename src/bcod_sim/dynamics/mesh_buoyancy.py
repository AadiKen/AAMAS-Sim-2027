"""Deterministic triangle/plane mesh buoyancy and equilibrium utilities."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any
import numpy as np
import torch
from bcod_sim.core.errors import OperatingEnvelopeError, PhysicalValidationError
from bcod_sim.dynamics.restoring import WrenchResult, rotate_world_to_body
from bcod_sim.state.vessel_state import VesselState

@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    content_hash: str

    @classmethod
    def from_arrays(cls,vertices:np.ndarray,faces:np.ndarray)->"TriangleMesh":
        vertices=np.asarray(vertices,dtype=np.float64); faces=np.asarray(faces,dtype=np.int64)
        payload=vertices.tobytes()+faces.tobytes()
        mesh=cls(vertices,faces,hashlib.sha256(payload).hexdigest()); mesh.validate(); return mesh

    @classmethod
    def from_stl(cls,path:str|Path)->"TriangleMesh":
        path=Path(path); data=path.read_bytes()
        triangles=[]
        if len(data)>=84 and 84+50*struct.unpack("<I",data[80:84])[0]==len(data):
            count=struct.unpack("<I",data[80:84])[0]
            for i in range(count):
                values=struct.unpack("<12fH",data[84+50*i:84+50*(i+1)])
                triangles.append(np.array(values[3:12],dtype=float).reshape(3,3))
        else:
            current=[]
            for line in data.decode("utf-8").splitlines():
                words=line.strip().split()
                if words and words[0].lower()=="vertex":
                    current.append([float(x) for x in words[1:4]])
                    if len(current)==3: triangles.append(np.asarray(current)); current=[]
        if not triangles: raise PhysicalValidationError("STL contains no triangles")
        lookup:dict[tuple[float,float,float],int]={}; vertices=[]; faces=[]
        for tri in triangles:
            face=[]
            for point in tri:
                key=tuple(float(x) for x in point)
                if key not in lookup: lookup[key]=len(vertices); vertices.append(point)
                face.append(lookup[key])
            faces.append(face)
        parsed=cls.from_arrays(np.asarray(vertices),np.asarray(faces))
        return cls(parsed.vertices,parsed.faces,hashlib.sha256(data).hexdigest())

    def validate(self)->None:
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or self.faces.ndim!=2 or self.faces.shape[1]!=3:
            raise PhysicalValidationError("Mesh must contain Nx3 vertices and Mx3 triangular faces")
        if not np.isfinite(self.vertices).all() or len(self.faces)<4 or self.faces.min()<0 or self.faces.max()>=len(self.vertices):
            raise PhysicalValidationError("Mesh contains invalid geometry")
        edges:dict[tuple[int,int],list[tuple[int,int]]]={}
        for face in self.faces:
            a,b,c=map(int,face)
            if len({a,b,c})<3: raise PhysicalValidationError("Mesh contains degenerate face")
            for u,v in ((a,b),(b,c),(c,a)): edges.setdefault(tuple(sorted((u,v))),[]).append((u,v))
        if any(len(uses)!=2 or uses[0]==uses[1] for uses in edges.values()):
            raise PhysicalValidationError("Mesh must be closed, watertight, and consistently oriented")
        volume,_=polyhedron_volume_centroid(self.vertices[self.faces])
        if not math.isfinite(volume) or volume<=1e-12: raise PhysicalValidationError("Mesh volume must be nonzero")

def polyhedron_volume_centroid(triangles:np.ndarray)->tuple[float,np.ndarray]:
    signed=0.0; first=np.zeros(3)
    for a,b,c in triangles:
        v=float(np.dot(a,np.cross(b,c))/6); signed+=v; first+=v*(a+b+c)/4
    if abs(signed)<1e-15: return 0.0,np.zeros(3)
    return abs(signed),first/signed

def _clip_polygon(poly:list[np.ndarray],normal:np.ndarray,offset:float,tol:float=1e-12)->tuple[list[np.ndarray],list[np.ndarray]]:
    out=[]; cuts=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        da=float(normal@a-offset); db=float(normal@b-offset); ina=da>=-tol; inb=db>=-tol
        if ina: out.append(a)
        if ina!=inb:
            point=a+(b-a)*(da/(da-db)); out.append(point); cuts.append(point)
    return out,cuts

def clipped_submerged_polyhedron(mesh:TriangleMesh,normal_body:np.ndarray,offset:float)->tuple[float,np.ndarray]:
    norm=np.linalg.norm(normal_body)
    if not np.isfinite(norm) or norm<1e-12: raise PhysicalValidationError("Invalid water plane")
    n=normal_body/norm; offset=offset/norm; triangles=[]; cap_points=[]
    signed=mesh.vertices@n-offset
    if np.all(signed>=-1e-12): return polyhedron_volume_centroid(mesh.vertices[mesh.faces])
    if np.all(signed<-1e-12): return 0.0,np.zeros(3)
    for face in mesh.faces:
        poly,cuts=_clip_polygon([mesh.vertices[i] for i in face],n,offset)
        cap_points.extend(cuts)
        for i in range(1,len(poly)-1): triangles.append(np.array((poly[0],poly[i],poly[i+1])))
    unique=[]
    for p in cap_points:
        if not any(np.linalg.norm(p-q)<1e-9 for q in unique): unique.append(p)
    if len(unique)>=3:
        center=np.mean(unique,axis=0); helper=np.array([1.,0,0]) if abs(n[0])<.9 else np.array([0.,1,0])
        u=np.cross(n,helper); u/=np.linalg.norm(u); v=np.cross(n,u)
        ordered=sorted(unique,key=lambda p:math.atan2((p-center)@v,(p-center)@u))
        # Desired cap outward normal is -n.  u x v == n, so reverse.
        ordered=list(reversed(ordered))
        for i in range(1,len(ordered)-1): triangles.append(np.array((ordered[0],ordered[i],ordered[i+1])))
    return polyhedron_volume_centroid(np.asarray(triangles))

def quaternion_rotation(q:np.ndarray)->np.ndarray:
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

@dataclass(frozen=True)
class MeshBuoyancy:
    mesh:TriangleMesh
    water_density_kg_m3:float
    gravity_mps2:float
    water_level_ned_m:float
    heave_range_m:tuple[float,float]
    max_abs_roll_rad:float
    max_abs_pitch_rad:float
    model_name:str="mesh_buoyancy_exact"

    def validate(self,*,dtype:torch.dtype,device:torch.device)->None:
        self.mesh.validate()
        if not all(math.isfinite(x) for x in (self.water_density_kg_m3,self.gravity_mps2,self.water_level_ned_m,*self.heave_range_m,self.max_abs_roll_rad,self.max_abs_pitch_rad)) or self.water_density_kg_m3<=0 or self.gravity_mps2<=0 or self.heave_range_m[0]>=self.heave_range_m[1]:
            raise PhysicalValidationError("Invalid mesh buoyancy configuration")

    def geometry(self,state:VesselState)->tuple[float,np.ndarray]:
        position=state.position_ned.detach().cpu().numpy(); q=state.q_body_to_ned.detach().cpu().numpy(); rotation=quaternion_rotation(q)
        from bcod_sim.dynamics.restoring import quaternion_to_rpy
        rpy=quaternion_to_rpy(state.q_body_to_ned)
        if not self.heave_range_m[0]<=position[2]<=self.heave_range_m[1] or abs(float(rpy[0]))>self.max_abs_roll_rad or abs(float(rpy[1]))>self.max_abs_pitch_rad:
            raise OperatingEnvelopeError("Mesh buoyancy pose exceeds configured envelope")
        normal=rotation.T@np.array([0.,0.,1.]); offset=self.water_level_ned_m-position[2]
        return clipped_submerged_polyhedron(self.mesh,normal,offset)

    def evaluate(self,state:VesselState,mass_kg:float,cg_frd_m:torch.Tensor)->WrenchResult:
        volume,cb=self.geometry(state); down=state.nu_body.new_tensor((0.,0.,1.)); down_body=rotate_world_to_body(down,state.q_body_to_ned)
        weight=mass_kg*self.gravity_mps2*down_body; buoyancy=-self.water_density_kg_m3*self.gravity_mps2*volume*down_body
        cb_tensor=state.nu_body.new_tensor(cb); moment=torch.linalg.cross(cg_frd_m,weight)+torch.linalg.cross(cb_tensor,buoyancy)
        return WrenchResult(torch.cat((weight+buoyancy,moment)),self.model_name,{"submerged_volume_m3":volume,"center_of_buoyancy_frd_m":cb_tensor,"mesh_hash":self.mesh.content_hash})

def solve_equilibrium_waterline(model:MeshBuoyancy,mass_kg:float,*,tolerance_kg:float=1e-9,max_iterations:int=100)->dict[str,Any]:
    dtype=torch.float64
    def displaced(z:float)->tuple[float,np.ndarray]:
        state=VesselState(torch.tensor((0.,0.,z),dtype=dtype),torch.tensor((1.,0.,0.,0.),dtype=dtype),torch.zeros(6,dtype=dtype))
        volume,cb=model.geometry(state); return model.water_density_kg_m3*volume,cb
    lo,hi=model.heave_range_m; flo=displaced(lo)[0]-mass_kg; fhi=displaced(hi)[0]-mass_kg
    if flo*fhi>0: raise PhysicalValidationError("No mesh buoyancy equilibrium is bracketed by heave envelope")
    for _ in range(max_iterations):
        mid=(lo+hi)/2; displaced_mass,cb=displaced(mid); residual=displaced_mass-mass_kg
        if abs(residual)<=tolerance_kg: break
        if flo*residual<=0: hi=mid; fhi=residual
        else: lo=mid; flo=residual
    else: raise PhysicalValidationError("Mesh buoyancy equilibrium solver did not converge")
    return {"equilibrium_heave_ned_m":mid,"equilibrium_draft_m":mid-model.water_level_ned_m,"submerged_volume_m3":displaced_mass/model.water_density_kg_m3,"center_of_buoyancy_frd_m":cb.tolist(),"displaced_mass_kg":displaced_mass,"residual_kg":residual,"iterations":_+1}

def linearize_mesh_hydrostatics(model:MeshBuoyancy,mass_kg:float,cg_frd_m:torch.Tensor,equilibrium_position_ned_m:tuple[float,float,float],equilibrium_rpy_rad:tuple[float,float,float],*,position_step_m:float=1e-4,angle_step_rad:float=1e-4)->dict[str,Any]:
    from bcod_sim.frames.transforms import rpy_to_quaternion
    steps=np.array([position_step_m]*3+[angle_step_rad]*3); matrix=np.zeros((6,6)); residuals=[]
    def wrench(delta:np.ndarray)->np.ndarray:
        p=np.asarray(equilibrium_position_ned_m)+delta[:3]; rpy=np.asarray(equilibrium_rpy_rad)+delta[3:]
        state=VesselState(torch.tensor(p,dtype=torch.float64),torch.tensor(rpy_to_quaternion(*rpy),dtype=torch.float64),torch.zeros(6,dtype=torch.float64))
        return model.evaluate(state,mass_kg,cg_frd_m).tau_body.detach().numpy()
    for i,step in enumerate(steps):
        delta=np.zeros(6); delta[i]=step; plus=wrench(delta); minus=wrench(-delta)
        matrix[:,i]=-(plus-minus)/(2*step); residuals.append(float(np.max(np.abs(plus+minus-2*wrench(np.zeros(6))))))
    equilibrium_payload={"position_ned_m":equilibrium_position_ned_m,"orientation_rpy_rad":equilibrium_rpy_rad}
    import json
    return {"stiffness_6x6":matrix.tolist(),"perturbation_sizes":steps.tolist(),"central_difference_residuals":residuals,
            "geometry_hash":model.mesh.content_hash,"equilibrium_hash":hashlib.sha256(json.dumps(equilibrium_payload,sort_keys=True).encode()).hexdigest()}
