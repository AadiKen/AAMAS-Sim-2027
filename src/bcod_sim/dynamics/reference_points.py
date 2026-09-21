"""Authoritative spatial reference-point transformations in body FRD."""
import torch
from bcod_sim.core.errors import PhysicalValidationError
def skew(v:torch.Tensor)->torch.Tensor:
    x,y,z=v.unbind(); zero=x.new_zeros(())
    return torch.stack((torch.stack((zero,-z,y)),torch.stack((z,zero,-x)),torch.stack((-y,x,zero))))

def spatial_motion_transform(offset_from_new_to_old_frd_m:torch.Tensor)->torch.Tensor:
    if offset_from_new_to_old_frd_m.shape!=(3,) or not torch.isfinite(offset_from_new_to_old_frd_m).all().item():
        raise PhysicalValidationError("Reference-point offset must be a finite FRD 3-vector")
    eye=torch.eye(3,dtype=offset_from_new_to_old_frd_m.dtype,device=offset_from_new_to_old_frd_m.device)
    s=skew(offset_from_new_to_old_frd_m)
    return torch.cat((torch.cat((eye,s.T),1),torch.cat((torch.zeros_like(eye),eye),1)),0)

def transform_spatial_matrix(matrix:torch.Tensor,offset_from_new_to_old_frd_m:torch.Tensor)->torch.Tensor:
    if matrix.shape!=(6,6) or not torch.isfinite(matrix).all().item(): raise PhysicalValidationError("Spatial matrix must be finite 6x6")
    h=spatial_motion_transform(offset_from_new_to_old_frd_m); return h.T@matrix@h

def transform_added_mass(matrix:torch.Tensor,offset_from_new_to_old_frd_m:torch.Tensor)->torch.Tensor:
    return transform_spatial_matrix(matrix,offset_from_new_to_old_frd_m)

def transform_hydrostatic_stiffness(matrix:torch.Tensor,offset_from_new_to_old_frd_m:torch.Tensor)->torch.Tensor:
    return transform_spatial_matrix(matrix,offset_from_new_to_old_frd_m)

def translate_wrench(tau_at_old:torch.Tensor,offset_from_new_to_old_frd_m:torch.Tensor)->torch.Tensor:
    if tau_at_old.shape!=(6,) or not torch.isfinite(tau_at_old).all().item(): raise PhysicalValidationError("Wrench must be finite 6-vector")
    force=tau_at_old[:3]; return torch.cat((force,tau_at_old[3:]+torch.linalg.cross(offset_from_new_to_old_frd_m,force)))

def transform_point(point_at_old:torch.Tensor,offset_from_new_to_old_frd_m:torch.Tensor)->torch.Tensor:
    return point_at_old+offset_from_new_to_old_frd_m

def parallel_axis_inertia(inertia_at_cg:torch.Tensor,mass_kg:float,offset_cg_from_origin:torch.Tensor)->torch.Tensor:
    if inertia_at_cg.shape!=(3,3) or mass_kg<=0: raise PhysicalValidationError("Invalid inertia transform inputs")
    s=skew(offset_cg_from_origin); return inertia_at_cg-mass_kg*s@s
