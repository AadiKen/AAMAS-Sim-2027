"""Experimental simultaneous source/QCM system; not a production load path."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .source import SourcePanels, source_velocity
from .vortex import Horseshoe, horseshoe_velocity, qcm_control_positions


def mirror_horseshoes(horseshoes: list[Horseshoe]) -> list[Horseshoe]:
    """Waterline-image system; reflection changes circulation orientation."""
    reflected=[]
    for h in horseshoes:
        transform=lambda points: np.asarray(points)*np.array([1.,1.,-1.])
        reflected.append(Horseshoe(transform(h.bound_start),transform(h.bound_end),
                                   transform(h.leg_start),transform(h.leg_end),
                                   -h.weight,h.hull_id,h.model,
                                   h.leg_start_free_index,h.leg_end_free_index))
    return reflected


def vortex_collocation(horseshoes: list[Horseshoe], *, x_stern: float,
                       x_bow: float, longitudinal: int) -> tuple[np.ndarray,np.ndarray]:
    control=qcm_control_positions(x_stern,x_bow,longitudinal)
    points=[]
    for i,h in enumerate(horseshoes):
        p=(h.bound_start+h.bound_end)/2
        p[0]=control[i%longitudinal]
        points.append(p)
    points=np.asarray(points)
    normals=np.tile(np.array([0.,1.,0.]),(len(points),1))
    return points,normals


@dataclass
class CoupledResult:
    source_strength: np.ndarray
    circulation_strength: np.ndarray
    surface_velocity: np.ndarray
    cp: np.ndarray
    panel_force: np.ndarray
    panel_moment: np.ndarray
    normal_residual: np.ndarray
    vortex_plane_residual: np.ndarray
    condition_number: float
    force: np.ndarray
    moment: np.ndarray


def solve_coupled(panels: SourcePanels, physical_mask: np.ndarray,
                  horseshoes: list[Horseshoe], vortex_points: np.ndarray,
                  vortex_normals: np.ndarray, flow: np.ndarray,
                  *, density: float=1025., reference: np.ndarray | None=None,
                  source_order: int=4, core_radius: float=0.001,
                  condition_limit: float=1e9) -> CoupledResult:
    """Solve thesis Eq. 4.40 for a prescribed fixed wake geometry."""
    flow=np.asarray(flow,dtype=float)
    if flow.shape!=(3,) or (flow@flow)<=0:
        raise ValueError("Forward-flow vector required")
    m,n=len(panels.corners),len(horseshoes)
    if physical_mask.shape!=(m,) or vortex_points.shape!=(n,3) or vortex_normals.shape!=(n,3):
        raise ValueError("Coupled geometry dimensions do not match")
    targets=np.vstack((panels.centers,vortex_points))
    normals=np.vstack((panels.normals,vortex_normals))
    source=source_velocity(targets,panels,order=source_order)
    source[np.arange(m),np.arange(m)]=.5*panels.normals
    vortex=horseshoe_velocity(targets,horseshoes,core_radius=core_radius)
    matrix=np.column_stack((np.einsum('ijk,ik->ij',source,normals),
                            np.einsum('ijk,ik->ij',vortex,normals)))
    condition=float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition>condition_limit:
        raise ValueError(f"Coupled SQCM matrix ill-conditioned: {condition:g}")
    strength=lu_solve(lu_factor(matrix),-normals@flow)
    velocity=flow+np.einsum('ijk,j->ik',source,strength[:m])+np.einsum('ijk,j->ik',vortex,strength[m:])
    residual=np.einsum('ik,ik->i',velocity,normals)
    cp=1.-np.sum(velocity[:m]**2,axis=1)/(flow@flow)
    pressure=.5*density*(flow@flow)*cp
    panel_force=-pressure[:,None]*panels.normals*panels.areas[:,None]
    reference=np.zeros(3) if reference is None else np.asarray(reference,dtype=float)
    panel_moment=np.cross(panels.centers-reference,panel_force)
    return CoupledResult(strength[:m],strength[m:],velocity[:m],cp,panel_force,panel_moment,
                         residual[:m],residual[m:],condition,
                         panel_force[physical_mask].sum(axis=0),panel_moment[physical_mask].sum(axis=0))
