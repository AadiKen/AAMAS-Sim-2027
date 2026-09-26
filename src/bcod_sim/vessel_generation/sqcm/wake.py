"""Experimental time-marched free-vortex deformation for ship SQCM."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .source import SourcePanels, source_velocity
from .vortex import Horseshoe, horseshoe_velocity
from .coupled_solver import CoupledResult, solve_coupled


@dataclass
class WakeResult:
    result: CoupledResult
    horseshoes: list[Horseshoe]
    history: list[dict]
    converged: bool


def _convect_nodes(panels: SourcePanels, horseshoes: list[Horseshoe],
                   source_strength: np.ndarray, circulation: np.ndarray,
                   ambient: np.ndarray, dt: float, core_radius: float,
                   wake_elements: int, nominal_segment: float) -> list[Horseshoe]:
    rows=[]
    for h in horseshoes:
        rows.append(h.leg_start[h.leg_start_free_index:])
        rows.append(h.leg_end[h.leg_end_free_index:])
    points=np.concatenate(rows)
    induced_source=np.einsum('ijk,j->ik',source_velocity(points,panels),source_strength)
    induced_vortex=np.einsum('ijk,j->ik',horseshoe_velocity(points,horseshoes,
                                                            core_radius=core_radius),circulation)
    updated=points+dt*(ambient+induced_source+induced_vortex)
    if not np.isfinite(updated).all():
        raise ValueError('Non-finite wake node')
    new=[];offset=0
    for h in horseshoes:
        legs=[]
        for old,free_index in ((h.leg_start,h.leg_start_free_index),
                               (h.leg_end,h.leg_end_free_index)):
            moved=updated[offset:offset+wake_elements]
            offset+=wake_elements
            prefix=old[:free_index]
            root=prefix[-1]
            # Insert a root-adjacent node only when the gap exceeds the
            # nominal element length (thesis §4.7), then retire the far end.
            gap=float(np.linalg.norm(moved[0]-root))
            if gap>nominal_segment:
                first=root+(nominal_segment/gap)*(moved[0]-root)
                leg=np.vstack((prefix,first,moved[:-1]))
            else:
                leg=np.vstack((prefix,moved))
            legs.append(leg)
        new.append(Horseshoe(h.bound_start,h.bound_end,legs[0],legs[1],
                             h.weight,h.hull_id,h.model,
                             h.leg_start_free_index,h.leg_end_free_index))
    return new


def iterate_wake(panels: SourcePanels, physical_mask: np.ndarray,
                 horseshoes: list[Horseshoe], vortex_points: np.ndarray,
                 vortex_normals: np.ndarray, ambient: np.ndarray,
                 *, density: float=1025., core_radius: float=.001,
                 max_iterations: int=100, minimum_iterations: int=8,
                 force_tolerance: float=.01, time_step_cfl: float=.1) -> WakeResult:
    """Eq. 4.41 time march with a bounded number of free-vortex elements."""
    if not horseshoes or max_iterations<1:
        raise ValueError('No wake to iterate')
    n=len(horseshoes[0].leg_start)-horseshoes[0].leg_start_free_index
    length=max(np.linalg.norm(h.leg_start[-1]-h.leg_start[h.leg_start_free_index-1])
               for h in horseshoes)
    nominal=length/n
    if not 0.<time_step_cfl<=1.:
        raise ValueError('Invalid wake time-step CFL')
    dt=time_step_cfl*nominal/np.linalg.norm(ambient)
    history=[];previous=None;converged=False
    for iteration in range(max_iterations):
        current=solve_coupled(panels,physical_mask,horseshoes,vortex_points,
                              vortex_normals,ambient,density=density,
                              core_radius=core_radius)
        yn=np.array([current.force[1],current.moment[2]])
        change=None if previous is None else float(np.max(np.abs(yn-previous)/
                                                         np.maximum(abs(yn),1e-9)))
        history.append({'iteration':iteration,'Y_n':float(yn[0]),'N_nm':float(yn[1]),
                        'relative_force_change':change,
                        'condition_number':current.condition_number,
                        'maximum_hull_residual':float(np.max(abs(current.normal_residual))),
                        'maximum_vortex_plane_residual':float(np.max(abs(current.vortex_plane_residual)))})
        if iteration>=minimum_iterations and change is not None and change<force_tolerance:
            # Require several consistent steps, not a one-step coincidence.
            tail=[item['relative_force_change'] for item in history[-4:]]
            if all(value is not None and value<force_tolerance for value in tail):
                converged=True
                break
        previous=yn
        horseshoes=_convect_nodes(panels,horseshoes,current.source_strength,
                                 current.circulation_strength,ambient,dt,core_radius,n,nominal)
    return WakeResult(current,horseshoes,history,converged)
