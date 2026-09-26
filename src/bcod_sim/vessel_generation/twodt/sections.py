"""Deterministic mesh-section contours and compact representative families."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import trimesh

from bcod_sim.vessel_generation.simple_crossflow import _clip_section
from bcod_sim.vessel_generation.simple_sections import section_properties, polygon_properties


@dataclass(frozen=True)
class TwoDtSection:
    hull_id: int
    x_m: float
    contour_yz_m: np.ndarray
    draft_m: float
    beam_m: float
    area_m2: float
    perimeter_m: float
    centroid_yz_m: tuple[float, float]
    fullness: float
    beam_draft_ratio: float
    sharp_turn_fraction: float

    @property
    def descriptor(self) -> np.ndarray:
        # Dimensionless descriptors; no experimental load data.
        return np.array([np.log(max(self.beam_draft_ratio, 1e-6)), self.fullness,
                         self.perimeter_m/max(2*(self.beam_m+self.draft_m), 1e-9),
                         self.sharp_turn_fraction])


def _resample_closed(contour_yz: np.ndarray, count: int) -> np.ndarray:
    points=np.asarray(contour_yz,dtype=float)
    if np.linalg.norm(points[0]-points[-1])<1e-10:
        points=points[:-1]
    closed=np.vstack((points,points[0]))
    distance=np.linalg.norm(np.diff(closed,axis=0),axis=1)
    cumulative=np.r_[0.,np.cumsum(distance)]
    if cumulative[-1]<=0:
        raise ValueError('Degenerate section perimeter')
    sample=np.linspace(0.,cumulative[-1],count,endpoint=False)
    return np.column_stack([np.interp(sample,cumulative,closed[:,axis]) for axis in (0,1)])


def extract_section_contours(mesh: trimesh.Trimesh, waterline: float,
                             *, count: int=31, contour_points: int=64) -> list[TwoDtSection]:
    if not mesh.is_watertight or count<3 or contour_points<16:
        raise ValueError('Invalid 2D+t section request')
    result=[]
    for hull_id,component in enumerate(mesh.split(only_watertight=True)):
        dx=float(component.extents[0])/count
        for x in np.linspace(component.bounds[0,0]+dx/2,component.bounds[1,0]-dx/2,count):
            loops=section_properties(component,0,float(x))['loops']
            for loop in loops:
                clipped=_clip_section(loop,waterline)
                if len(clipped)<3:
                    continue
                area,cy,cz,_,_=polygon_properties(clipped,1,2)
                beam=float(np.ptp(clipped[:,1]));draft=float(np.ptp(clipped[:,2]))
                if min(area,beam,draft)<=1e-9:
                    continue
                contour=_resample_closed(clipped[:,1:3],contour_points)
                edge=np.roll(contour,-1,axis=0)-contour
                vectors=edge/np.maximum(np.linalg.norm(edge,axis=1)[:,None],1e-12)
                turns=np.arccos(np.clip(np.sum(vectors*np.roll(vectors,1,axis=0),axis=1),-1.,1.))
                result.append(TwoDtSection(hull_id,float(x),contour,draft,beam,float(area),
                                            float(np.linalg.norm(edge,axis=1).sum()),
                                            (float(cy),float(cz)),float(np.clip(area/(beam*draft),0.,1.)),
                                            beam/draft,float(np.mean(turns>np.pi/6))))
    if len(result)<3:
        raise ValueError('No usable submerged sections')
    return result


def representative_sections(sections: list[TwoDtSection], maximum: int=6) -> dict:
    """Deterministic farthest-point medoids within each hull, no force fitting."""
    if maximum<1 or not sections:
        raise ValueError('Invalid section family request')
    families=[];assignment=np.full(len(sections),-1,dtype=int)
    for hull_id in sorted({s.hull_id for s in sections}):
        ids=np.array([i for i,s in enumerate(sections) if s.hull_id==hull_id],dtype=int)
        descriptors=np.array([sections[i].descriptor for i in ids])
        scale=np.maximum(np.ptp(descriptors,axis=0),np.array([.1,.1,.1,.05]))
        normalized=(descriptors-descriptors.mean(axis=0))/scale
        centroid=normalized.mean(axis=0)
        selected=[int(np.argmin(np.linalg.norm(normalized-centroid,axis=1)))]
        while len(selected)<min(maximum,len(ids)):
            distance=np.min(np.linalg.norm(normalized[:,None,:]-normalized[selected][None,:,:],axis=2),axis=1)
            distance[selected]=-1
            selected.append(int(np.argmax(distance)))
        distances=np.linalg.norm(normalized[:,None,:]-normalized[selected][None,:,:],axis=2)
        labels=np.argmin(distances,axis=1)
        for local,global_id in enumerate(ids):
            assignment[global_id]=len(families)+int(labels[local])
        families.extend([int(ids[j]) for j in selected])
    return {'representative_indices':families,'section_to_family':assignment.tolist(),
            'family_count':len(families),'method':'dimensionless_farthest_point_medoids_per_hull'}
