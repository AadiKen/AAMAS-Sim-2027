"""Experimental deterministic mesh-to-structured-double-body reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import trimesh

from bcod_sim.vessel_generation.simple_sections import section_properties, polygon_properties
from bcod_sim.vessel_generation.simple_crossflow import _clip_section
from bcod_sim.vessel_generation.volume_clip import submerged_volume_centroid
from .source import SourcePanels


@dataclass(frozen=True)
class StructuredHull:
    panels: SourcePanels
    physical_mask: np.ndarray
    hull_ids: np.ndarray
    component_properties: list[dict]
    preservation: dict


def _side_at_depth(contour: np.ndarray, z: float) -> tuple[float,float]:
    hits=[]
    for a,b in zip(contour,np.roll(contour,-1,axis=0)):
        if min(a[2],b[2])-1e-10 <= z <= max(a[2],b[2])+1e-10 and abs(b[2]-a[2])>1e-12:
            fraction=(z-a[2])/(b[2]-a[2])
            hits.append(float(a[1]+fraction*(b[1]-a[1])))
    if len(hits)<2:
        # At a smooth keel, both sides meet at the lowest sampled vertex.
        bottom=contour[np.argmax(contour[:,2])]
        return float(bottom[1]),float(bottom[1])
    return min(hits),max(hits)


def reconstruct(mesh: trimesh.Trimesh, waterline: float, *, longitudinal: int=30,
                vertical: int=5, volume_tolerance: float=.05,
                wetted_tolerance: float=.10) -> StructuredHull:
    """Build separate component strips; reject rather than alter the source mesh."""
    if not mesh.is_watertight or longitudinal<5 or vertical<2:
        raise ValueError("SQCM requires watertight hull and adequate panel count")
    components=mesh.split(only_watertight=True)
    quads=[]; mask=[]; hull_ids=[]; props=[]; section_measurements=[]
    for hull_id,component in enumerate(components):
        length=float(component.extents[0]); inset=length*1e-5
        trial=np.linspace(component.bounds[0,0]+inset,component.bounds[1,0]-inset,101)
        wet=[]
        for xi in trial:
            loops=section_properties(component,0,float(xi))["loops"]
            area=sum(polygon_properties(_clip_section(loop,waterline),1,2)[0]
                     for loop in loops if len(_clip_section(loop,waterline))>=3)
            wet.append(area>1e-9)
        indices=np.flatnonzero(wet)
        if len(indices)<3:
            raise ValueError(f"No usable submerged station range for hull {hull_id}")
        x=np.linspace(trial[indices[0]],trial[indices[-1]],longitudinal+1)
        nodes=np.empty((longitudinal+1,vertical+1,4))
        for i,xi in enumerate(x):
            loops=section_properties(component,0,float(xi))["loops"]
            clipped=[_clip_section(loop,waterline) for loop in loops]
            clipped=[loop for loop in clipped if len(loop)>=3]
            if not clipped:
                raise ValueError(f"No submerged contour at hull {hull_id}, x={xi:g}")
            contour=max(clipped,key=lambda loop:polygon_properties(loop,1,2)[0])
            keel=float(np.max(contour[:,2]));
            if keel<=waterline:
                raise ValueError("Invalid submerged contour depth")
            section_area=polygon_properties(contour,1,2)[0]
            section_measurements.append((float(xi),section_area))
            for j,t in enumerate(np.linspace(0.,1.,vertical+1)):
                z=waterline+t*(keel-waterline)
                lo,hi=_side_at_depth(contour,z)
                nodes[i,j]=[xi,lo,hi,z]
            if np.any(nodes[i,:,2]<nodes[i,:,1]-1e-9):
                raise ValueError("Invalid section ordering")
        props.append({'hull_id':hull_id,'x_stern':float(x[0]),'x_bow':float(x[-1]),
                      'y_center':float(component.centroid[1]),
                      'z_waterline':float(waterline),'z_keel':float(component.bounds[1,2]),
                      'longitudinal':longitudinal,'vertical':vertical})
        for mirror in (1.,-1.):
            for side in (0,1):
                for i in range(longitudinal):
                    for j in range(vertical):
                        indices=((i,j),(i+1,j),(i+1,j+1),(i,j+1))
                        quad=np.zeros((4,3))
                        for k,(ii,jj) in enumerate(indices):
                            xi,lo,hi,z=nodes[ii,jj]
                            quad[k]=[xi,lo if side==0 else hi,
                                     z if mirror>0 else 2*waterline-z]
                        orient=np.cross(quad[1]-quad[0],quad[2]-quad[0])
                        if orient[1]*(-1 if side==0 else 1)<0:
                            quad=quad[::-1]
                        if np.linalg.norm(np.cross(quad[1]-quad[0],quad[2]-quad[0]))<1e-14:
                            raise ValueError("Degenerate reconstructed SQCM panel")
                        quads.append(quad);mask.append(mirror>0);hull_ids.append(hull_id)
            # A hard or flat keel has a finite span between port/starboard
            # contour endpoints; omitting it loses much of a KCS wetted area.
            for i in range(longitudinal):
                left,right=nodes[i,-1],nodes[i+1,-1]
                quad=np.array([[left[0],left[1],left[3]],
                               [left[0],left[2],left[3]],
                               [right[0],right[2],right[3]],
                               [right[0],right[1],right[3]]],dtype=float)
                if mirror<0:
                    quad[:,2]=2*waterline-quad[:,2]
                normal=np.cross(quad[1]-quad[0],quad[2]-quad[0])
                if np.linalg.norm(normal)<1e-14:
                    continue
                if normal[2]*mirror<0:
                    quad=quad[::-1]
                quads.append(quad);mask.append(mirror>0);hull_ids.append(hull_id)
            # Close the truncated wetted station network at both x ends.
            for end_index,end_sign in ((0,-1.),(-1,1.)):
                for j in range(vertical):
                    lo0,lo1=nodes[end_index,j],nodes[end_index,j+1]
                    quad=np.array([[lo0[0],lo0[1],lo0[3]],
                                   [lo1[0],lo1[1],lo1[3]],
                                   [lo1[0],lo1[2],lo1[3]],
                                   [lo0[0],lo0[2],lo0[3]]],dtype=float)
                    if mirror<0:
                        quad[:,2]=2*waterline-quad[:,2]
                    normal=np.cross(quad[1]-quad[0],quad[2]-quad[0])
                    if np.linalg.norm(normal)<1e-14:
                        continue
                    if normal[0]*end_sign<0:
                        quad=quad[::-1]
                    quads.append(quad);mask.append(mirror>0);hull_ids.append(hull_id)
    panels=SourcePanels.from_corners(np.asarray(quads))
    physical=np.asarray(mask);hull_ids=np.asarray(hull_ids)
    source_volume,source_cb=submerged_volume_centroid(mesh,waterline)
    # Cross-section trapz checks displacement independent of panel orientation.
    xs,areas=np.array(section_measurements).T
    reconstructed_volume=sum(float(np.trapezoid(areas[k*(longitudinal+1):(k+1)*(longitudinal+1)],
                                               xs[k*(longitudinal+1):(k+1)*(longitudinal+1)]))
                             for k in range(len(components)))
    reconstructed_wetted=float(panels.areas[physical].sum())
    # Exact clipped wetted area from the original normalized hull.
    from bcod_sim.vessel_generation.simple_sections import hydrostatic_state
    source_wetted=float(hydrostatic_state(mesh,waterline,include_waterplane=False)['wetted_area_m2'])
    volume_error=abs(reconstructed_volume-source_volume)/source_volume
    wetted_error=abs(reconstructed_wetted-source_wetted)/source_wetted
    preservation={'source_volume_m3':source_volume,'reconstructed_volume_m3':reconstructed_volume,
                  'relative_volume_error':volume_error,'source_wetted_area_m2':source_wetted,
                  'reconstructed_wetted_area_m2':reconstructed_wetted,
                  'relative_wetted_area_error':wetted_error,'source_cb_frd_m':source_cb.tolist(),
                  'component_count':len(components),
                  'accepted':volume_error<=volume_tolerance and wetted_error<=wetted_tolerance}
    if not preservation['accepted']:
        raise ValueError(f"SQCM geometry-preservation gate failed: {preservation}")
    return StructuredHull(panels,physical,hull_ids,props,preservation)
