"""Connected-component audit of the pre-v3 aft source discrepancy."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from qualify_surveyor_envelopes import _surface_distances,_to_frd

BASE=ROOT/'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'
SOURCE=ROOT/'docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/source/mixed_low_faces_source_mm.stl'
OLD=BASE/'provenance/v2_targeted_conformal_nominal.stl'


def one_side(source:trimesh.Trimesh,mesh:trimesh.Trimesh,sign:int)->dict:
    side=source.submesh([np.where(np.sign(source.triangles_center[:,0])==sign)[0]],
                        append=True,repair=False)
    centers=side.triangles_center
    dist=_surface_distances(_to_frd(centers),mesh,128)*1000
    area=side.area_faces
    aft=(centers[:,2]>=5)&(centers[:,2]<95)&(centers[:,1]<=-75)
    bad=aft&(dist>5)
    indices=np.where(bad)[0]
    lookup=np.full(len(bad),-1,dtype=int);lookup[indices]=np.arange(len(indices))
    adjacency=side.face_adjacency
    edges=adjacency[bad[adjacency].all(axis=1)]
    matrix=coo_matrix((np.ones(len(edges)*2),
        (np.r_[lookup[edges[:,0]],lookup[edges[:,1]]],
         np.r_[lookup[edges[:,1]],lookup[edges[:,0]]])),
        shape=(len(indices),len(indices))).tocsr()
    count,labels=connected_components(matrix)
    components=[]
    for number in range(count):
        ids=indices[labels==number]
        points=side.triangles[ids].reshape(-1,3)
        components.append({'triangle_count':int(len(ids)),
                           'area_m2':float(area[ids].sum()/1e6),
                           'source_xyz_bounds_mm':[points.min(0).tolist(),points.max(0).tolist()],
                           'maximum_distance_mm':float(dist[ids].max()),
                           'mean_source_normal_xyz':side.face_normals[ids].mean(0).tolist()})
    components.sort(key=lambda row:row['area_m2'],reverse=True)
    face_angles=np.degrees(side.face_adjacency_angles)
    local_edges=side.face_adjacency[(aft[side.face_adjacency]).all(axis=1)]
    local_angles=face_angles[(aft[side.face_adjacency]).all(axis=1)]
    unique_edges,edge_counts=np.unique(side.edges_sorted,axis=0,return_counts=True)
    midpoints=side.vertices[unique_edges].mean(axis=1)
    boundary=(edge_counts==1)&(midpoints[:,2]>=5)&(midpoints[:,2]<95)
    nonmanifold=(edge_counts>2)&(midpoints[:,2]>=5)&(midpoints[:,2]<95)
    triangles=side.triangles[aft]
    lengths=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2)
    sliver=(area[aft]/np.maximum(lengths.max(axis=1)**2,1e-12))<1e-4
    return {'aft_mask_source_area_m2':float(area[aft].sum()/1e6),
            'aft_area_over_5mm_m2':float(area[bad].sum()/1e6),
            'maximum_distance_mm':float(dist[aft].max()),
            'component_count':len(components),'components':components,
            'local_face_dihedral_p95_deg':float(np.percentile(local_angles,95)),
            'local_face_dihedral_max_deg':float(local_angles.max()),
            'open_boundary_edge_count_source_shell':int(boundary.sum()),
            'nonmanifold_edge_count_source_shell':int(nonmanifold.sum()),
            'thin_tessellation_triangle_count':int(sliver.sum()),
            'triangle_count_in_mask':int(aft.sum())}


def run()->None:
    source=trimesh.load_mesh(SOURCE,process=True)
    mesh=trimesh.load_mesh(OLD,process=True)
    data={'schema':'surveyor-v3-aft-defect-audit-1',
          'method':'source selected-face triangle centroids to conformally remeshed v2 STEP; 128 closest triangle candidates',
          'port':one_side(source,mesh,-1),
          'starboard':one_side(source,mesh,1),
          'section_evidence':'Original shell has two principal open section chains at Z=18.8 and 23–95 mm and four disconnected chains at Z=19–22 mm.',
          'diagnosis':'The pre-v3 scaled cap terminates too narrowly at the aft source tip; source surface topology is incomplete/disconnected at Z=19–22 mm. Side outliers farther forward in the coarse v2 STL were tessellation artifacts.'}
    (BASE/'aft_defect_diagnostics.json').write_text(json.dumps(data,indent=2)+'\n')
    print('port',data['port']['aft_area_over_5mm_m2'],data['port']['maximum_distance_mm'],
          'starboard',data['starboard']['aft_area_over_5mm_m2'],data['starboard']['maximum_distance_mm'],flush=True)


if __name__=='__main__':run()
