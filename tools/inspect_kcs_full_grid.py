"""Inventory boundaries in NMRI KCS bow2/stn2 above-water surface grids."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import trimesh


def zones(path: Path) -> list[np.ndarray]:
    text=path.read_text()
    matches=list(re.finditer(r"Zone\s+i=\s*(\d+)\s+j=\s*(\d+)\s+f=point",text,re.I))
    result=[]
    for k,match in enumerate(matches):
        body=text[match.end():matches[k+1].start() if k+1<len(matches) else None]
        values=np.asarray([[float(x) for x in line.split()] for line in body.splitlines()
                           if len(line.split())==3],float)
        ni,nj=map(int,match.groups())
        if values.shape!=(ni*nj,3): raise ValueError(f"incomplete zone in {path}")
        result.append(values.reshape(nj,ni,3))
    return result


def inspect(directory: Path) -> dict:
    vertices=[];faces=[];patches=[]
    for name in ("kcs_bow2.dat","kcs_stn2.dat"):
        for zone_id,array in enumerate(zones(directory/name)):
            nj,ni=array.shape[:2];offset=len(vertices)
            vertices.extend(array.reshape(-1,3))
            for j in range(nj-1):
                for i in range(ni-1):
                    a=offset+j*ni+i;b=a+1;c=a+ni+1;d=a+ni
                    faces.extend(((a,b,c),(a,c,d)))
            patches.append({"file":name,"zone":zone_id,"shape":[nj,ni],
                            "bbox":np.stack((array.min((0,1)),array.max((0,1)))).tolist()})
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    mesh.merge_vertices(digits_vertex=6)
    count=np.bincount(mesh.edges_unique_inverse)
    open_edges=mesh.edges_unique[count==1]
    points=mesh.vertices[open_edges]
    classification={"above_water_both_endpoints":int(np.sum(np.all(points[:,:,2]>1e-6,axis=1))),
                    "at_or_below_water_any_endpoint":int(np.sum(np.any(points[:,:,2]<=1e-6,axis=1)))}
    return {"source":"NMRI official *_bow2/stn2.dat grids (full profile including freeboard)",
            "patches":patches,"vertices_after_welding":len(mesh.vertices),"faces":len(mesh.faces),
            "open_edge_count":len(open_edges),"open_edge_classes":classification,
            "open_edge_bbox":np.stack((points.min((0,1)),points.max((0,1)))).tolist(),
            "note":"This is an unclosed source-surface inventory. Open edges include separate panel boundaries, deck and end closures; no CFD shell is inferred."}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("grid_dir",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();report=inspect(args.grid_dir)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("patches",)},indent=2))
