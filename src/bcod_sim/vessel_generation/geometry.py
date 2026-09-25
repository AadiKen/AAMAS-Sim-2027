"""Production geometry import and normalization for vessel generation."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil

import numpy as np

from .cfd import CFDExecutionError


@dataclass(frozen=True)
class ImportedGeometry:
    path: Path
    content_hash: str
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    source_frame: str
    normalized_frame: str = "FPU_m"


def import_ascii_stl(source: str|Path, destination: str|Path, *, source_frame: str) -> ImportedGeometry:
    if source_frame != "FPU_m":
        raise CFDExecutionError("ASCII STL import requires declared FPU_m frame (X forward, Y port, Z up)")
    source, destination = Path(source), Path(destination)
    data=source.read_bytes()
    if not data.startswith(b"solid ") or b"endsolid" not in data:
        raise CFDExecutionError("geometry importer requires an ASCII STL solid")
    text=data.decode("ascii")
    vertices=np.asarray([[float(x) for x in row] for row in re.findall(
        r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",text)],float)
    if len(vertices)<12 or len(vertices)%3 or not np.isfinite(vertices).all():
        raise CFDExecutionError("STL contains malformed or nonfinite triangles")
    def key(v): return tuple(np.round(v,12))
    edges={}
    for tri in vertices.reshape(-1,3,3):
        if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0])) <= 1e-14:
            raise CFDExecutionError("STL contains a degenerate triangle")
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edge=tuple(sorted((key(a),key(b)))); edges[edge]=edges.get(edge,0)+1
    if any(count!=2 for count in edges.values()):
        raise CFDExecutionError("STL is not a closed two-manifold surface")
    destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,destination)
    copied=destination.read_bytes()
    return ImportedGeometry(destination,hashlib.sha256(copied).hexdigest(),
        (tuple(vertices.min(axis=0)),tuple(vertices.max(axis=0))),source_frame)
