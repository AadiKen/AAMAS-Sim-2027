"""OpenFOAM Foundation 11 rigid steady-yaw region with non-conformal coupling.

The syntax follows the installed v11 incompressibleVoF/mixerVessel tutorial:
snappy cellZone/faceZone, createBaffles, splitBaffles,
createNonConformalCouples, and a solidBody fvMesh mover.
"""
from __future__ import annotations

import math
import re
import struct
from pathlib import Path

import numpy as np

from .frame_contract import point_to_foam


def _vertices(stl: bytes) -> np.ndarray:
    if len(stl)>=84 and len(stl)==84+50*struct.unpack_from("<I",stl,80)[0]:
        values=[]
        for offset in range(84,len(stl),50):
            for vertex in range(3):
                values.append(struct.unpack_from("<3f",stl,offset+12+vertex*12))
        return np.asarray(values,float)
    values=re.findall(rb"\bvertex\s+([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)",stl)
    if not values: raise ValueError("rotating-region geometry has no STL vertices")
    return np.asarray(values,float)


def cylinder_stl(center: tuple[float,float,float], radius: float,
                 bottom: float, top: float, *, segments: int=48) -> str:
    """Closed triangulated cylindrical selector for snappy's cell/face zones."""
    cx,cy,_=center;lines=["solid rotatingCylinder"]
    for index in range(segments):
        first=2*math.pi*index/segments;second=2*math.pi*(index+1)/segments
        a=(cx+radius*math.cos(first),cy+radius*math.sin(first),bottom)
        b=(cx+radius*math.cos(second),cy+radius*math.sin(second),bottom)
        A=(a[0],a[1],top);B=(b[0],b[1],top)
        for triangle in ((a,b,B),(a,B,A),((cx,cy,bottom),b,a),((cx,cy,top),A,B)):
            lines.extend(("facet normal 0 0 0","outer loop"))
            lines.extend(f"vertex {x:.12g} {y:.12g} {z:.12g}" for x,y,z in triangle)
            lines.extend(("endloop","endfacet"))
    return "\n".join((*lines,"endsolid rotatingCylinder"))+"\n"


def _patches(field: str, entries: str) -> str:
    start=field.index("boundaryField");opening=field.index("{",start);depth=0
    for index in range(opening,len(field)):
        depth+=(field[index]=="{")-(field[index]=="}")
        if depth==0:
            return (field[:opening+1]+'\n#includeEtc "caseDicts/setConstraintTypes"\n'
                    +field[opening+1:index]+"\n"+entries+"\n"+field[index:])
    raise ValueError("field boundaryField is incomplete")


def configure_steady_yaw(files: dict[str,str], case, shifted_stl: bytes) -> tuple[dict[str,str],dict]:
    """Add a rigid cylindrical cell zone and v11 NCC sliding interface."""
    if case.dof.value!="yaw":
        raise ValueError("rigid rotating-region topology currently supports yaw only")
    files=dict(files);vertices=_vertices(shifted_stl)
    origin=np.asarray(point_to_foam(case.reference_point_frd_m,case.waterline_z_m),float)
    cell=float(case.mesh_settings.base_cell_size_m)
    radius=float(np.max(np.linalg.norm(vertices[:,:2]-origin[:2],axis=1))+2*cell/3)
    bottom=float(vertices[:,2].min()-cell);top=float(vertices[:,2].max()+cell)
    limits=np.asarray([point_to_foam(v,case.waterline_z_m)
        for v in (case.domain_settings.minimum_frd_m,case.domain_settings.maximum_frd_m)],float)
    low=limits.min(axis=0);high=limits.max(axis=0)
    if (origin[0]-radius<=low[0]+cell/2 or origin[0]+radius>=high[0]-cell/2 or
        origin[1]-radius<=low[1]+cell/2 or origin[1]+radius>=high[1]-cell/2 or
        bottom<=low[2]+cell/4 or top>=high[2]-cell/4):
        raise ValueError("rotating cylinder does not fit within the stationary domain")
    inside=(origin[0]+radius-.4*cell,origin[1],origin[2])
    snappy=files["system/snappyHexMeshDict"]
    snappy=snappy.replace("geometry { hull.stl {type triSurfaceMesh; name hull;} }",
        "geometry { hull.stl {type triSurfaceMesh; name hull;} "
        "rotating.stl {type triSurfaceMesh; name rotating;} }")
    original="refinementSurfaces {hull {level ("
    if original not in snappy: raise ValueError("unrecognised snappy hull refinement dictionary")
    snappy=snappy.replace("patchInfo {type wall;}}}",
        ("patchInfo {type wall;}} rotating {level (1 1); cellZone rotating; "
         "faceZone rotating; mode insidePoint; insidePoint "
         f"({inside[0]} {inside[1]} {inside[2]});}}}}"),1)
    snappy=snappy.replace("allowFreeStandingZoneFaces true","allowFreeStandingZoneFaces false")
    files["system/snappyHexMeshDict"]=snappy
    files["constant/triSurface/rotating.stl"]=cylinder_stl(tuple(origin),radius,bottom,top)
    files["system/createBafflesDict"]=(
        "FoamFile {version 2.0; format ascii; class dictionary; object createBafflesDict;}\n"
        "internalFacesOnly true;\n"
        "baffles {nonCouple {type faceZone; zoneName rotating; "
        "owner {name nonCouple1; type patch;} "
        "neighbour {name nonCouple2; type patch;}}}\n")
    rpm=float(case.magnitude)*60/(2*math.pi)
    files["constant/dynamicMeshDict"]=(
        "FoamFile {version 2.0; format ascii; class dictionary; location \"constant\"; "
        "object dynamicMeshDict;}\n"
        "mover {type motionSolver; libs (\"libfvMeshMovers.so\" \"libfvMotionSolvers.so\"); "
        "motionSolver solidBody; cellZone rotating; solidBodyMotionFunction rotatingMotion; "
        f"origin ({origin[0]} {origin[1]} {origin[2]}); axis (0 0 -1); rpm {rpm:.12g};}}\n")
    files["0/U"]=_patches(files["0/U"],
        "nonCouple1 {type movingWallSlipVelocity; value uniform (0 0 0);} "
        "nonCouple2 {type movingWallSlipVelocity; value uniform (0 0 0);}")
    pressure="0/p_rgh" if "0/p_rgh" in files else "0/p"
    files[pressure]=_patches(files[pressure],
        "nonCouple1 {type fixedFluxPressure; value uniform 0;} "
        "nonCouple2 {type fixedFluxPressure; value uniform 0;}")
    for name in ("0/alpha.water","0/k","0/omega","0/nut"):
        if name in files:
            files[name]=_patches(files[name],
                "nonCouple1 {type zeroGradient;} nonCouple2 {type zeroGradient;}")
    files["system/fvSolution"]=files["system/fvSolution"].replace(
        "PIMPLE {","PIMPLE {correctMeshPhi false; ")
    return files,{"method":"OpenFOAM11-solidBody-NCC","cell_zone":"rotating",
        "face_zone":"rotating","baffle_patches":["nonCouple1","nonCouple2"],
        "ncc_patches":["nonConformalCyclic_on_nonCouple1","nonConformalCyclic_on_nonCouple2"],
        "center_foam_m":origin.tolist(),"axis_foam":[0,0,-1],"rpm":rpm,
        "cylinder_radius_m":radius,"cylinder_bottom_foam_z_m":bottom,
        "cylinder_top_foam_z_m":top,"segments":48}


def hull_point_indices(mesh: Path) -> np.ndarray:
    """Return hull patch point IDs from the final NCC mesh topology."""
    boundary=(mesh/"boundary").read_text()
    patch=re.search(r"\bhull\s*\{([^}]+)\}",boundary)
    if not patch: raise ValueError("NCC mesh has no hull patch")
    count_match=re.search(r"nFaces\s+(\d+)",patch.group(1))
    start_match=re.search(r"startFace\s+(\d+)",patch.group(1))
    if not count_match or not start_match: raise ValueError("hull patch face range is missing")
    count=int(count_match.group(1));start=int(start_match.group(1))
    faces_text=(mesh/"faces").read_text()
    list_match=re.search(r"\n(\d+)\s*\(\s*",faces_text)
    if not list_match: raise ValueError("mesh face list is missing")
    faces=re.findall(r"\d+\((\d+(?:\s+\d+)*)\)",faces_text[list_match.end():])[:int(list_match.group(1))]
    if len(faces)<start+count: raise ValueError("mesh face list is incomplete")
    return np.asarray(sorted({int(value) for row in faces[start:start+count]
                              for value in row.split()}),int)


def yaw_hull_point_error(initial: np.ndarray, moved: np.ndarray, ids: np.ndarray,
                         *, center: tuple[float,float,float], angle_rad: float) -> float:
    """Compare written hull points to positive-FRD yaw (negative foam Z)."""
    if initial.shape!=moved.shape or initial.ndim!=2 or initial.shape[1]!=3 or not len(ids):
        raise ValueError("incompatible NCC mesh points")
    origin=np.asarray(center,float);angle=-angle_rad
    c=math.cos(angle);s=math.sin(angle)
    rotation=np.asarray(((c,-s,0),(s,c,0),(0,0,1)))
    expected=origin+(initial[ids]-origin)@rotation.T
    return float(np.max(np.linalg.norm(expected-moved[ids],axis=1)))


def validate_rotation_mesh_quality(initial_log: str, current_log: str) -> dict:
    """Reject volume inversion or growth of the initial mesh's quality faults."""
    def scalar(source: str,pattern: str) -> float:
        matches=re.findall(pattern,source)
        if not matches: raise ValueError("rotating mesh quality metric is missing")
        return float(matches[-1].rstrip("."))
    def count(source: str,pattern: str) -> int:
        matches=re.findall(pattern,source)
        return int(matches[-1]) if matches else 0
    if ("Cell volumes OK" not in current_log or "Non-orthogonality check OK" not in current_log or
        "negative cell volume" in current_log):
        raise ValueError("rotating mesh contains invalid cell volumes or nonorthogonality")
    baseline={"min_volume":scalar(initial_log,r"Min volume = ([-+0-9.eE]+)"),
        "max_nonorthogonality":scalar(initial_log,r"Mesh non-orthogonality Max: ([-+0-9.eE]+)"),
        "max_skewness":scalar(initial_log,r"Max skewness = ([-+0-9.eE]+)"),
        "low_quality_tets":count(initial_log,r"Error in face tets: (\d+)"),
        "small_determinant_cells":count(initial_log,r"Cells with small determinant .*number of cells: (\d+)"),
        "concave_cells":count(initial_log,r"Concave cells .*number of cells: (\d+)")}
    current={"min_volume":scalar(current_log,r"Min volume = ([-+0-9.eE]+)"),
        "max_nonorthogonality":scalar(current_log,r"Mesh non-orthogonality Max: ([-+0-9.eE]+)"),
        "max_skewness":scalar(current_log,r"Max skewness = ([-+0-9.eE]+)"),
        "low_quality_tets":count(current_log,r"Error in face tets: (\d+)"),
        "small_determinant_cells":count(current_log,r"Cells with small determinant .*number of cells: (\d+)"),
        "concave_cells":count(current_log,r"Concave cells .*number of cells: (\d+)")}
    if (current["min_volume"]<.95*baseline["min_volume"] or
        current["max_nonorthogonality"]>baseline["max_nonorthogonality"]+5 or
        current["max_skewness"]>1.25*baseline["max_skewness"] or
        any(current[key]>baseline[key] for key in
            ("low_quality_tets","small_determinant_cells","concave_cells"))):
        raise ValueError("steady-yaw rotating mesh degraded")
    return {"baseline":baseline,"current":current}
