"""Deterministic experimental OpenFOAM 11 mesh/case for 2D section startup.

The O-grid handles star-shaped closed contours. It is intentionally isolated
from the frozen 3D vessel pipeline. No benchmark coefficients enter a case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def radial_contour(shape: str, angles: np.ndarray, *, ratio: float = 1.) -> np.ndarray:
    ca, sa = np.cos(angles), np.sin(angles)
    if shape == "circle":
        radius = np.ones_like(ca)
    elif shape == "ellipse":
        radius = 1 / np.sqrt((ca/ratio)**2 + sa**2)
    elif shape == "rectangle":
        radius = np.minimum(ratio/np.maximum(np.abs(ca), 1e-12),
                            1/np.maximum(np.abs(sa), 1e-12))
    else:
        raise ValueError("Unsupported canonical shape")
    return np.column_stack((radius*ca, radius*sa))


def radial_mesh_contour(contour: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Intersect rays with a closed star-shaped contour; reject ambiguous rays."""
    points=np.asarray(contour,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<16 or not np.isfinite(points).all():
        raise ValueError('Invalid section contour')
    # Scale transverse draft to diameter 2; the CFD Reynolds number is then
    # defined using physical draft, independent of source vessel scale.
    center=np.mean(points,axis=0)
    draft=float(np.ptp(points[:,1]))
    if draft<=0:
        raise ValueError('Degenerate section draft')
    points=(points-center)/(draft/2)
    edges=np.roll(points,-1,axis=0)-points
    result=[]
    for angle in angles:
        direction=np.array([math.cos(angle),math.sin(angle)])
        cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
        hits=[]
        for start,edge in zip(points,edges):
            denominator=cross(direction,edge)
            if abs(denominator)<1e-12:continue
            distance=cross(start,edge)/denominator
            fraction=cross(start,direction)/denominator
            if distance>1e-9 and -1e-9<=fraction<=1+1e-9:hits.append(distance)
        unique=np.unique(np.round(hits,9))
        if len(unique)!=1:
            raise ValueError('Section is not star-shaped around its centroid')
        result.append(unique[0]*direction)
    return np.asarray(result)


def write_case(path: Path, *, shape: str = "circle", aspect_ratio: float = 1.,
               reynolds: float = 100., sectors: int = 32, angular_cells: int = 4,
               radial_cells: int = 20, outer_radii: float = 20., end_time: float = 20.,
               dt: float = .01, contour: np.ndarray | None = None):
    if sectors % 8 or sectors < 16 or min(angular_cells, radial_cells) < 2:
        raise ValueError("Invalid O-grid resolution")
    if min(reynolds, outer_radii, end_time, dt, aspect_ratio) <= 0:
        raise ValueError("Positive case settings required")
    # Vertical extrusion depth=1 m; returned force is per unit section length.
    theta = 2*math.pi*np.arange(sectors)/sectors
    inner = (radial_mesh_contour(contour,theta) if contour is not None else
             radial_contour(shape, theta, ratio=aspect_ratio))
    outer = outer_radii*np.column_stack((np.cos(theta), np.sin(theta)))
    if np.max(np.linalg.norm(inner, axis=1)) >= outer_radii/2:
        raise ValueError("Section too large for farfield")
    points = []
    for z in (0., 1.):
        for ring in (inner, outer):
            points.extend((float(x), float(y), z) for x, y in ring)
    def vertex(z, ring, j):
        return z*2*sectors + ring*sectors + j % sectors
    vertices = "\n".join(f"    ({x:.12g} {y:.12g} {z:.12g})" for x,y,z in points)
    blocks=[]; walls=[]; inlets=[]; outlets=[]; sides=[]; front=[]; back=[]
    for j in range(sectors):
        a,b,c,d = (vertex(0,0,j),vertex(0,1,j),vertex(0,1,j+1),vertex(0,0,j+1))
        A,B,C,D = (vertex(1,0,j),vertex(1,1,j),vertex(1,1,j+1),vertex(1,0,j+1))
        blocks.append(f"    hex ({a} {b} {c} {d} {A} {B} {C} {D}) ({radial_cells} {angular_cells} 1) simpleGrading (30 1 1)")
        walls.append(f"({d} {a} {A} {D})")
        mid=theta[j]+math.pi/sectors
        face=f"({b} {c} {C} {B})"
        if math.cos(mid) < -math.sqrt(.5): inlets.append(face)
        elif math.cos(mid) > math.sqrt(.5): outlets.append(face)
        else: sides.append(face)
        front.append(f"({a} {d} {c} {b})")
        back.append(f"({A} {B} {C} {D})")
    def patch(name,kind,faces):
        return f"{name} {{ type {kind}; faces ( {' '.join(faces)} ); }}"
    mesh = "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
    mesh += "convertToMeters 1;\nvertices (\n"+vertices+"\n);\nblocks (\n"+"\n".join(blocks)+"\n);\n"
    mesh += "edges ();\nboundary (\n"+"\n".join((patch("section","wall",walls),patch("inlet","patch",inlets),
                                                 patch("outlet","patch",outlets),patch("side","patch",sides),
                                                 patch("frontAndBack","empty",front+back)))+"\n);\n"
    path.mkdir(parents=True, exist_ok=True)
    for folder in ("system", "constant", "0"):
        (path/folder).mkdir(exist_ok=True)
    (path/"system/blockMeshDict").write_text(mesh)
    (path/"0/U").write_text("""FoamFile { version 2.0; format ascii; class volVectorField; location \"0\"; object U; }
dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);
boundaryField {
 section { type noSlip; }
 inlet { type fixedValue; value uniform (1 0 0); }
 outlet { type inletOutlet; inletValue uniform (1 0 0); value uniform (1 0 0); }
 side { type slip; }
 frontAndBack { type empty; }
}
""")
    (path/"0/p").write_text("""FoamFile { version 2.0; format ascii; class volScalarField; location \"0\"; object p; }
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField {
 section { type zeroGradient; }
 inlet { type zeroGradient; }
 outlet { type fixedValue; value uniform 0; }
 side { type zeroGradient; }
 frontAndBack { type empty; }
}
""")
    # Radius=1 means diameter=2 and Re=2/nu at unit inflow.
    (path/"constant/physicalProperties").write_text(
        f"FoamFile {{ version 2.0; format ascii; class dictionary; object physicalProperties; }}\nviscosityModel constant; nu {2/reynolds:.12g};\n")
    (path/"constant/momentumTransport").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object momentumTransport; }\nsimulationType laminar;\n")
    (path/"system/controlDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application foamRun; solver incompressibleFluid;
startFrom startTime; startTime 0; stopAt endTime; endTime {end_time:.12g};
deltaT {dt:.12g}; adjustTimeStep yes; maxCo 0.5; maxDeltaT {dt:.12g};
writeControl runTime; writeInterval 2; purgeWrite 2; writeFormat ascii;
runTimeModifiable no;
functions {{
 sectionForces {{
  type forces; libs (\"libforces.so\"); patches (section);
  p p; U U; rho rhoInf; rhoInf 1; CofR (0 0 0);
  writeControl timeStep; writeInterval 1;
 }}
}}
""")
    (path/"system/fvSchemes").write_text("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U); }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
    (path/"system/fvSolution").write_text("""FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {
 p { solver GAMG; tolerance 1e-8; relTol 0.01; smoother GaussSeidel; }
 pFinal { $p; relTol 0; }
 U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.01; }
 UFinal { $U; relTol 0; }
}
PIMPLE { momentumPredictor yes; nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; }
relaxationFactors { equations { U 1; } }
""")
    config = dict(shape='mesh_contour' if contour is not None else shape,
                  contour_sha256=hashlib.sha256(np.asarray(contour,dtype='<f8').tobytes()).hexdigest() if contour is not None else None,
                  aspect_ratio=aspect_ratio, reynolds=reynolds, sectors=sectors,
                  angular_cells=angular_cells, radial_cells=radial_cells,
                  outer_radii=outer_radii, end_time=end_time, dt=dt)
    content = json.dumps(config, sort_keys=True).encode()+mesh.encode()
    config["case_sha256"] = hashlib.sha256(content).hexdigest()
    (path/"case_config.json").write_text(json.dumps(config, indent=2)+"\n")
    return config


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--shape", choices=("circle","ellipse","rectangle"), default="circle")
    parser.add_argument("--contour-json", type=Path)
    parser.add_argument("--aspect-ratio", type=float, default=1.)
    parser.add_argument("--reynolds", type=float, default=100.)
    parser.add_argument("--sectors", type=int, default=32)
    parser.add_argument("--angular-cells", type=int, default=4)
    parser.add_argument("--radial-cells", type=int, default=20)
    parser.add_argument("--outer-radii", type=float, default=20.)
    parser.add_argument("--end-time", type=float, default=20.)
    parser.add_argument("--dt", type=float, default=.01)
    args=parser.parse_args()
    contour=np.asarray(json.loads(args.contour_json.read_text()),dtype=float) if args.contour_json else None
    print(json.dumps(write_case(args.path, shape=args.shape, aspect_ratio=args.aspect_ratio,
                                reynolds=args.reynolds, sectors=args.sectors,
                                angular_cells=args.angular_cells, radial_cells=args.radial_cells,
                                outer_radii=args.outer_radii, end_time=args.end_time, dt=args.dt,
                                contour=contour), indent=2))
