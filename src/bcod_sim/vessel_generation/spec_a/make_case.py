"""OpenFOAM 11 double-body captive case generation from one family template.

Cases are written but never launched implicitly. The turn uses the installed
MRFProperties/MRFFreestreamVelocity convention rather than an unavailable
SRFFreestreamVelocity boundary condition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import trimesh

from bcod_sim.vessel_generation.frame_contract import point_to_foam, body_velocity_to_fixed_hull_inlet
from .matrix import CaseState, froude_gate


def _head(cls: str, obj: str, location: str) -> str:
    return (f'FoamFile {{ version 2.0; format ascii; class {cls}; '
            f'location "{location}"; object {obj}; }}\n')


def _vector(x) -> str:
    return "("+" ".join(f"{float(v):.12g}" for v in x)+")"


@dataclass(frozen=True)
class VesselCaseSpec:
    hull_path: Path
    length_m: float
    beam_m: float
    draft_m: float
    block_coefficient: float
    density_kg_m3: float
    kinematic_viscosity_m2_s: float
    speed_mps: float
    cg_frd_m: tuple[float, float, float]
    waterline_frd_z_m: float = 0.
    scale_to_m: float = 1.

    def validate(self):
        dimensions = (self.length_m, self.beam_m, self.draft_m, self.density_kg_m3,
                      self.kinematic_viscosity_m2_s, self.speed_mps, self.scale_to_m)
        if not all(math.isfinite(z) and z > 0 for z in dimensions):
            raise ValueError("Positive finite vessel dimensions/fluid properties required")
        if not 0 < self.block_coefficient <= 1:
            raise ValueError("Invalid block coefficient")
        if not self.hull_path.is_file():
            raise FileNotFoundError(self.hull_path)
        if len(self.cg_frd_m) != 3 or not all(math.isfinite(z) for z in
                                            (*self.cg_frd_m, self.waterline_frd_z_m)):
            raise ValueError("Finite CG and waterline required")


def first_cell_height(spec: VesselCaseSpec, *, target_y_plus: float = 50.) -> float:
    re = spec.speed_mps*spec.length_m/spec.kinematic_viscosity_m2_s
    cf = .026/re**(1/7)
    return target_y_plus*spec.kinematic_viscosity_m2_s/(spec.speed_mps*math.sqrt(cf/2))


def write_case(root: Path, spec: VesselCaseSpec, state: CaseState, *,
               mesh_family: str = "displacement_monohull", target_y_plus: float = 50.,
               mesh_level: str | None = None, mesh_profile: str = "fast") -> dict:
    spec.validate()
    from .mesh_profiles import PROFILES, background_grid
    if mesh_level is not None:
        mesh_profile = {"coarse": "fast", "medium": "standard", "fine": "reference"}[mesh_level]
    if mesh_profile not in PROFILES or not 30 <= target_y_plus <= 100:
        raise ValueError("Invalid family mesh level or wall-function y+ target")
    gate = froude_gate(spec.speed_mps, spec.length_m)
    if gate["status"] == "out_of_envelope":
        raise ValueError("Double-body CFD prohibited above Fr_L=0.45")
    if gate["status"] == "extrapolated_fr" and gate["fr_l"] > .30+1e-12:
        raise ValueError("Requested Fr exceeds 0.30; explicitly generate the CFD matrix at Fr=0.30")
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Refusing to overwrite a generated case")
    for sub in ("0", "constant/triSurface", "system"):
        (root/sub).mkdir(parents=True, exist_ok=True)
    raw = trimesh.load(spec.hull_path, force="mesh")
    if not isinstance(raw, trimesh.Trimesh) or not raw.is_watertight:
        raise ValueError("Input hull must be watertight")
    mesh = raw.copy()
    mesh.vertices *= spec.scale_to_m
    mesh.vertices[:, 1] *= -1
    mesh.vertices[:, 2] = -(mesh.vertices[:, 2]-spec.waterline_frd_z_m)
    if mesh.volume < 0:
        mesh.invert()
    if mesh.vertices[:, 2].max() > 1e-7:
        raise ValueError("Input contains hull geometry above the frozen waterline")
    if mesh.vertices[:, 2].min() < -spec.draft_m*1.1:
        raise ValueError("Input draft disagrees with geometry")
    (root/"constant/triSurface/hull.stl").write_bytes(mesh.export(file_type="stl"))
    l = spec.length_m
    xmin, xmax, ymax, zmin = -3*l, 3*l, 3*l, -1.5*l
    profile = PROFILES[mesh_profile]
    (nx, ny, nz), cell = background_grid((xmax-xmin, 2*ymax, -zmin), profile["background_cells"])
    vertices = [(xmin,-ymax,zmin),(xmax,-ymax,zmin),(xmax,ymax,zmin),(xmin,ymax,zmin),
                (xmin,-ymax,0),(xmax,-ymax,0),(xmax,ymax,0),(xmin,ymax,0)]
    block = _head("dictionary", "blockMeshDict", "system")
    block += "convertToMeters 1;\nvertices ("+" ".join(_vector(v) for v in vertices)+");\n"
    block += f"blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));\n"
    block += "edges (); boundary (farField {type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1));} waterline {type symmetryPlane; faces ((4 5 6 7));});\n"
    (root/"system/blockMeshDict").write_text(block)
    # Exact same dictionaries for all cases in a family. A separate family
    # template may later be qualified with the documented three-mesh gate.
    # y+ is evaluated at the cell center; the full first layer is twice y.
    center_height = first_cell_height(spec, target_y_plus=target_y_plus)
    layer = 2*center_height
    snappy = _head("dictionary", "snappyHexMeshDict", "system")
    snappy += "castellatedMesh true; snap true; addLayers true;\n"
    snappy += ("geometry {hull.stl {type triSurfaceMesh; name hull;} "
               f"wake {{type searchableBox; min ({-2*l:.12g} {-spec.beam_m:.12g} {-2*spec.draft_m:.12g}); "
               f"max ({-.3*l:.12g} {spec.beam_m:.12g} 0);}}}}\n")
    snappy += ("castellatedMeshControls {maxLocalCells 5000000; maxGlobalCells 6000000; "
               "minRefinementCells 0; nCellsBetweenLevels 2; features (); "
               "refinementSurfaces {hull {level (4 5); patchInfo {type wall;}}} "
               f"refinementRegions {{wake {{mode inside; levels ((1e15 2));}}}} locationInMesh ({-2.5*l:.12g} 0 {-l:.12g}); "
               "resolveFeatureAngle 30; allowFreeStandingZoneFaces false;}\n")
    snappy += "snapControls {nSmoothPatch 3; tolerance 2; nSolveIter 30; nRelaxIter 5;}\n"
    snappy += ("addLayersControls {relativeSizes false; layers {hull {nSurfaceLayers 5;}} "
               f"expansionRatio 1.2; firstLayerThickness {layer:.12g}; "
               f"minThickness {layer*.1:.12g}; nGrow 0; featureAngle 60; nRelaxIter 5; "
               "nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; "
               "maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; "
               "minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;}\n")
    snappy += 'meshQualityControls {#includeEtc "caseDicts/mesh/generation/meshQualityDict"}\nmergeTolerance 1e-6;\n'
    (root/"system/snappyHexMeshDict").write_text(snappy)
    u, v, r = state.body_velocity(spec.speed_mps, l)
    inlet = body_velocity_to_fixed_hull_inlet((u, v, 0.))
    velocity = _vector(inlet)
    # point_to_foam expects the source Z-up waterline ordinate; the input here
    # is FRD Z-down. Project the force origin onto the waterline as specified.
    projected_cg = (*spec.cg_frd_m[:2], spec.waterline_frd_z_m)
    cg_foam = point_to_foam(projected_cg, -spec.waterline_frd_z_m)
    if abs(r) > 0:
        # Positive FRD yaw = negative foam-Z rotation. Turning-center offset
        # is body +Y at R=U/r, hence foam -Y.
        center = (cg_foam[0]-v/r, cg_foam[1]-u/r, cg_foam[2])
        rpm = -r*60/(2*math.pi)
        (root/"constant/MRFProperties").write_text(
            _head("dictionary", "MRFProperties", "constant")+
            f"SRF {{select all; origin {_vector(center)}; axis (0 0 1); rpm {rpm:.12g};}}\n")
        # U is ABSOLUTE velocity in Foundation 11's MRF formulation. The
        # rotating body moves through still water; MRF supplies the relative
        # -Omega x (x-center) flow. Adding -body velocity here doubles surge.
        velocity = "(0 0 0)"
        far_u = ("type MRFFreestreamVelocity; freestreamValue0 (0 0 0); "
                 "freestreamValue uniform (0 0 0); value uniform (0 0 0);")
    else:
        far_u = "type freestreamVelocity; freestreamValue uniform " + velocity + "; value uniform " + velocity + ";"
    p_bc = "type freestreamPressure; freestreamValue uniform 0; value uniform 0;"
    def field(name, cls, dims, internal, hull, far):
        return (_head(cls,name,"0")+f"dimensions {dims}; internalField uniform {internal};\n"
                f"boundaryField {{#includeEtc \"caseDicts/setConstraintTypes\" hull {{{hull}}} farField {{{far}}}}}\n")
    k = 1.5*(.05*spec.speed_mps)**2
    omega = math.sqrt(k)/(.09**.25*.1*l)
    fields = {
        "U": field("U","volVectorField","[0 1 -1 0 0 0 0]",velocity,
                   "type MRFnoSlip;" if r else "type noSlip;",far_u),
        "p": field("p","volScalarField","[0 2 -2 0 0 0 0]","0","type zeroGradient;",p_bc),
        "k": field("k","volScalarField","[0 2 -2 0 0 0 0]",f"{k:.12g}",
                   f"type kqRWallFunction; value uniform {k:.12g};",
                   f"type freestream; freestreamValue uniform {k:.12g}; value uniform {k:.12g};"),
        "omega": field("omega","volScalarField","[0 0 -1 0 0 0 0]",f"{omega:.12g}",
                       f"type omegaWallFunction; value uniform {omega:.12g};",
                       f"type freestream; freestreamValue uniform {omega:.12g}; value uniform {omega:.12g};"),
        "nut": field("nut","volScalarField","[0 2 -1 0 0 0 0]","0",
                     "type nutkWallFunction; value uniform 0;","type calculated; value uniform 0;")}
    for name, content in fields.items():
        (root/"0"/name).write_text(content)
    (root/"constant/physicalProperties").write_text(_head("dictionary","physicalProperties","constant")+
        f"viscosityModel constant; nu [0 2 -1 0 0 0 0] {spec.kinematic_viscosity_m2_s:.12g};\n")
    (root/"constant/momentumTransport").write_text(_head("dictionary","momentumTransport","constant")+
        "simulationType RAS; RAS {model kOmegaSST; turbulence on; printCoeffs on;}\n")
    control = _head("dictionary","controlDict","system")
    control += ("application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; "
                "endTime 3000; deltaT 1; writeControl timeStep; writeInterval 500; "
                "writeFormat ascii; runTimeModifiable false;\n")
    control += (f'functions {{forces {{type forces; libs ("libforces.so"); patches (hull); '
                f'rho rhoInf; rhoInf {spec.density_kg_m3:.12g}; CofR {_vector(cg_foam)}; '
                'writeControl timeStep; writeInterval 1;} '
                'yPlus {type yPlus; libs ("libfieldFunctionObjects.so"); '
                'executeControl timeStep; executeInterval 500; writeControl timeStep; writeInterval 500;}}\n')
    (root/"system/controlDict").write_text(control)
    (root/"system/fvSchemes").write_text(_head("dictionary","fvSchemes","system")+
        "ddtSchemes {default steadyState;} gradSchemes {default Gauss linear; grad(U) cellLimited Gauss linear 1;} "
        "divSchemes {default none; div(phi,U) bounded Gauss linearUpwind grad(U); "
        "div(phi,k) bounded Gauss upwind; div(phi,omega) bounded Gauss upwind; "
        "div((nuEff*dev2(T(grad(U))))) Gauss linear;} "
        "laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} "
        "snGradSchemes {default corrected;} wallDist {method meshWave;}\n")
    (root/"system/fvSolution").write_text(_head("dictionary","fvSolution","system")+
        "solvers {p {solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01;} "
        "U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1;} "
        "k {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1;} "
        "omega {$k;}} SIMPLE {nNonOrthogonalCorrectors 1;} "
        "relaxationFactors {fields {p 0.3;} equations {U 0.7; k 0.7; omega 0.7;}}\n")
    metadata = {"state": asdict(state), "body_velocity_frd": [u, v, r],
                "relative_flow_at_cg_foam": inlet,
                "inlet_foam": (0., 0., 0.) if r else inlet,
                "froude_gate": gate, "mesh_family": mesh_family,
                "target_y_plus": target_y_plus, "first_cell_height_m": center_height,
                "first_layer_thickness_m": layer, "mesh_level": mesh_level, "mesh_profile": mesh_profile,
                "mesh_budget": profile, "background_spacing_m": cell,
                "cell_count_background": nx*ny*nz, "moment_reference_foam": cg_foam,
                "moment_reference_body": projected_cg,
                "solver_moment_reference_body": projected_cg,
                "runtime_moment_reference_body": spec.cg_frd_m,
                "cell_budget_warning": "Background mesh exceeds 3M cells" if nx*ny*nz > 3000000 else None,
                "solver": "simpleFoam", "rotation_method": "OpenFOAM11-MRF-all" if r else None,
                "force_output": "physical_fluid_on_hull_foam", "case_spec": {
                    **asdict(spec), "hull_path": str(spec.hull_path)}}
    (root/"case_config.json").write_text(json.dumps(metadata, indent=2, default=float)+"\n")
    return metadata
