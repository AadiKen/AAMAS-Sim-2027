"""Backend-neutral CFD adapter with an OpenFOAM baseline and synthetic oracle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import json
import math
import re
import struct
import subprocess
import time

import numpy as np

from .fitting import ForceMomentDataset, synthetic_dataset
from .identification import IdentificationCase, FluidModel, MotionType, TurbulenceModel
from .frame_contract import (CONVENTION, body_velocity_to_fixed_hull_inlet,
    point_to_foam, vector_to_foam, foam_wrench_to_body)
from .rotating_ncc import (configure_steady_yaw, hull_point_indices,
    yaw_hull_point_error, validate_rotation_mesh_quality)


class CFDExecutionError(RuntimeError): pass


@dataclass(frozen=True)
class ManeuverPoint:
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]


@dataclass(frozen=True)
class CFDCase:
    geometry_hash: str
    backend: str
    backend_version: str
    solver_settings: dict
    tests: tuple[ManeuverPoint, ...]
    units: tuple[str, str, str] = ("m/s,rad/s", "m/s2,rad/s2", "N,Nm")


@dataclass(frozen=True)
class CFDResult:
    case: CFDCase
    dataset: ForceMomentDataset
    converged: bool
    raw_provenance: dict

    def validated(self) -> "CFDResult":
        if not self.converged: raise CFDExecutionError("CFD solver did not converge")
        self.dataset.validate()
        if len(self.case.tests) != self.dataset.velocity.shape[0]: raise CFDExecutionError("incomplete CFD run")
        return self


@dataclass(frozen=True)
class OpenFOAMResult:
    case_id: str
    velocity_body_frd_mps: tuple[float,float,float]
    angular_rate_body_frd_radps: tuple[float,float,float]
    force_body_frd_n: tuple[float,float,float]
    moment_body_frd_nm: tuple[float,float,float]
    units: dict
    frame: str
    moment_reference_point_frd_m: tuple[float,float,float]
    source_case: str
    raw_output_sha256: str
    solver: str
    converged: bool
    wall_seconds: float
    fitting_window_s: tuple[float,float]|None = None
    convention: str = "legacy-unqualified"
    sample_mode: str = "unqualified"


class CFDAdapter(Protocol):
    def generate_case(self, geometry_hash: str, tests: tuple[ManeuverPoint,...]) -> CFDCase: ...
    def execute(self, case: CFDCase) -> CFDResult: ...


def deterministic_test_matrix(seed: int=0, samples: int=96) -> tuple[ManeuverPoint,...]:
    rng=np.random.default_rng(seed); v=rng.uniform(-2,2,(samples,6)); a=rng.uniform(-1,1,(samples,6))
    return tuple(ManeuverPoint(tuple(v[i]),tuple(a[i])) for i in range(samples))


class SyntheticCFDAdapter:
    def __init__(self, added_mass, linear, quadratic, *, seed=0, noise_std=0):
        self.coefficients=(added_mass,linear,quadratic); self.seed=seed; self.noise_std=noise_std

    def generate_case(self, geometry_hash, tests):
        return CFDCase(geometry_hash,"synthetic-cfd","1",{"model":"known diagonal 6-DOF"},tests)

    def execute(self, case):
        v=np.asarray([x.velocity for x in case.tests]); a=np.asarray([x.acceleration for x in case.tests])
        am,lin,quad=map(np.asarray,self.coefficients); wrench=a*am+v*lin+np.abs(v)*v*quad
        if self.noise_std: wrench += np.random.default_rng(self.seed).normal(0,self.noise_std,wrench.shape)
        return CFDResult(case,ForceMomentDataset(v,a,wrench,case.units,self.seed),True,
                         {"backend":"synthetic-cfd","seed":self.seed}).validated()


class OpenFOAMAdapter:
    """Owns genuine OpenFOAM case generation, meshing, solving, and parsing."""
    adapter_version = "4"

    @staticmethod
    def _mesh_points(path: Path) -> np.ndarray:
        content=path.read_text()
        match=re.search(r"\n(\d+)\s*\(\s*",content)
        if match is None: raise CFDExecutionError("mesh point list is missing")
        count=int(match.group(1))
        vectors=re.findall(r"\(([-+\d.eE\s]+)\)",content[match.end():])[:count]
        if len(vectors)!=count: raise CFDExecutionError("mesh point list is incomplete")
        points=np.asarray([[float(value) for value in row.split()] for row in vectors])
        if points.shape!=(count,3) or not np.isfinite(points).all():
            raise CFDExecutionError("mesh points are malformed")
        return points

    @staticmethod
    def _waterline_aligned_stl(data: bytes, waterline_z_m: float) -> bytes:
        """Translate source-up STL to a z=0 free surface without deforming it."""
        if not waterline_z_m: return data
        if len(data)>=84 and len(data)==84+50*struct.unpack_from("<I",data,80)[0]:
            shifted=bytearray(data)
            for offset in range(84,len(data),50):
                for vertex in range(3):
                    z_offset=offset+12+vertex*12+8
                    struct.pack_into("<f",shifted,z_offset,
                        struct.unpack_from("<f",data,z_offset)[0]-waterline_z_m)
            return bytes(shifted)
        source=data.decode("ascii")
        vertex=re.compile(r"(\bvertex\s+)([-+\d.eE]+)(\s+)([-+\d.eE]+)(\s+)([-+\d.eE]+)")
        shifted,count=vertex.subn(lambda match:(f"{match[1]}{match[2]}{match[3]}{match[4]}"
            f"{match[5]}{float(match[6])-waterline_z_m:.10g}"),source)
        if not count: raise CFDExecutionError("STL has no vertices to align with waterline")
        return shifted.encode("ascii")
    def __init__(self, work_root: Path, *, executable="foamRun", version="11",
                 docker_image="openfoam/openfoam11-paraview510:11", runtime="docker"):
        self.work_root=Path(work_root); self.executable=executable; self.version=version
        self.docker_image=docker_image; self.runtime=runtime

    def generate_case(self, geometry_hash, tests):
        if len(geometry_hash)!=64: raise CFDExecutionError("malformed geometry hash")
        case=CFDCase(geometry_hash,"OpenFOAM",self.version,{"solver":"incompressibleFluid","residual":1e-6},tests)
        root=self.work_root/geometry_hash; root.mkdir(parents=True,exist_ok=True)
        (root/"case.json").write_text(json.dumps({"case":case.geometry_hash,"tests":[x.__dict__ for x in tests]},sort_keys=True))
        return case

    def execute(self, case):
        proc=subprocess.run([self.executable],cwd=self.work_root/case.geometry_hash,capture_output=True,text=True)
        if proc.returncode: raise CFDExecutionError(f"OpenFOAM failed: {proc.stderr[-500:]}")
        result_file=self.work_root/case.geometry_hash/"forces.json"
        if not result_file.exists(): raise CFDExecutionError("OpenFOAM force/moment output is missing")
        raw=json.loads(result_file.read_text()); required={"velocity","acceleration","wrench","units","converged"}
        if set(raw)!=required: raise CFDExecutionError("malformed OpenFOAM result channels")
        return CFDResult(case,ForceMomentDataset(np.asarray(raw["velocity"],float),np.asarray(raw["acceleration"],float),
            np.asarray(raw["wrench"],float),tuple(raw["units"])),bool(raw["converged"]),
            {"backend":"OpenFOAM","version":self.version,"settings":case.solver_settings}).validated()

    def verify_runtime(self) -> dict:
        if self.runtime!="docker": raise CFDExecutionError("only the verified Docker runtime is configured")
        inspect=subprocess.run(["docker","image","inspect",self.docker_image,"--format",
            "{{index .RepoDigests 0}} {{.Id}}"],capture_output=True,text=True)
        if inspect.returncode: raise CFDExecutionError("required OpenFOAM Docker image is unavailable")
        version=self._container(["foamVersion"],capture=True)
        version_text=(version.stdout+version.stderr).strip()
        if version.returncode or "OpenFOAM" not in version_text:
            raise CFDExecutionError("OpenFOAM cannot be invoked in the configured container")
        return {"distribution":"OpenFOAM Foundation","version":version_text,"solver":self.executable,
                "runtime":"docker","image":self.docker_image,"image_identity":inspect.stdout.strip()}

    def _container(self, command: list[str], *, case_dir: Path|None=None, capture=False):
        args=self._container_args(command,case_dir=case_dir)
        return subprocess.run(args,capture_output=capture,text=True)

    def _container_args(self, command: list[str], *, case_dir: Path|None=None):
        script="source /opt/openfoam11/etc/bashrc && "+" ".join(command)
        args=["docker","run","--rm","--platform","linux/amd64","--entrypoint","/bin/bash"]
        if case_dir is not None:
            args += ["-v",f"{case_dir.resolve()}:/case","-w","/case"]
        args += [self.docker_image,"-lc",script]
        return args

    def generate_operating_case(self, geometry, *, case_id: str, surge_mps: float) -> Path:
        if not math.isfinite(surge_mps) or surge_mps<=0: raise CFDExecutionError("surge speed must be positive")
        root=self.work_root/case_id
        if root.exists(): raise CFDExecutionError(f"case already exists: {case_id}")
        for folder in (root/"0",root/"constant"/"triSurface",root/"system"): folder.mkdir(parents=True,exist_ok=True)
        (root/"constant"/"triSurface"/"hull.stl").write_bytes(Path(geometry.path).read_bytes())
        files=self._case_files(surge_mps)
        for name,text in files.items(): (root/name).write_text(text)
        metadata={"case_id":case_id,"solver":"foamRun -solver incompressibleFluid","flow_speed_mps":surge_mps,
            "water_density_kg_m3":1000.,"kinematic_viscosity_m2_s":1e-6,
            "domain_extent_frd_m":{"min":[-3,-2,-1],"max":[5,2,1]},"geometry_scale":1.,
            "geometry_orientation":"STL x/y/z = body FRD x/y/z","moment_reference_point_frd_m":[0,0,0],
            "boundary_conditions":{"inlet":"fixed velocity","outlet":"fixed kinematic pressure","sides":"slip","hull":"noSlip"},
            "turbulence_model":"laminar","free_surface_model":None,"steady_state":True,"maximum_iterations":300,
            "force_output":"postProcessing/forces/*/forces.dat","geometry_sha256":geometry.content_hash,
            "adapter_version":self.adapter_version}
        (root/"case_metadata.json").write_text(json.dumps(metadata,sort_keys=True,indent=2))
        return root

    def generate_identification_case(self, geometry, case: IdentificationCase) -> Path:
        """Materialize a generic case without hiding motion or measured channels.

        Steady translation retains the verified legacy solver setup. Rotations and
        forced motions add an explicit prescribed-motion dictionary. Free-surface
        cases select OpenFOAM's VOF module and carry phase/gravity initial fields.
        """
        if geometry.content_hash != case.geometry_hash: raise CFDExecutionError("case geometry hash mismatch")
        root=self.work_root/case.case_id
        if root.exists():
            metadata=root/"case_metadata.json"
            if metadata.exists() and json.loads(metadata.read_text()).get("case_hash")==case.case_id: return root
            raise CFDExecutionError(f"case already exists with incompatible metadata: {case.case_id}")
        for folder in (root/"0",root/"constant"/"triSurface",root/"system"): folder.mkdir(parents=True,exist_ok=True)
        original_geometry=Path(geometry.path).read_bytes()
        (root/"constant"/"triSurface"/"source_hull.stl").write_bytes(original_geometry)
        shifted_geometry=self._waterline_aligned_stl(original_geometry,case.waterline_z_m)
        (root/"constant"/"triSurface"/"hull.stl").write_bytes(shifted_geometry)
        velocity=case.velocity_vector()
        inlet=body_velocity_to_fixed_hull_inlet(velocity[:3]) if case.motion_type==MotionType.STEADY_VELOCITY else (0.,0.,0.)
        speed=inlet[0] if case.motion_type==MotionType.STEADY_VELOCITY else 1e-12
        files=self._case_files(speed)
        files=self._apply_case_settings(files,case)
        u=files["0/U"]
        translational=inlet
        vector=f"({translational[0]} {translational[1]} {translational[2]})"
        u=re.sub(r"uniform \([-+0-9.eE]+ 0 0\)",f"uniform {vector}",u)
        files["0/U"]=u
        solver="incompressibleFluid"
        if case.fluid_model==FluidModel.FREE_SURFACE:
            solver="incompressibleVoF"
            files.pop("constant/physicalProperties",None)
            files.pop("0/p",None)
            files.update(self._free_surface_files(case,translational))
            high=case.domain_settings.maximum_frd_m
            cell=case.mesh_settings.base_cell_size_m
            location=f"({high[0]-cell} 0 {high[2]-cell})"
            files["system/snappyHexMeshDict"]=files["system/snappyHexMeshDict"].replace("locationInMesh (4 0 0)",f"locationInMesh {location}")
            if case.turbulence_settings.model==TurbulenceModel.K_OMEGA_SST:
                snappy=files["system/snappyHexMeshDict"]
                snappy=snappy.replace("addLayers false","addLayers true")
                layers=("addLayersControls {relativeSizes false; layers {hull {nSurfaceLayers "
                    f"{case.mesh_settings.boundary_layers};}}}} expansionRatio 1.25; "
                    f"finalLayerThickness {case.mesh_settings.boundary_layer_outer_thickness_m}; "
                    "minThickness 0.002; nGrow 0; "
                    "featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; "
                    "nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; "
                    "maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; "
                    "nBufferCellsNoExtrude 0; nLayerIter 50; nRelaxedIter 20;}")
                snappy=snappy.replace("addLayersControls {relativeSizes true; layers {};}",layers)
                snappy=snappy.replace('meshQualityControls {#includeEtc "caseDicts/mesh/generation/meshQualityDict"}',
                    'meshQualityControls {#includeEtc "caseDicts/mesh/generation/meshQualityDict" relaxed {maxNonOrtho 65;}}')
                files["system/snappyHexMeshDict"]=snappy
            files["system/controlDict"]=files["system/controlDict"].replace("solver incompressibleFluid",f"solver {solver}")
        rotation_topology=None
        if case.motion_type==MotionType.STEADY_ROTATION:
            files["0/U"]=files["0/U"].replace("hull {type noSlip;}",
                                                "hull {type movingWallVelocity; value uniform (0 0 0);}")
            files,rotation_topology=configure_steady_yaw(files,case,shifted_geometry)
            files["system/controlDict"]=files["system/controlDict"].replace("solver incompressibleFluid",f"solver {solver}")
        elif case.motion_type in {MotionType.FORCED_TRANSLATION,MotionType.FORCED_ROTATION}:
            files["constant/dynamicMeshDict"]=self._motion_dictionary(case)
            files["0/pointDisplacement"]=self._point_displacement(case,files["0/U"])
            files["0/U"]=files["0/U"].replace("hull {type noSlip;}",
                                                "hull {type movingWallVelocity; value uniform (0 0 0);}")
            files["system/fvSolution"]=files["system/fvSolution"].replace(
                "solvers {", "solvers {cellDisplacement {solver GAMG; smoother GaussSeidel; "
                "tolerance 1e-7; relTol 0.01;} cellDisplacementFinal {$cellDisplacement; relTol 0;} ")
            files["system/controlDict"]=files["system/controlDict"].replace("solver incompressibleFluid",f"solver {solver}")
        for name,text in files.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        metadata={"case_id":case.case_id,"case_hash":case.case_id,"geometry_sha256":case.geometry_hash,
            "motion_type":case.motion_type.value,"dof":case.dof.value,"magnitude":case.magnitude,
            "frequency_rad_s":case.frequency_rad_s,"amplitude":case.amplitude,"velocity_body_frd":velocity,
            "fluid_model":case.fluid_model.value,"solver":f"foamRun -solver {solver}",
            "turbulence_model":(case.turbulence_settings.model.value if case.turbulence_settings else "laminar"),
            "turbulence_settings":({**case.turbulence_settings.__dict__,
                "model":case.turbulence_settings.model.value} if case.turbulence_settings else None),
            "moment_reference_point_frd_m":case.reference_point_frd_m,
            "solver_moment_reference_point_m":point_to_foam(case.reference_point_frd_m,case.waterline_z_m),
            "source_waterline_z_m":case.waterline_z_m,"geometry_translation_z_m":-case.waterline_z_m,
            "frame_contract":CONVENTION,"geometry_input_frame":"FPU_m",
            "canonical_body_frame":"FRD_m","openfoam_frame":"FPU_m",
            "geometry_transform":{"rotation_source_to_foam":[[1,0,0],[0,1,0],[0,0,1]],
                "translation_m":[0,0,-case.waterline_z_m]},
            "cg_body_frd_m":case.cg_frd_m,
            "inlet_velocity_foam_mps":inlet,
            "body_angular_rate_to_foam":"diag(1,-1,-1)",
            "force_to_resisting_body":"-diag(1,-1,-1) * fluid_on_body",
            "moment_to_resisting_body":"-R * (M_A + (A-B) cross F)",
            "units":{"length":"m","velocity":"m/s","angular_rate":"rad/s",
                "force":"N","moment":"N*m"},
            "measured_channels":["Fx","Fy","Fz","Mx","My","Mz"],
            "water_properties":case.water_properties.__dict__,"mesh_settings":case.mesh_settings.__dict__,
            "solver_settings":case.solver_settings.__dict__,"domain_settings":case.domain_settings.__dict__,
            "openfoam_version":case.openfoam_version,"adapter_version":self.adapter_version}
        if rotation_topology is not None: metadata["rotation_topology"]=rotation_topology
        (root/"case_metadata.json").write_text(json.dumps(metadata,sort_keys=True,indent=2))
        return root

    @staticmethod
    def _motion_dictionary(case: IdentificationCase) -> str:
        return ("FoamFile {version 2.0; format ascii; class dictionary; location \"constant\"; "
                "object dynamicMeshDict;}\n"
                "mover {type motionSolver; libs (\"libfvMeshMovers.so\" "
                "\"libfvMotionSolvers.so\"); motionSolver displacementLaplacian; "
                "diffusivity quadratic inverseDistance (hull);}\n")

    @staticmethod
    def _point_displacement(case: IdentificationCase, velocity_file: str) -> str:
        axis=[0.,0.,0.];axis[case.dof.index%3]=1.
        direction=vector_to_foam(axis)
        vector=lambda values:"("+" ".join(str(value) for value in values)+")"
        origin=vector(point_to_foam(case.reference_point_frd_m,case.waterline_z_m))
        if case.motion_type==MotionType.FORCED_TRANSLATION:
            amplitude=vector(tuple(case.amplitude*x for x in direction))
            hull=(f"type oscillatingDisplacement; amplitude {amplitude}; "
                  f"omega {case.frequency_rad_s}; value uniform (0 0 0);")
        elif case.motion_type==MotionType.FORCED_ROTATION:
            hull=(f"type angularOscillatingDisplacement; axis {vector(direction)}; "
                  f"origin {origin}; angle0 0; amplitude {case.amplitude}; "
                  f"omega {case.frequency_rad_s}; value uniform (0 0 0);")
        else:
            hull=(f"type solidBodyMotionDisplacement; solidBodyMotionFunction rotatingMotion; "
                  f"rotatingMotionCoeffs {{origin {origin}; axis {vector(direction)}; "
                  f"omega {case.magnitude};}} value uniform (0 0 0);")
        patches=("inletWater","inletAir","outletWater","outletAir","bottom",
                 "atmosphere","sides") if case.fluid_model==FluidModel.FREE_SURFACE else ("inlet","outlet","sides")
        boundary=" ".join(f"{patch} {{type fixedValue; value uniform (0 0 0);}}" for patch in patches)
        return ("FoamFile {version 2.0; format ascii; class pointVectorField; location \"0\"; "
                "object pointDisplacement;}\n"
                "dimensions [0 1 0 0 0 0 0]; internalField uniform (0 0 0); "
                f"boundaryField {{{boundary} hull {{{hull}}}}}\n")

    @staticmethod
    def _apply_case_settings(files: dict[str,str], case: IdentificationCase) -> dict[str,str]:
        files=dict(files);low=case.domain_settings.minimum_frd_m;high=case.domain_settings.maximum_frd_m
        vertices=(f"({low[0]} {low[1]} {low[2]})({high[0]} {low[1]} {low[2]})({high[0]} {high[1]} {low[2]})"
            f"({low[0]} {high[1]} {low[2]})({low[0]} {low[1]} {high[2]})({high[0]} {low[1]} {high[2]})"
            f"({high[0]} {high[1]} {high[2]})({low[0]} {high[1]} {high[2]})")
        cells=tuple(max(2,int(round((high[i]-low[i])/case.mesh_settings.base_cell_size_m))) for i in range(3))
        block=files["system/blockMeshDict"]
        block=re.sub(r"vertices \([^;]+;",f"vertices ({vertices});",block)
        block=re.sub(r"\(32 20 16\)",f"({cells[0]} {cells[1]} {cells[2]})",block);files["system/blockMeshDict"]=block
        levels=case.mesh_settings.hull_refinement_levels
        files["system/snappyHexMeshDict"]=files["system/snappyHexMeshDict"].replace("level (2 2)",f"level ({levels[0]} {levels[1]})")
        control=files["system/controlDict"].replace("endTime 300",f"endTime {case.solver_settings.end_time_s}")
        control=control.replace("deltaT 1",f"deltaT {case.solver_settings.initial_timestep_s}")
        origin=" ".join(map(str,point_to_foam(case.reference_point_frd_m,case.waterline_z_m)))
        control=control.replace("CofR (0 0 0)",f"CofR ({origin})")
        control=control.replace("rhoInf 1000",f"rhoInf {case.water_properties.density_kg_m3}");files["system/controlDict"]=control
        files["constant/physicalProperties"]=files["constant/physicalProperties"].replace("1e-6",str(case.water_properties.kinematic_viscosity_m2_s))
        return files

    @staticmethod
    def _free_surface_files(case: IdentificationCase,velocity=(0.,0.,0.),*,include_hull=True) -> dict[str,str]:
        rho=case.water_properties.density_kg_m3;nu=case.water_properties.kinematic_viscosity_m2_s;g=case.water_properties.gravity_mps2
        header=lambda cls,obj,loc:f'FoamFile\n{{ version 2.0; format ascii; class {cls}; location "{loc}"; object {obj}; }}\n'
        low,high=case.domain_settings.minimum_frd_m,case.domain_settings.maximum_frd_m
        nx=max(2,int(round((high[0]-low[0])/case.mesh_settings.base_cell_size_m)));ny=max(2,int(round((high[1]-low[1])/case.mesh_settings.base_cell_size_m)))
        nz_water=max(1,int(round((0-low[2])/case.mesh_settings.base_cell_size_m)));nz_air=max(1,int(round((high[2]-0)/case.mesh_settings.base_cell_size_m)))
        vertices=f"({low[0]} {low[1]} {low[2]})({high[0]} {low[1]} {low[2]})({high[0]} {high[1]} {low[2]})({low[0]} {high[1]} {low[2]}) ({low[0]} {low[1]} 0)({high[0]} {low[1]} 0)({high[0]} {high[1]} 0)({low[0]} {high[1]} 0) ({low[0]} {low[1]} {high[2]})({high[0]} {low[1]} {high[2]})({high[0]} {high[1]} {high[2]})({low[0]} {high[1]} {high[2]})"
        block=header("dictionary","blockMeshDict","system")+f"convertToMeters 1; vertices ({vertices}); blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz_water}) simpleGrading (1 1 1) hex (4 5 6 7 8 9 10 11) ({nx} {ny} {nz_air}) simpleGrading (1 1 1)); edges (); boundary (inletWater {{type patch; faces ((0 4 7 3));}} inletAir {{type patch; faces ((4 8 11 7));}} outletWater {{type patch; faces ((1 2 6 5));}} outletAir {{type patch; faces ((5 6 10 9));}} bottom {{type wall; faces ((0 3 2 1));}} atmosphere {{type patch; faces ((8 9 10 11));}} sides {{type wall; faces ((0 1 5 4)(3 7 6 2)(4 5 9 8)(7 11 10 6));}});\n"
        hull_alpha="hull {type zeroGradient;}" if include_hull else ""
        alpha=header("volScalarField","alpha.water","0")+f"dimensions [0 0 0 0 0 0 0]; internalField uniform 0; boundaryField {{inletWater {{type fixedValue; value uniform 1;}} inletAir {{type fixedValue; value uniform 0;}} outletWater {{type inletOutlet; inletValue uniform 1; value uniform 1;}} outletAir {{type inletOutlet; inletValue uniform 0; value uniform 0;}} atmosphere {{type inletOutlet; inletValue uniform 0; value uniform 0;}} bottom {{type zeroGradient;}} sides {{type zeroGradient;}} {hull_alpha}}}\n"
        hull_p="hull {type fixedFluxPressure; value uniform 0;}" if include_hull else ""
        prgh=header("volScalarField","p_rgh","0")+f"dimensions [1 -1 -2 0 0 0 0]; internalField uniform 0; boundaryField {{inletWater {{type fixedFluxPressure; value uniform 0;}} inletAir {{type fixedFluxPressure; value uniform 0;}} outletWater {{type fixedValue; value uniform 0;}} outletAir {{type fixedValue; value uniform 0;}} atmosphere {{type totalPressure; p0 uniform 0;}} bottom {{type fixedFluxPressure; value uniform 0;}} sides {{type fixedFluxPressure; value uniform 0;}} {hull_p}}}\n"
        vec=f"({velocity[0]} {velocity[1]} {velocity[2]})";hull_u="hull {type noSlip;}" if include_hull else ""
        U=header("volVectorField","U","0")+f"dimensions [0 1 -1 0 0 0 0]; internalField uniform {vec}; boundaryField {{inletWater {{type fixedValue; value uniform {vec};}} inletAir {{type fixedValue; value uniform {vec};}} outletWater {{type inletOutlet; inletValue uniform {vec}; value uniform {vec};}} outletAir {{type inletOutlet; inletValue uniform {vec}; value uniform {vec};}} atmosphere {{type pressureInletOutletVelocity; value uniform (0 0 0);}} bottom {{type slip;}} sides {{type slip;}} {hull_u}}}\n"
        phase=header("dictionary","phaseProperties","constant")+"phases (water air); sigma 0.072;\n"
        water=header("dictionary","physicalProperties.water","constant")+f"viscosityModel constant; nu {nu}; rho {rho};\n"
        air=header("dictionary","physicalProperties.air","constant")+"viscosityModel constant; nu 1.48e-5; rho 1.225;\n"
        sst=bool(case.turbulence_settings and case.turbulence_settings.model==TurbulenceModel.K_OMEGA_SST)
        transport=(header("dictionary","momentumTransport","constant")+
            ("simulationType RAS; RAS {model kOmegaSST; turbulence on; printCoeffs on;}\n"
             if sst else "simulationType laminar;\n"))
        gravity=header("uniformDimensionedVectorField","g","constant")+f"dimensions [0 1 -2 0 0 0 0]; value (0 0 {-g});\n"
        set_fields=header("dictionary","setFieldsDict","system")+f"defaultFieldValues (volScalarFieldValue alpha.water 0); regions (boxToCell {{box ({low[0]} {low[1]} {low[2]}) ({high[0]} {high[1]} 0); fieldValues (volScalarFieldValue alpha.water 1);}});\n"
        turbulent_div=("div(phi,k) Gauss linearUpwind limitedGrad; div(phi,omega) Gauss linearUpwind limitedGrad; "
                       if sst else "")
        schemes=header("dictionary","fvSchemes","system")+f"ddtSchemes {{default Euler;}} gradSchemes {{default Gauss linear;}} divSchemes {{default none; div(rhoPhi,U) Gauss linearUpwind grad(U); div(phi,alpha) Gauss interfaceCompression vanLeer 1; {turbulent_div}div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;}} laplacianSchemes {{default Gauss linear corrected;}} interpolationSchemes {{default linear;}} snGradSchemes {{default corrected;}} wallDist {{method meshWave;}}\n"
        turbulent_solvers=("k {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1;} kFinal {$k; relTol 0;} "
                           "omega {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1;} omegaFinal {$omega; relTol 0;} "
                           if sst else "")
        solution=header("dictionary","fvSolution","system")+f"solvers {{\"alpha.water.*\" {{nAlphaCorr 2; nAlphaSubCycles 1; MULESCorr yes; nLimiterIter 5; solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;}} \"pcorr.*\" {{solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0;}} p_rgh {{solver GAMG; smoother DIC; tolerance 1e-7; relTol 0.01;}} p_rghFinal {{$p_rgh; relTol 0;}} U {{solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.1;}} {turbulent_solvers}}} PIMPLE {{momentumPredictor no; nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 1;}} relaxationFactors {{equations {{\".*\" 1;}}}}\n"
        control=header("dictionary","controlDict","system")+f"application foamRun; solver incompressibleVoF; startFrom startTime; startTime 0; stopAt endTime; endTime {case.solver_settings.end_time_s}; deltaT {case.solver_settings.initial_timestep_s}; adjustTimeStep yes; maxCo {case.solver_settings.max_courant}; maxAlphaCo {case.solver_settings.max_alpha_courant}; maxDeltaT {case.solver_settings.timestep_s}; writeControl adjustableRunTime; writeInterval {case.solver_settings.write_interval_s}; writeFormat ascii; writePrecision 10; runTimeModifiable false;\n"
        if include_hull:
            center=point_to_foam(case.reference_point_frd_m,case.waterline_z_m)
            control+=f"functions {{forces {{type forces; libs (\"libforces.so\"); patches (hull); rho rho; CofR ({center[0]} {center[1]} {center[2]}); writeControl timeStep; writeInterval 1;}}"
            if sst:
                control+=" yPlus {type yPlus; libs (\"libfieldFunctionObjects.so\"); executeControl writeTime; writeControl writeTime;}"
            control+="}\n"
        files={"system/blockMeshDict":block,"system/controlDict":control,"0/U":U,"0/alpha.water":alpha,"0/p_rgh":prgh,"constant/phaseProperties":phase,
            "constant/physicalProperties.water":water,"constant/physicalProperties.air":air,"constant/momentumTransport":transport,"constant/g":gravity,
            "system/setFieldsDict":set_fields,"system/fvSchemes":schemes,"system/fvSolution":solution}
        if sst:
            values=case.turbulence_settings
            def turbulence_field(name,dimensions,value,hull_type):
                inlet=f"type fixedValue; value uniform {value};"
                outlet=f"type inletOutlet; inletValue uniform {value}; value uniform {value};"
                hull=f"hull {{type {hull_type}; value uniform {value};}}" if include_hull else ""
                return (header("volScalarField",name,"0")+
                    f"dimensions {dimensions}; internalField uniform {value}; boundaryField {{"
                    f"inletWater {{{inlet}}} inletAir {{{inlet}}} "
                    f"outletWater {{{outlet}}} outletAir {{{outlet}}} "
                    f"atmosphere {{{outlet}}} bottom {{type zeroGradient;}} "
                    f"sides {{type zeroGradient;}} {hull}}}\n")
            files["0/k"]=turbulence_field("k","[0 2 -2 0 0 0 0]",values.inlet_k_m2_s2,"kqRWallFunction")
            files["0/omega"]=turbulence_field("omega","[0 0 -1 0 0 0 0]",values.inlet_omega_s_inv,"omegaWallFunction")
            files["0/nut"]=turbulence_field("nut","[0 2 -1 0 0 0 0]",values.inlet_nut_m2_s,"nutkWallFunction")
        return files

    def generate_free_surface_precheck(self, root: Path, case: IdentificationCase, *, geometry=None) -> Path:
        """Generate CFD-000 (empty) or CFD-001 (fixed hull) using production dictionaries."""
        root=Path(root)
        if root.exists(): raise CFDExecutionError(f"precheck already exists: {root}")
        for folder in (root/"0",root/"constant",root/"system"): folder.mkdir(parents=True,exist_ok=True)
        files=self._free_surface_files(case,(0.,0.,0.),include_hull=geometry is not None)
        if geometry is not None:
            (root/"constant"/"triSurface").mkdir()
            (root/"constant"/"triSurface"/"hull.stl").write_bytes(self._waterline_aligned_stl(
                Path(geometry.path).read_bytes(),case.waterline_z_m))
            files["system/snappyHexMeshDict"]=self._case_files(0)["system/snappyHexMeshDict"].replace("locationInMesh (4 0 0)","locationInMesh (4 0 0.5)")
        for name,text in files.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        (root/"case_metadata.json").write_text(json.dumps({"case_id":root.name,"solver":"foamRun -solver incompressibleVoF","precheck":True,"with_hull":geometry is not None},indent=2))
        return root

    def mesh_case(self, root: Path) -> dict:
        commands=[("blockMesh",["blockMesh"])]
        if (root/"system/snappyHexMeshDict").exists(): commands.append(("snappyHexMesh",["snappyHexMesh","-overwrite"]))
        rotating=(root/"system/createBafflesDict").exists()
        if rotating:
            commands += [("createBaffles",["createBaffles","-overwrite"]),
                         ("splitBaffles",["splitBaffles","-overwrite"]),
                         ("createNonConformalCouples",["createNonConformalCouples","-overwrite",
                             "nonCouple1","nonCouple2"])]
        commands.append(("checkMesh",["checkMesh","-allGeometry","-allTopology"]))
        outputs={}
        for label,command in commands:
            started=time.monotonic(); result=self._container(command,case_dir=root,capture=True)
            log=root/("mesh.log" if label!="checkMesh" else "checkMesh.log")
            mode="a" if log.exists() else "w"; log.open(mode).write(f"$ {' '.join(command)}\n{result.stdout}{result.stderr}")
            outputs[label]={"exit_code":result.returncode,"seconds":time.monotonic()-started}
            if result.returncode or "FOAM FATAL" in result.stdout+result.stderr:
                raise CFDExecutionError(f"{label} failed; see {log}")
        check=(root/"checkMesh.log").read_text()
        cells=re.findall(r"cells:\s+(\d+)",check)
        region_marker="Number of regions: 2" if rotating else "Number of regions: 1 (OK)"
        required=("Boundary definition OK",region_marker,"Cell volumes OK",
                  "Non-orthogonality check OK","Max skewness")
        fatal_failures=re.findall(r"\*\*\*([^\n]+)",check)
        accepted_concavity=all("Concave cells" in item or
            (rotating and "Cells with small determinant" in item) for item in fatal_failures)
        if rotating:
            determinant_counts=[int(value) for value in re.findall(
                r"Cells with small determinant .*number of cells: (\d+)",check)]
            if determinant_counts and determinant_counts[-1]>.03*int(cells[0]):
                raise CFDExecutionError("too many small-determinant cells in rotating mesh")
            ncc=(root/"mesh.log").read_text()
            if ("Adding nonConformalCyclic interfaces" not in ncc or
                    "fvMeshStitcher: Connecting" not in ncc or
                    not (root/"constant/polyMesh/cellZones").exists()):
                raise CFDExecutionError("steady-yaw NCC mesh did not initialize")
        if (not cells or int(cells[-1])<=0 or not all(item in check for item in required)
                or not accepted_concavity or "FOAM FATAL" in check):
            raise CFDExecutionError("checkMesh did not report a usable mesh")
        outputs["cell_count"]=int(cells[0]); outputs["accepted_quality_findings"]=fatal_failures
        return outputs

    def solve_case(self, root: Path, *, resume: bool=False, parallel_ranks: int=1) -> dict:
        root=Path(root)
        if parallel_ranks<1: raise CFDExecutionError("parallel_ranks must be positive")
        runtime_path=root/"openfoam_runtime.json"
        runtime=self.verify_runtime()
        if resume and runtime_path.exists() and json.loads(runtime_path.read_text())!=runtime:
            raise CFDExecutionError("OpenFOAM runtime changed across restart")
        runtime_path.write_text(json.dumps(runtime,sort_keys=True,indent=2))
        if not resume and (root/"system/setFieldsDict").exists():
            initialized=self._container(["setFields"],case_dir=root,capture=True)
            (root/"setFields.log").write_text(initialized.stdout+initialized.stderr)
            if initialized.returncode: raise CFDExecutionError("free-surface initialization failed")
        solver="incompressibleFluid"
        metadata=root/"case_metadata.json"
        if metadata.exists() and "incompressibleVoF" in json.loads(metadata.read_text()).get("solver",""): solver="incompressibleVoF"
        if resume and "startFrom latestTime" not in (root/"system/controlDict").read_text():
            raise CFDExecutionError("resume requires startFrom latestTime in controlDict")
        # Stream each continuation to a durable log while the solver runs.
        # Capturing stdout only in memory loses every diagnostic if a long run
        # is interrupted and can leave no auditable Courant/residual history.
        previous=[float(path.name) for path in root.iterdir() if path.is_dir() and
                  re.fullmatch(r"\d+(?:\.\d+)?",path.name)]
        start_time=max(previous,default=0.) if resume else 0.
        segment_dir=root/"solver_segments";segment_dir.mkdir(exist_ok=True)
        segment=segment_dir/f"from_{start_time:g}_{int(time.time())}.log"
        started=time.monotonic()
        command=["foamRun","-solver",solver]
        if parallel_ranks>1:
            (root/"system/decomposeParDict").write_text(
                "FoamFile {version 2.0; format ascii; class dictionary; object decomposeParDict;}\n"
                f"numberOfSubdomains {parallel_ranks}; method scotch;\n")
            decompose=["decomposePar","-force"]+(["-latestTime"] if resume else [])
            prepared=self._container(decompose,case_dir=root,capture=True)
            (root/"decomposePar.log").write_text(prepared.stdout+prepared.stderr)
            if prepared.returncode: raise CFDExecutionError("OpenFOAM domain decomposition failed")
            command=["mpirun","--allow-run-as-root","-np",str(parallel_ranks),
                "foamRun","-parallel","-solver",solver]
        args=self._container_args(command,case_dir=root)
        with segment.open("w") as segment_log, (root/"solver.log").open("a" if resume else "w") as joined_log:
            process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                segment_log.write(line);segment_log.flush()
                joined_log.write(line);joined_log.flush()
            exit_code=process.wait()
        elapsed=time.monotonic()-started; text=segment.read_text()
        bad=re.search(r"FOAM FATAL|floating point exception(?! trapping)|segmentation fault|\b(?:nan|inf)\b",text,re.I|re.M)
        completed=re.search(r"\nEnd\s*\n",text) is not None
        converged=completed and (solver=="incompressibleVoF" or "SIMPLE solution converged" in text)
        if exit_code or bad or not converged:
            raise CFDExecutionError("OpenFOAM solver failed or did not satisfy convergence criteria")
        if (root/"constant/dynamicMeshDict").exists():
            case_metadata=json.loads((root/"case_metadata.json").read_text())
            rotating=case_metadata.get("rotation_topology",{}).get("method")=="OpenFOAM11-solidBody-NCC"
            if "Selecting fvMeshMover motionSolver" not in text:
                raise CFDExecutionError("prescribed hull motion did not initialize")
            if rotating:
                if ("Selecting motion solver: solidBody" not in text or
                    "patchToPatch: Calculating couplings" not in text or
                    "nonConformalCyclic_on_nonCouple1 min/avg/max mesh flux error" not in text):
                    raise CFDExecutionError("rigid steady-yaw mover or NCC coupling is missing")
            elif "Solving for cellDisplacement" not in text:
                raise CFDExecutionError("prescribed hull motion did not initialize")
            initial_points=root/"constant/polyMesh/points"
            moved_points=list(root.glob("[0-9]*/polyMesh/points"))+list(root.glob("[0-9]*.[0-9]*/polyMesh/points"))
            if not initial_points.exists():
                raise CFDExecutionError("initial mesh points are missing")
            baseline=self._mesh_points(initial_points)
            changed=False
            for path in moved_points:
                current=self._mesh_points(path)
                if current.shape!=baseline.shape:
                    raise CFDExecutionError("prescribed motion changed mesh point count")
                if np.max(np.abs(current-baseline))>1e-8:
                    changed=True;break
            if not changed:
                raise CFDExecutionError("prescribed hull motion wrote no changed mesh points")
            if rotating:
                topology=case_metadata["rotation_topology"]
                ids=hull_point_indices(root/"constant/polyMesh")
                if len(ids)<3: raise CFDExecutionError("rotating hull patch has too few points")
                for path in moved_points:
                    try: instant=float(path.parent.parent.name)
                    except ValueError: continue
                    error=yaw_hull_point_error(baseline,self._mesh_points(path),ids,
                        center=tuple(topology["center_foam_m"]),
                        angle_rad=float(case_metadata["magnitude"])*instant)
                    if error>1e-5:
                        raise CFDExecutionError(f"steady-yaw hull rotation error {error:g} m at {instant:g} s")
            mesh_check=self._container(["checkMesh","-latestTime","-allGeometry","-allTopology"],
                                       case_dir=root,capture=True)
            (root/"checkMesh.latest.log").write_text(mesh_check.stdout+mesh_check.stderr)
            if (mesh_check.returncode or "Cell volumes OK" not in mesh_check.stdout or
                    "Non-orthogonality check OK" not in mesh_check.stdout or
                    "FOAM FATAL" in mesh_check.stdout+mesh_check.stderr):
                raise CFDExecutionError("deformed latest-time mesh failed checkMesh")
            if rotating:
                try:
                    validate_rotation_mesh_quality((root/"checkMesh.log").read_text(),mesh_check.stdout)
                except ValueError as error:
                    raise CFDExecutionError(str(error)) from error
                coverages=[float(value) for value in re.findall(
                    r"(?:Source|Target) min/average/max coverage = [-+0-9.eE]+/([-+0-9.eE]+)/",text)]
                if not coverages or min(coverages)<.995:
                    raise CFDExecutionError("NCC interface coverage is inadequate")
        if parallel_ranks>1:
            self.reconstruct_checkpoint(root)
        return {"exit_code":exit_code,"seconds":elapsed,"converged":True,"segment_log":str(segment)}

    def reconstruct_checkpoint(self, root: Path) -> Path:
        """Make the latest parallel checkpoint available for a later restart."""
        root=Path(root)
        reconstructed=self._container(["reconstructPar","-latestTime"],case_dir=root,capture=True)
        log=root/"reconstructPar.log"
        log.write_text(reconstructed.stdout+reconstructed.stderr)
        if reconstructed.returncode: raise CFDExecutionError("OpenFOAM parallel state reconstruction failed")
        return log

    @staticmethod
    def force_history(root: Path) -> list[list[float]]:
        number=r"[-+0-9.eE]+"
        force_files=sorted(Path(root).glob("postProcessing/forces/*/forces.dat"),
                           key=lambda path: float(path.parent.name))
        samples={}
        for index,force_file in enumerate(force_files):
            next_start=(float(force_files[index+1].parent.name) if index+1<len(force_files)
                        else float("inf"))
            for line in force_file.read_text().splitlines():
                if not line.strip() or line.lstrip().startswith("#"): continue
                vectors=re.findall(rf"\(({number})\s+({number})\s+({number})\)",line)
                if len(vectors)==4:
                    value=np.asarray(vectors,float);sample_time=float(line.split()[0])
                    if sample_time<next_start:
                        samples[sample_time]=[sample_time,*((value[0]+value[1]).tolist()),*((value[2]+value[3]).tolist())]
        return [samples[sample_time] for sample_time in sorted(samples)]

    @staticmethod
    def _stitched_solver_text(root: Path) -> str:
        segments=sorted((Path(root)/"solver_segments").glob("from_*.log"),
                        key=lambda path: int(path.stem.rsplit("_",1)[1]))
        if not segments: return (Path(root)/"solver.log").read_text()
        stitched=[]
        for index,path in enumerate(segments):
            next_start=(float(segments[index+1].stem.split("_")[1])
                        if index+1<len(segments) else float("inf"))
            step=[];step_time=None
            for line in path.read_text().splitlines(keepends=True):
                match=re.match(r"^Time = ([0-9.eE+-]+)s?$",line.strip())
                if match:
                    if step and (step_time is None or step_time<next_start): stitched.extend(step)
                    step=[line];step_time=float(match.group(1))
                else: step.append(line)
            if step and (step_time is None or step_time<next_start): stitched.extend(step)
        return "".join(stitched)

    @staticmethod
    def diagnose_case(root: Path) -> dict:
        text=OpenFOAMAdapter._stitched_solver_text(root);number=r"[-+0-9.eE]+"
        times=[float(x) for x in re.findall(rf"^Time = ({number})s?$",text,re.M)]
        mean_co=[float(x) for x in re.findall(rf"Courant Number mean: ({number})",text)]
        max_co=[float(x) for x in re.findall(rf"Courant Number mean: {number} max: ({number})",text)]
        alpha_mean=[float(x) for x in re.findall(rf"Interface Courant Number mean: ({number})",text)]
        alpha_max=[float(x) for x in re.findall(rf"Interface Courant Number mean: {number} max: ({number})",text)]
        actual_timestep=[float(x) for x in re.findall(rf"^deltaT = ({number})$",text,re.M)]
        volume=[float(x) for x in re.findall(rf"Phase-1 volume fraction = ({number})",text)]
        residuals={}
        for field,initial,final in re.findall(rf"Solving for ([^,]+), Initial residual = ({number}), Final residual = ({number})",text):
            residuals.setdefault(field,[]).append((float(initial),float(final)))
        # PIMPLE performs multiple pressure corrections in one timestep. Its
        # first correction may be above tolerance even when the final
        # correction is converged; preserve both histories explicitly.
        step_final_residuals={};step={};in_step=False
        for line in text.splitlines():
            if re.match(rf"^Time = {number}s?$",line):
                if in_step:
                    for field,value in step.items(): step_final_residuals.setdefault(field,[]).append(value)
                step={};in_step=True
            match=re.search(rf"Solving for ([^,]+), Initial residual = ({number}), Final residual = ({number})",line)
            if in_step and match: step[match.group(1)]=float(match.group(3))
        if in_step:
            for field,value in step.items(): step_final_residuals.setdefault(field,[]).append(value)
        # Include all restart segments; a single file hides later load drift.
        forces=OpenFOAMAdapter.force_history(root)
        yplus_files=sorted(Path(root).glob("[0-9]*/yPlus"),
            key=lambda path: float(path.parent.name))
        yplus=None
        if yplus_files:
            hull_section=yplus_files[-1].read_text().split("    hull\n",1)
            if len(hull_section)==2:
                match=re.search(r"nonuniform List<scalar>\s*(\d+)\s*\((.*?)\)",
                    hull_section[1],re.S)
                if match:
                    values=np.fromstring(match.group(2),sep=" ")
                    if len(values)==int(match.group(1)) and len(values):
                        yplus={"time_s":float(yplus_files[-1].parent.name),
                            "min":float(values.min()),"median":float(np.median(values)),
                            "p95":float(np.percentile(values,95)),
                            "p99":float(np.percentile(values,99)),"max":float(values.max()),
                            "fraction_30_to_300":float(np.mean((values>=30)&(values<=300)))}
        segments=sorted((Path(root)/"solver_segments").glob("from_*.log"),
            key=lambda path: int(path.stem.rsplit("_",1)[1]))
        final_segment=(segments[-1].read_text() if segments else text)
        return {"time":times,"timestep":actual_timestep,"courant_mean":mean_co,"courant_max":max_co,
            "alpha_courant_mean":alpha_mean,"alpha_courant_max":alpha_max,"water_volume_fraction":volume,
            "residuals":residuals,"step_final_residuals":step_final_residuals,
            "force_moment_history":forces,"hull_y_plus":yplus,
            "solver_completed":bool(re.search(r"\nEnd\s*\n",final_segment))}

    def parse_case(self, root: Path, *, debug_last_sample: bool=False) -> OpenFOAMResult:
        files=sorted(root.glob("postProcessing/forces/*/forces.dat"))
        if not files: raise CFDExecutionError("OpenFOAM force/moment output is missing")
        raw=files[-1].read_bytes(); lines=[x for x in raw.decode().splitlines() if x.strip() and not x.lstrip().startswith("#")]
        if not lines: raise CFDExecutionError("OpenFOAM force/moment output contains no samples")
        if "forces(pressure viscous)" not in raw.decode() or "moments(pressure viscous)" not in raw.decode():
            raise CFDExecutionError("OpenFOAM force output channel declaration is missing")
        # time, (pressure force, viscous force), (pressure moment, viscous moment)
        vectors=re.findall(r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)",lines[-1])
        if len(vectors)!=4: raise CFDExecutionError("OpenFOAM force output is missing required six-axis channels")
        values=np.asarray(vectors,float)
        force=values[0]+values[1]; moment=values[2]+values[3]
        fitting_window=None
        qualification_path=root/"qualified_fitting_window.json"
        if qualification_path.exists():
            qualification=json.loads(qualification_path.read_text())
            if qualification.get("status")!="CFD_QUALIFIED":
                raise CFDExecutionError("qualified fitting-window manifest is not accepted")
            window=qualification.get("fitting_window_s")
            if (not isinstance(window,list) or len(window)!=2 or
                not all(isinstance(value,(int,float)) and math.isfinite(value) for value in window) or
                window[0]>=window[1]):
                raise CFDExecutionError("qualified fitting-window bounds are invalid")
            fitting_window=(float(window[0]),float(window[1]))
            history=np.asarray(self.force_history(root),dtype="<f8")
            selected=history[(history[:,0]>=fitting_window[0])&
                             (history[:,0]<=fitting_window[1])]
            if (len(selected)<8 or selected[0,0]>fitting_window[0]+.001 or
                selected[-1,0]<fitting_window[1]-.001):
                raise CFDExecutionError("qualified fitting-window samples are incomplete")
            selected_sha=hashlib.sha256(selected.tobytes()).hexdigest()
            if selected_sha!=qualification.get("selected_samples_sha256"):
                raise CFDExecutionError("qualified fitting-window sample hash mismatch")
            average=selected[:,1:].mean(axis=0)
            declared_mean=np.asarray(qualification.get("mean_six_axis_wrench",()),float)
            if declared_mean.shape!=(6,) or not np.allclose(
                    average,declared_mean,rtol=1e-10,atol=1e-10):
                raise CFDExecutionError("qualified fitting-window mean wrench mismatch")
            force=average[:3];moment=average[3:]
        elif not debug_last_sample:
            raise CFDExecutionError("production parsing requires a qualified fitting-window manifest")
        if not np.isfinite(force).all() or not np.isfinite(moment).all(): raise CFDExecutionError("nonfinite OpenFOAM force output")
        metadata=json.loads((root/"case_metadata.json").read_text()); solver_log=(root/"solver.log").read_text()
        if not debug_last_sample and metadata.get("frame_contract")!=CONVENTION:
            raise CFDExecutionError("case has no verified frame contract")
        if metadata.get("frame_contract")==CONVENTION:
            declared=metadata.get("units")
            if declared != {"length":"m","velocity":"m/s","angular_rate":"rad/s",
                            "force":"N","moment":"N*m"} and not debug_last_sample:
                raise CFDExecutionError("case has no verified SI units")
            center=re.search(r"^#\s*CofR\s*:\s*\(([^)]+)\)",raw.decode(),re.M)
            if center is not None:
                observed=np.fromstring(center.group(1),sep=" ")
                expected=np.asarray(metadata["solver_moment_reference_point_m"],float)
                if observed.shape!=(3,) or not np.allclose(observed,expected,rtol=0,atol=1e-8):
                    raise CFDExecutionError("forces.dat CofR differs from case metadata")
        if fitting_window is not None and qualification.get("case_id")!=metadata["case_id"]:
            raise CFDExecutionError("qualified fitting-window case ID mismatch")
        if not re.search(r"\nEnd\s*\n",solver_log): raise CFDExecutionError("solver completion marker is missing")
        velocity=tuple(metadata.get("velocity_body_frd",(metadata.get("flow_speed_mps",0.),0.,0.)))
        angular=velocity[3:] if len(velocity)==6 else (0.,0.,0.)
        translation=velocity[:3]
        if metadata.get("frame_contract")==CONVENTION:
            force,moment=foam_wrench_to_body(force,moment,
                foam_reference=metadata["solver_moment_reference_point_m"],
                body_reference=metadata["moment_reference_point_frd_m"],
                waterline_z_m=metadata["source_waterline_z_m"])
        return OpenFOAMResult(metadata["case_id"],translation,angular,tuple(force),tuple(moment),
            {"force":"N","moment":"N*m","velocity":"m/s","angular_rate":"rad/s"},"body_FRD",
            tuple(metadata["moment_reference_point_frd_m"]),str(root),
            (selected_sha if fitting_window is not None else hashlib.sha256(raw).hexdigest()),
            metadata["solver"],True,0.,fitting_window,
            (CONVENTION if metadata.get("frame_contract")==CONVENTION else "legacy-unqualified"),
            ("qualified_window" if fitting_window is not None else "debug_last_sample"))

    @staticmethod
    def _case_files(speed: float) -> dict[str,str]:
        header=lambda cls,obj,location: f'''FoamFile\n{{\n version 2.0; format ascii; class {cls}; location "{location}"; object {obj};\n}}\n'''
        block=header("dictionary","blockMeshDict","system")+'''convertToMeters 1;\nvertices ((-3 -2 -1)(5 -2 -1)(5 2 -1)(-3 2 -1)(-3 -2 1)(5 -2 1)(5 2 1)(-3 2 1));\nblocks (hex (0 1 2 3 4 5 6 7) (32 20 16) simpleGrading (1 1 1));\nedges ();\nboundary (inlet {type patch; faces ((0 4 7 3));} outlet {type patch; faces ((1 2 6 5));} sides {type wall; faces ((0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));});\n'''
        snappy=header("dictionary","snappyHexMeshDict","system")+'''castellatedMesh true; snap true; addLayers false;\ngeometry { hull.stl {type triSurfaceMesh; name hull;} }\ncastellatedMeshControls {maxLocalCells 300000; maxGlobalCells 600000; minRefinementCells 0; nCellsBetweenLevels 2; features (); refinementSurfaces {hull {level (2 2); patchInfo {type wall;}}} resolveFeatureAngle 30; refinementRegions {}; locationInMesh (4 0 0); allowFreeStandingZoneFaces true;}\nsnapControls {nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;}\naddLayersControls {relativeSizes true; layers {};}\nmeshQualityControls {#includeEtc "caseDicts/mesh/generation/meshQualityDict"}\nmergeTolerance 1e-6;\n'''
        control=header("dictionary","controlDict","system")+'''application foamRun; solver incompressibleFluid; startFrom startTime; startTime 0; stopAt endTime; endTime 300; deltaT 1; writeControl timeStep; writeInterval 100; writeFormat ascii; writePrecision 10; runTimeModifiable false;\nfunctions { forces {type forces; libs ("libforces.so"); patches (hull); rho rhoInf; rhoInf 1000; CofR (0 0 0); writeControl timeStep; writeInterval 1;} }\n'''
        schemes=header("dictionary","fvSchemes","system")+'''ddtSchemes {default steadyState;} gradSchemes {default Gauss linear; grad(U) cellLimited Gauss linear 1;} divSchemes {default none; div(phi,U) bounded Gauss linearUpwindV grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;}\n'''
        solution=header("dictionary","fvSolution","system")+'''solvers {p {solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01;} Phi {$p;} U {solver smoothSolver; smoother GaussSeidel; tolerance 1e-8; relTol 0.05; nSweeps 1;}} SIMPLE {nNonOrthogonalCorrectors 1; consistent yes; residualControl {p 1e-4; U 1e-5;}} relaxationFactors {equations {U 0.7;}} cache {grad(U);}\n'''
        physical=header("dictionary","physicalProperties","constant")+'''viscosityModel constant; nu [0 2 -1 0 0 0 0] 1e-6;\n'''
        transport=header("dictionary","momentumTransport","constant")+'''simulationType laminar;\n'''
        U=header("volVectorField","U","0")+f'''dimensions [0 1 -1 0 0 0 0]; internalField uniform ({speed} 0 0); boundaryField {{inlet {{type fixedValue; value uniform ({speed} 0 0);}} outlet {{type inletOutlet; inletValue uniform ({speed} 0 0); value uniform ({speed} 0 0);}} sides {{type slip;}} hull {{type noSlip;}}}}\n'''
        p=header("volScalarField","p","0")+'''dimensions [0 2 -2 0 0 0 0]; internalField uniform 0; boundaryField {inlet {type zeroGradient;} outlet {type fixedValue; value uniform 0;} sides {type zeroGradient;} hull {type zeroGradient;}}\n'''
        return {"system/blockMeshDict":block,"system/snappyHexMeshDict":snappy,"system/controlDict":control,
            "system/fvSchemes":schemes,"system/fvSolution":solution,"constant/physicalProperties":physical,
            "constant/momentumTransport":transport,"0/U":U,"0/p":p}
