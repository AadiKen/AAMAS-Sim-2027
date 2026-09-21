"""Backend-neutral CFD adapter with an OpenFOAM baseline and synthetic oracle."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import json
import math
import re
import subprocess
import time

import numpy as np

from .fitting import ForceMomentDataset, synthetic_dataset


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
    adapter_version = "2"
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
        script="source /opt/openfoam11/etc/bashrc && "+" ".join(command)
        args=["docker","run","--rm","--platform","linux/amd64","--entrypoint","/bin/bash"]
        if case_dir is not None:
            args += ["-v",f"{case_dir.resolve()}:/case","-w","/case"]
        args += [self.docker_image,"-lc",script]
        return subprocess.run(args,capture_output=capture,text=True)

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

    def mesh_case(self, root: Path) -> dict:
        commands=(("blockMesh",["blockMesh"]),("snappyHexMesh",["snappyHexMesh","-overwrite"]),
                  ("checkMesh",["checkMesh","-allGeometry","-allTopology"]))
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
        required=("Boundary definition OK","Number of regions: 1 (OK)","Cell volumes OK",
                  "Non-orthogonality check OK","Max skewness")
        fatal_failures=re.findall(r"\*\*\*([^\n]+)",check)
        accepted_concavity=all("Concave cells" in item for item in fatal_failures)
        if (not cells or int(cells[-1])<=0 or not all(item in check for item in required)
                or not accepted_concavity or "FOAM FATAL" in check):
            raise CFDExecutionError("checkMesh did not report a usable mesh")
        outputs["cell_count"]=int(cells[0]); outputs["accepted_quality_findings"]=fatal_failures
        return outputs

    def solve_case(self, root: Path) -> dict:
        started=time.monotonic(); result=self._container(["foamRun","-solver","incompressibleFluid"],case_dir=root,capture=True)
        elapsed=time.monotonic()-started; text=result.stdout+result.stderr; (root/"solver.log").write_text(text)
        bad=re.search(r"FOAM FATAL|floating point exception(?! trapping)|segmentation fault|(^|\s)(nan|inf)(\s|$)",text,re.I|re.M)
        converged="SIMPLE solution converged" in text and re.search(r"\nEnd\s*\n",text) is not None
        if result.returncode or bad or not converged:
            raise CFDExecutionError("OpenFOAM solver failed or did not satisfy convergence criteria")
        return {"exit_code":result.returncode,"seconds":elapsed,"converged":True}

    def parse_case(self, root: Path) -> OpenFOAMResult:
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
        if not np.isfinite(force).all() or not np.isfinite(moment).all(): raise CFDExecutionError("nonfinite OpenFOAM force output")
        metadata=json.loads((root/"case_metadata.json").read_text()); solver_log=(root/"solver.log").read_text()
        if not re.search(r"\nEnd\s*\n",solver_log): raise CFDExecutionError("solver completion marker is missing")
        return OpenFOAMResult(metadata["case_id"],(metadata["flow_speed_mps"],0,0),(0,0,0),tuple(force),tuple(moment),
            {"force":"N","moment":"N*m","velocity":"m/s","angular_rate":"rad/s"},"body_FRD",
            tuple(metadata["moment_reference_point_frd_m"]),str(root),hashlib.sha256(raw).hexdigest(),
            metadata["solver"],True,0.)

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
