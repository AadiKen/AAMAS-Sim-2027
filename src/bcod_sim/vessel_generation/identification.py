"""Backend-neutral definitions and orchestration for offline 6-DOF identification."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Protocol, TypeVar
import hashlib
import json
import math


class MotionType(str, Enum):
    STEADY_VELOCITY = "steady_velocity"
    STEADY_ROTATION = "steady_rotation"
    FORCED_TRANSLATION = "forced_translation"
    FORCED_ROTATION = "forced_rotation"


class DOF(str, Enum):
    SURGE = "surge"; SWAY = "sway"; HEAVE = "heave"
    ROLL = "roll"; PITCH = "pitch"; YAW = "yaw"

    @property
    def index(self) -> int: return tuple(DOF).index(self)

    @property
    def translational(self) -> bool: return self.index < 3


class FluidModel(str, Enum):
    SINGLE_PHASE = "single_phase"
    FREE_SURFACE = "free_surface"


class TurbulenceModel(str, Enum):
    LAMINAR = "laminar"
    K_OMEGA_SST = "kOmegaSST"


@dataclass(frozen=True)
class TurbulenceSettings:
    model: TurbulenceModel = TurbulenceModel.K_OMEGA_SST
    inlet_k_m2_s2: float = 0.00015
    inlet_omega_s_inv: float = 2.0
    inlet_nut_m2_s: float = 5e-7
    smooth_wall: bool = True
    target_y_plus_min: float = 30.0
    target_y_plus_max: float = 300.0


@dataclass(frozen=True)
class WaterProperties:
    density_kg_m3: float = 1025.0
    kinematic_viscosity_m2_s: float = 1.05e-6
    gravity_mps2: float = 9.80665


@dataclass(frozen=True)
class MeshSettings:
    base_cell_size_m: float = 0.25
    hull_refinement_levels: tuple[int, int] = (2, 3)
    boundary_layers: int = 0
    boundary_layer_outer_thickness_m: float = 0.008


@dataclass(frozen=True)
class SolverSettings:
    end_time_s: float = 300.0
    timestep_s: float = 0.01
    initial_timestep_s: float = 0.001
    max_courant: float = 0.5
    max_alpha_courant: float = 0.5
    residual_tolerance: float = 1e-5
    stationarity_window_fraction: float = 0.2
    write_interval_s: float = 1.0


@dataclass(frozen=True)
class DomainSettings:
    minimum_frd_m: tuple[float, float, float] = (-3.0, -2.0, -1.0)
    maximum_frd_m: tuple[float, float, float] = (5.0, 2.0, 1.0)


@dataclass(frozen=True)
class IdentificationCase:
    geometry_hash: str
    motion_type: MotionType
    dof: DOF
    magnitude: float | None = None
    frequency_rad_s: float | None = None
    amplitude: float | None = None
    water_properties: WaterProperties = field(default_factory=WaterProperties)
    fluid_model: FluidModel = FluidModel.SINGLE_PHASE
    turbulence_settings: TurbulenceSettings | None = None
    mesh_settings: MeshSettings = field(default_factory=MeshSettings)
    solver_settings: SolverSettings = field(default_factory=SolverSettings)
    domain_settings: DomainSettings = field(default_factory=DomainSettings)
    reference_point_frd_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cg_frd_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    waterline_z_m: float = 0.0
    openfoam_version: str = "11"
    template_version: str = "7-openfoam11-mover"

    def __post_init__(self) -> None:
        if len(self.geometry_hash) != 64: raise ValueError("geometry_hash must be SHA-256")
        if not math.isfinite(self.waterline_z_m): raise ValueError("waterline_z_m must be finite")
        if self.fluid_model==FluidModel.FREE_SURFACE:
            if self.turbulence_settings is None:
                object.__setattr__(self,"turbulence_settings",TurbulenceSettings())
            if self.mesh_settings==MeshSettings():
                object.__setattr__(self,"mesh_settings",MeshSettings(.5,(3,4),3))
            if self.domain_settings==DomainSettings():
                object.__setattr__(self,"domain_settings",DomainSettings((-12.,-10.,-5.),(22.,10.,5.)))
            if self.turbulence_settings.model==TurbulenceModel.K_OMEGA_SST:
                if self.mesh_settings.boundary_layers<3:
                    raise ValueError("SST surface-vessel CFD requires at least three hull prism layers")
                if not math.isfinite(self.mesh_settings.boundary_layer_outer_thickness_m) or self.mesh_settings.boundary_layer_outer_thickness_m<=0:
                    raise ValueError("SST hull prism-layer thickness must be positive and finite")
                settings=self.turbulence_settings
                if not all(math.isfinite(value) and value>0 for value in
                           (settings.inlet_k_m2_s2,settings.inlet_omega_s_inv,
                            settings.inlet_nut_m2_s,settings.target_y_plus_min,
                            settings.target_y_plus_max)):
                    raise ValueError("SST turbulence and y+ settings must be positive and finite")
                if settings.target_y_plus_max<=settings.target_y_plus_min:
                    raise ValueError("SST y+ maximum must exceed minimum")
        forced = self.motion_type in {MotionType.FORCED_TRANSLATION, MotionType.FORCED_ROTATION}
        if forced != (self.frequency_rad_s is not None and self.amplitude is not None):
            raise ValueError("forced motion requires frequency and amplitude; steady motion does not")
        if forced and (self.frequency_rad_s <= 0 or self.amplitude <= 0):
            raise ValueError("forced-motion frequency and amplitude must be positive")
        if not forced and (self.magnitude is None or not math.isfinite(self.magnitude) or self.magnitude == 0):
            raise ValueError("steady motion requires a finite nonzero signed magnitude")
        if self.dof.translational != (self.motion_type in {MotionType.STEADY_VELOCITY, MotionType.FORCED_TRANSLATION}):
            raise ValueError("motion type and DOF are inconsistent")

    @property
    def case_id(self) -> str:
        definition=asdict(self)
        if self.motion_type==MotionType.STEADY_ROTATION:
            definition["steady_rotation_topology_version"]="openfoam11-solidbody-ncc-v1"
        payload=json.dumps(definition,sort_keys=True,separators=(",",":"),default=lambda x:x.value)
        return hashlib.sha256(payload.encode()).hexdigest()

    def velocity_vector(self) -> tuple[float, ...]:
        values=[0.0]*6
        if self.magnitude is not None: values[self.dof.index]=self.magnitude
        return tuple(values)


@dataclass(frozen=True)
class Sweep:
    motion_type: MotionType
    dofs: tuple[DOF, ...]
    magnitudes: tuple[float, ...] = ()
    amplitudes: tuple[float, ...] = ()
    frequencies_rad_s: tuple[float, ...] = ()

    def cases(self, geometry_hash: str, **settings) -> tuple[IdentificationCase, ...]:
        forced=self.motion_type in {MotionType.FORCED_TRANSLATION,MotionType.FORCED_ROTATION}
        if forced:
            return tuple(IdentificationCase(geometry_hash,self.motion_type,dof,frequency_rad_s=f,
                amplitude=a,**settings) for dof in self.dofs for a in self.amplitudes for f in self.frequencies_rad_s)
        if not self.magnitudes or not any(x<0 for x in self.magnitudes) or not any(x>0 for x in self.magnitudes):
            raise ValueError("steady identification sweeps must contain positive and negative magnitudes")
        return tuple(IdentificationCase(geometry_hash,self.motion_type,dof,magnitude=value,**settings)
                     for dof in self.dofs for value in self.magnitudes)


class CaseCache:
    """Content-addressed cache. Only explicitly accepted results are reusable."""
    def __init__(self, root: str | Path): self.root=Path(root)
    def path(self, case: IdentificationCase) -> Path: return self.root/case.case_id
    def load(self, case: IdentificationCase) -> dict | None:
        path=self.path(case)/"result.json"
        if not path.exists(): return None
        value=json.loads(path.read_text())
        return value if value.get("accepted") is True else None
    def store(self, case: IdentificationCase, result: dict) -> Path:
        if result.get("accepted") is not True: raise ValueError("rejected CFD runs must not enter the valid cache")
        root=self.path(case); root.mkdir(parents=True,exist_ok=True)
        (root/"case.json").write_text(json.dumps(asdict(case),sort_keys=True,indent=2,default=lambda x:x.value))
        (root/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2))
        return root


T = TypeVar("T")
class DispatchBackend(Protocol):
    def map(self, function: Callable[[IdentificationCase], T], cases: Iterable[IdentificationCase]) -> list[T]: ...


class ExecutionBackend(Protocol):
    def execute_identification_case(self, case: IdentificationCase) -> dict: ...


class LocalDispatcher:
    def __init__(self, workers: int | None = None): self.workers=workers
    def map(self, function, cases):
        cases=tuple(cases)
        if self.workers == 1: return [function(case) for case in cases]
        with ProcessPoolExecutor(max_workers=self.workers) as pool: return list(pool.map(function,cases))


class CampaignRunner:
    """Cache-aware orchestration; rejected runs are returned but never cached."""
    def __init__(self, backend: ExecutionBackend, cache: CaseCache, dispatcher: DispatchBackend):
        self.backend,self.cache,self.dispatcher=backend,cache,dispatcher

    def run(self, cases: Iterable[IdentificationCase]) -> dict[str,dict]:
        cases=tuple(cases);results={};pending=[]
        for case in cases:
            cached=self.cache.load(case)
            if cached is None: pending.append(case)
            else: results[case.case_id]=cached
        produced=self.dispatcher.map(self.backend.execute_identification_case,pending)
        for case,result in zip(pending,produced):
            results[case.case_id]=result
            if result.get("accepted") is True: self.cache.store(case,result)
        return results
