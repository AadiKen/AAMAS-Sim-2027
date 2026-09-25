"""End-to-end production identification workflow, independent of a CFD backend."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

from .fitting import CoefficientFitter, ForceMomentDataset, Plant6ModelForm
from .generation import VesselFactory
from .hydrostatics import derive_hydrostatics, load_ascii_stl
from .models import CanonicalVessel, GeometryReference, ParameterLineage
from .package import write_vessel_package
from .frame_contract import CONVENTION
from .cfd import OpenFOAMResult


@dataclass(frozen=True)
class AcceptedObservation:
    case_id: str
    velocity: tuple[float,...]
    acceleration: tuple[float,...]
    wrench: tuple[float,...]
    accepted: bool
    frame_contract: str = ""
    fitting_window_s: tuple[float,float] | None = None
    provenance_verified: bool = False

    @classmethod
    def from_openfoam(cls, result: OpenFOAMResult, *, acceleration=(0., 0., 0., 0., 0., 0.)):
        if result.convention != CONVENTION or result.sample_mode != "qualified_window" or result.fitting_window_s is None:
            raise ValueError("OpenFOAM result is not qualified under the frame contract")
        velocity = result.velocity_body_frd_mps + result.angular_rate_body_frd_radps
        wrench = result.force_body_frd_n + result.moment_body_frd_nm
        return cls(result.case_id, velocity, tuple(acceleration), wrench, True,
                   result.convention, result.fitting_window_s)


def identify_package(*, name: str, geometry_path: str|Path, mass_kg: float,
                     cg_frd_m: tuple[float,float,float], observations: tuple[AcceptedObservation,...],
                     output: str|Path, solver: dict, coupled_terms=(),
                     unsafe_debug_observations: bool=False) -> Path:
    selected=tuple(item for item in observations if item.accepted)
    if not selected: raise ValueError("no quality-accepted CFD observations")
    if not unsafe_debug_observations and any(not item.provenance_verified for item in selected):
        raise ValueError("production fitting requires verified qualified-observation artifacts")
    if any(item.frame_contract != CONVENTION or item.fitting_window_s is None or
           len(item.fitting_window_s) != 2 or item.fitting_window_s[0] >= item.fitting_window_s[1]
           for item in selected):
        raise ValueError("accepted observations require the frame contract and qualified fitting window")
    path=Path(geometry_path);content=path.read_bytes();digest=hashlib.sha256(content).hexdigest()
    points,faces=load_ascii_stl(path);extents=np.ptp(points,axis=0)
    hydro=derive_hydrostatics(points,faces,mass_kg=mass_kg,cg_frd_m=cg_frd_m)
    dataset=ForceMomentDataset(np.asarray([x.velocity for x in selected]),np.asarray([x.acceleration for x in selected]),np.asarray([x.wrench for x in selected]))
    form=Plant6ModelForm.discover(coupled_terms);fit=CoefficientFitter(bounds=(-np.inf,np.inf)).fit_plant6(dataset,model_form=form,source_cases=tuple(x.case_id for x in selected))
    inertia=(mass_kg*(extents[1]**2+extents[2]**2)/12,mass_kg*(extents[0]**2+extents[2]**2)/12,mass_kg*(extents[0]**2+extents[1]**2)/12)
    lineage=ParameterLineage(source_kind="geometry-derived",source_id=digest,uncertainty=.05)
    cfd_lineage=ParameterLineage(source_kind="CFD-derived",source_id=hashlib.sha256("".join(x.case_id for x in selected).encode()).hexdigest(),uncertainty=max((e.uncertainty for e in fit.estimates),default=0))
    diagonal=tuple(float(fit.linear_damping[i][i]) for i in range(6))
    coupled_payload=tuple({"output_axis":term.output_axis,"factors":term.factors,
        "absolute_factors":term.absolute_factors,"coefficient":fit.coupled_damping[index]}
        for index,term in enumerate(form.coupled_terms))
    vessel=CanonicalVessel(id=name,version="1",geometry=GeometryReference(format="stl",content_hash=digest,
        length_m=float(extents[0]),beam_m=float(extents[1]),draft_m=hydro.draft_m),
        collision={"kind":"box","half_extents_m":tuple(float(x/2) for x in extents)},mass_kg=mass_kg,cg_frd_m=cg_frd_m,
        inertia_cg_kg_m2=((inertia[0],0,0),(0,inertia[1],0),(0,0,inertia[2])),added_mass_kg=fit.added_mass,
        linear_damping=diagonal,linear_damping_matrix=fit.linear_damping,quadratic_damping=fit.quadratic_damping,
        coupled_damping_terms=coupled_payload,
        buoyancy_n=mass_kg*9.80665,center_buoyancy_frd_m=hydro.center_of_buoyancy_frd_m,
        hydrostatics={"model":"linear_matrix","stiffness_6x6":hydro.stiffness_6x6,
            "equilibrium":{"position_ned_m":[0,0,0],"orientation_rpy_rad":[0,0,0]},"reference_point_frd_m":[0,0,0],
            "validity":{"max_abs_roll_rad":.35,"max_abs_pitch_rad":.35}},equilibrium_heave_roll_pitch=(0,0,0),
        max_abs_nu=(15,10,8,3,3,3),max_substep_s=.05,
        provenance={"mass_kg":lineage,"cg_frd_m":lineage,"inertia_cg_kg_m2":lineage,"added_mass_kg":cfd_lineage,
            "linear_damping":cfd_lineage,"quadratic_damping":cfd_lineage,"buoyancy_n":lineage})
    geometry={"sha256":digest,"path":str(path),"extents_m":extents.tolist(),"hydrostatics":hydro.to_dict()}
    return write_vessel_package(output,vessel,fit,geometry_details=geometry,case_ids=tuple(x.case_id for x in selected),solver=solver)


def read_observations(path: str|Path, *, unsafe_debug: bool=False) -> tuple[AcceptedObservation,...]:
    if unsafe_debug:
        values=json.loads(Path(path).read_text())
        return tuple(AcceptedObservation(**item) for item in values)
    from .qualified_observations import verify_artifact
    paths=(sorted(Path(path).glob("*.json")) if Path(path).is_dir() else [Path(path)])
    if not paths: raise ValueError("no qualified observation artifacts")
    accepted=[];seen_cases=set()
    for source in paths:
        artifact=verify_artifact(source)
        if artifact["case_id"] in seen_cases:
            raise ValueError("duplicate qualified case ID in observation set")
        seen_cases.add(artifact["case_id"])
        window=tuple(artifact["fitting_window_s"])
        for sample in artifact["observations"]:
            accepted.append(AcceptedObservation(artifact["case_id"],tuple(sample["velocity"]),
                tuple(sample["acceleration"]),tuple(sample["resisting_wrench"]),True,
                artifact["frame_contract"],window,True))
    return tuple(accepted)
