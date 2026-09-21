import json
from pathlib import Path

import numpy as np
import pytest

from bcod_sim.config.resolver import resolve
from bcod_sim.vessel_generation.artifacts import write_run_artifact
from bcod_sim.vessel_generation.calibration import calibrate, simulate_maneuver
from bcod_sim.vessel_generation.cfd import (CFDExecutionError, CFDResult, SyntheticCFDAdapter,
    deterministic_test_matrix)
from bcod_sim.vessel_generation.fitting import CalibrationError, CoefficientFitter, synthetic_dataset
from bcod_sim.vessel_generation.generation import VesselFactory
from bcod_sim.vessel_generation.pipeline import calibrate_from_logs, identify_from_cfd
from bcod_sim.web.runtime_factory import build_engine
from bcod_sim.web.service import build_registry


TRUE_ADDED=(18.,22.,25.,3.,4.,8.)
TRUE_LINEAR=(12.,16.,19.,2.,3.,7.)
TRUE_QUAD=(4.,5.,6.,.5,.7,2.)


def generated(name="generated",seed=3):
    return VesselFactory.procedural(name,seed=seed,length_m=4,beam_m=1.4,mass_kg=900)


@pytest.mark.parametrize("length,beam,mass,strength",[(2,.6,80,40),(8,2.5,4500,700),(30,8,90000,9000)])
def test_procedural_schema_generalizes_across_hulls(length,beam,mass,strength):
    vessel=VesselFactory.procedural(f"hull-{length}",seed=17,length_m=length,beam_m=beam,
                                    mass_kg=mass,actuator_strength_n=strength)
    assert vessel.geometry.length_m == length
    assert vessel.components[0].payload["thrust_bounds_n"] == [-strength,strength]
    assert vessel.cg_frd_m == (0,0,0)


def test_all_vessel_creation_paths_share_canonical_schema():
    procedural=generated()
    manual=VesselFactory.manual(procedural.model_dump(mode="json"))
    imported=VesselFactory.imported(procedural.model_dump_json().encode())
    geometry=VesselFactory.geometry("cad",b"solid vessel geometry data",format="stl",length_m=4,beam_m=1.4,draft_m=.5)
    for vessel in (procedural,manual,imported,geometry):
        assert type(vessel) is type(procedural)
        assert vessel.validation_claim == "synthetic/reference validation only"
        assert len(vessel.simulator_definitions()) == 3
    with pytest.raises(ValueError):
        VesselFactory.procedural("bad",seed=0,length_m=2,beam_m=1.5,mass_kg=1)
    with pytest.raises(ValueError): VesselFactory.geometry("bad",b"tiny",format="stl",length_m=4,beam_m=1,draft_m=.4)


def test_geometry_to_cfd_to_coefficients_is_backend_neutral_and_reproducible():
    vessel=generated(); adapter=SyntheticCFDAdapter(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD,seed=9)
    case=adapter.generate_case(vessel.geometry.content_hash,deterministic_test_matrix(seed=9))
    first=adapter.execute(case); second=adapter.execute(case)
    assert np.array_equal(first.dataset.wrench,second.dataset.wrench)
    fit=CoefficientFitter().fit(first.dataset)
    assert np.allclose(fit.added_mass,TRUE_ADDED,atol=1e-10)
    cfd_vessel=VesselFactory.from_cfd(vessel,tuple(tuple(TRUE_ADDED[i] if i==j else 0 for j in range(6)) for i in range(6)),
        TRUE_LINEAR,TRUE_QUAD,run_id="cfd-9",uncertainty=.02)
    assert cfd_vessel.provenance["linear_damping"].source_kind == "CFD-derived"
    broken=CFDResult(case,first.dataset,False,{})
    with pytest.raises(CFDExecutionError): broken.validated()


def test_coefficient_recovery_noise_identifiability_bounds_and_nonfinite():
    exact=CoefficientFitter().fit(synthetic_dataset(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD,seed=4))
    assert np.allclose(exact.quadratic_damping,TRUE_QUAD,atol=1e-10)
    noisy=CoefficientFitter().fit(synthetic_dataset(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD,seed=4,noise_std=.01))
    assert np.allclose(noisy.linear_damping,TRUE_LINEAR,rtol=.01,atol=.03)
    partial=synthetic_dataset(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD,seed=5,active_axes=(0,2,5))
    partial_fit=CoefficientFitter().fit(partial,axes=(0,2,5))
    assert np.allclose(np.asarray(partial_fit.added_mass)[[0,2,5]],np.asarray(TRUE_ADDED)[[0,2,5]])
    under=synthetic_dataset(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD,seed=1)
    under.velocity[:]=1; under.acceleration[:]=1
    with pytest.raises(CalibrationError,match="not identifiable"): CoefficientFitter().fit(under)
    bad=synthetic_dataset(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD); bad.wrench[0,0]=np.nan
    with pytest.raises(CalibrationError,match="finite"): CoefficientFitter().fit(bad)
    with pytest.raises(CalibrationError,match="bounds"):
        CoefficientFitter(bounds=(0,10)).fit(synthetic_dataset(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD))


def test_synthetic_calibration_improves_separate_held_out_maneuver():
    initial=generated().model_copy(update={
        "added_mass_kg":tuple(tuple(TRUE_ADDED[i]*.55 if i==j else 0 for j in range(6)) for i in range(6)),
        "linear_damping":tuple(x*.55 for x in TRUE_LINEAR),"quadratic_damping":tuple(x*1.5 for x in TRUE_QUAD)})
    families=("straight_acceleration","coast_down","differential_thrust","combined_excitation")
    train=tuple(simulate_maneuver(f,mass=initial.mass_kg,added_mass=TRUE_ADDED,linear=TRUE_LINEAR,quadratic=TRUE_QUAD) for f in families)
    held=simulate_maneuver("turning",mass=initial.mass_kg,added_mass=TRUE_ADDED,linear=TRUE_LINEAR,quadratic=TRUE_QUAD)
    result=calibrate(initial,train,held,run_id="synthetic-cal-1")
    assert result.post_rmse < result.pre_rmse*.05
    assert result.held_out_family not in result.train_families
    assert result.vessel.provenance["linear_damping"].source_kind == "calibration-adjusted"
    assert result.vessel.provenance["linear_damping"].original_value is not None


def test_generated_vessel_loads_normal_simulator_and_artifacts_are_complete(tmp_path):
    vessel=generated(); definitions=vessel.simulator_definitions()
    definitions.append({"kind":"task","id":"waypoint","version":"1","source":"test",
        "payload":{"kind":"waypoint","agent_id":"agent","target_ned_m":[10,0,0],"radius_m":1}})
    config={"schema_version":1,"experiment":{"id":"generated","seed":1},
        "simulation":{"dynamics_mode":"full6","master_dt_s":.02,"dynamics_substeps":1,
            "policy_every_n_master_steps":1,"max_master_steps":1},
        "world":{"source":{"kind":"parametric"},"environment":{"current":{"kind":"uniform","ned_mps":[0,0,0]},
            "wind":{"kind":"uniform","ned_mps":[0,0,0]},"waves":{"kind":"calm"},"visibility_m":1000}},
        "vessels":[{"instance_id":"agent","definition":f"{vessel.id}@{vessel.version}",
            "actuators":["port@1","starboard@1"],"controller":{"mode":"direct_actuator"},
            "spawn":{"ned_m":[0,0,0],"rpy_rad":[0,0,0]}}],
        "task":{"type":"waypoint@1","reward":{"individual_weight":1,"team_weight":0},
            "disabled_agent_behavior":"deactivate_keep_physical"},"logging":{}}
    engine=build_engine(resolve(config,build_registry(definitions))); assert engine.reset().master_step == 0
    result=SyntheticCFDAdapter(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD).execute(
        SyntheticCFDAdapter(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD).generate_case(vessel.geometry.content_hash,deterministic_test_matrix()))
    fit=CoefficientFitter().fit(result.dataset)
    root=write_run_artifact(tmp_path,run_id="phase12",vessel=vessel,cfd_result=result,fit_result=fit,seed=0)
    manifest=json.loads((root/"manifest.json").read_text())
    assert set(manifest["files"]) == {"resolved_vessel.json","cfd.json","fit.json","manifest.json"}
    assert manifest["validation_claim"] == "synthetic/reference validation only"


def test_supported_workflows_always_write_identification_and_calibration_artifacts(tmp_path):
    vessel=generated(); adapter=SyntheticCFDAdapter(TRUE_ADDED,TRUE_LINEAR,TRUE_QUAD,seed=12)
    fitted,_,cfd_root=identify_from_cfd(vessel,adapter,run_id="cfd-workflow",artifact_root=tmp_path,seed=12)
    assert (cfd_root/"cfd.json").exists() and (cfd_root/"fit.json").exists()
    initial=fitted.model_copy(update={"linear_damping":tuple(x*.7 for x in fitted.linear_damping)})
    train=tuple(simulate_maneuver(f,mass=fitted.mass_kg,added_mass=TRUE_ADDED,linear=TRUE_LINEAR,
        quadratic=TRUE_QUAD) for f in ("straight_acceleration","differential_thrust","combined_excitation"))
    held=simulate_maneuver("turning",mass=fitted.mass_kg,added_mass=TRUE_ADDED,linear=TRUE_LINEAR,quadratic=TRUE_QUAD)
    calibrated,cal_root=calibrate_from_logs(initial,train,held,run_id="cal-workflow",artifact_root=tmp_path,seed=12)
    assert calibrated.post_rmse <= calibrated.pre_rmse
    assert (cal_root/"calibration_dataset_manifest.json").exists()
    assert (cal_root/"held_out_validation.json").exists()
