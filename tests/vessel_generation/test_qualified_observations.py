import json
from pathlib import Path
import numpy as np
import pytest

from bcod_sim.vessel_generation.cfd import CFDExecutionError
from bcod_sim.vessel_generation.qualified_observations import (
    _reference_compatible, _window, align_motion, verify_artifact)
from bcod_sim.vessel_generation.workflow import read_observations, AcceptedObservation, identify_package
from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.frame_contract import foam_wrench_to_body


def _motion(kind="forced_translation", dof="surge"):
    return {"motion_type": kind, "dof": dof, "frequency_rad_s": 2*np.pi,
            "amplitude": .2, "magnitude": .3}


def test_sinusoidal_alignment_uses_force_times():
    times=np.array([0., .125, .25, .375, .5])
    q,v,a=align_motion(times,_motion())
    assert q[2,0] == pytest.approx(.2)
    assert v[0,0] == pytest.approx(.4*np.pi)
    assert a[2,0] == pytest.approx(-.2*(2*np.pi)**2)
    assert np.all(q[:,1:] == 0)


def test_periodic_window_excludes_startup_and_rejects_evolving_cycles():
    t=np.linspace(0,5,501); wrench=np.zeros((len(t),6))
    wrench[:,0]=3*np.sin(2*np.pi*t)
    selected,quality=_window(np.column_stack((t,wrench)),_motion())
    assert quality["method"] == "three_complete_cycles_after_startup"
    assert selected[0,0] >= 2
    wrench[:,0]=(1+1.5*t)*np.sin(2*np.pi*t)
    with pytest.raises(CFDExecutionError,match="repeatable"):
        _window(np.column_stack((t,wrench)),_motion())


def test_steady_active_channel_ignores_near_zero_cross_axis_noise():
    t=np.linspace(0,1,101); wrench=np.zeros((len(t),6))
    wrench[:,0]=5.;wrench[:,1]=.01*np.sin(40*t)
    selected,quality=_window(np.column_stack((t,wrench)),_motion("steady_velocity"))
    assert len(selected)>8 and quality["active_channel"] == 0


def test_steady_rotation_uses_orientation_cycles():
    t=np.linspace(0,4,401);wrench=np.zeros((len(t),6))
    wrench[:,5]=2+np.sin(2*np.pi*t)
    selected,quality=_window(np.column_stack((t,wrench)),{"motion_type":"steady_rotation","dof":"yaw","magnitude":2*np.pi})
    assert quality["method"] == "two_orientation_revolutions_after_startup"
    assert selected[0,0]>=2
    wrench[:,5]=(2+.4*t)+np.sin(2*np.pi*t)
    with pytest.raises(CFDExecutionError,match="orientation-periodic"):
        _window(np.column_stack((t,wrench)),{"motion_type":"steady_rotation","dof":"yaw","magnitude":2*np.pi})


def test_reference_requires_matching_waterline_and_fluid():
    common={key: 0 for key in ("geometry_sha256", "fluid_model", "water_properties", "mesh_settings",
            "domain_settings", "cg_body_frd_m", "moment_reference_point_frd_m",
            "solver_moment_reference_point_m", "source_waterline_z_m", "openfoam_version")}
    reference={**common,"motion_type":"steady_velocity","magnitude":1e-9}
    _reference_compatible(common,reference)
    with pytest.raises(CFDExecutionError,match="static reference"):
        _reference_compatible({**common,"source_waterline_z_m":.1},reference)


def test_arbitrary_json_is_rejected_in_production(tmp_path):
    path=tmp_path/"observations.json"
    path.write_text(json.dumps([{"case_id":"claimed","velocity":[0]*6,"acceleration":[0]*6,
        "wrench":[0]*6,"accepted":True,"frame_contract":"bcod-openfoam-frd-v1",
        "fitting_window_s":[0,1]}]))
    with pytest.raises(CFDExecutionError):
        read_observations(path)
    assert len(read_observations(path,unsafe_debug=True)) == 1


def test_direct_fitter_rejects_unverified_observation(tmp_path):
    observation=AcceptedObservation("manual",(0,)*6,(0,)*6,(0,)*6,True,
        "bcod-openfoam-frd-v1",(0,1))
    with pytest.raises(ValueError,match="verified"):
        identify_package(name="manual",geometry_path=tmp_path/"missing.stl",mass_kg=1,
            cg_frd_m=(0,0,0),observations=(observation,),output=tmp_path/"out",solver={})


def test_artifact_version_fails_closed(tmp_path):
    path=tmp_path/"artifact.json";path.write_text(json.dumps({"schema":"invented","quality_status":"CFD_QUALIFIED"}))
    with pytest.raises(CFDExecutionError,match="unsupported"):
        verify_artifact(path)


def test_real_qualified_artifact_is_recomputed_and_tampering_rejected(tmp_path):
    artifact=Path(__file__).resolve().parents[2]/"stage3_results/stage3a-qualification/observations/forced_surge.json"
    accepted=verify_artifact(artifact)
    assert accepted["reference_case_id"]
    assert len(read_observations(artifact)) == len(accepted["observations"])
    changed=json.loads(artifact.read_text());changed["observations"][0]["resisting_wrench"][0]+=1
    tampered=tmp_path/"tampered.json";tampered.write_text(json.dumps(changed))
    with pytest.raises(CFDExecutionError,match="differs"):
        verify_artifact(tampered)
    changed=json.loads(artifact.read_text());changed["case_sources"]["solver.log"]="0"*64
    tampered.write_text(json.dumps(changed))
    with pytest.raises(CFDExecutionError,match="source hash"):
        verify_artifact(tampered)


def test_static_reference_is_subtracted_before_resisting_transform():
    artifact=Path(__file__).resolve().parents[2]/"stage3_results/stage3a-qualification/observations/steady_surge.json"
    value=verify_artifact(artifact)
    row=value["observations"][0]
    history=np.asarray(OpenFOAMAdapter.force_history(Path(value["case_root"])))
    raw=history[np.argmin(abs(history[:,0]-row["time_s"])),1:]
    corrected=raw-np.asarray(value["static_reference_wrench_foam"])
    expected=sum(foam_wrench_to_body(corrected[:3],corrected[3:],
        foam_reference=value["moment_reference_point_frd_m"],
        body_reference=value["moment_reference_point_frd_m"],
        waterline_z_m=value["waterline_z_m"]),())
    assert row["resisting_wrench"] == pytest.approx(expected)
