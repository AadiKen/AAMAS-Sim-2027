import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.fitting import CoefficientFitter, ForceMomentDataset, Plant6ModelForm
from bcod_sim.vessel_generation.identification import (CampaignRunner, CaseCache, DOF, FluidModel, IdentificationCase,
    LocalDispatcher, MotionType, SolverSettings, Sweep, TurbulenceModel, TurbulenceSettings)
from bcod_sim.vessel_generation.quality import (convergence_check, periodic_stationarity,
    numerical_history_gate, validate_run, wrench_stationarity)
from bcod_sim.vessel_generation.rotating_ncc import yaw_hull_point_error, validate_rotation_mesh_quality


HASH="a"*64


def test_signed_sweep_covers_all_translation_directions():
    cases=Sweep(MotionType.STEADY_VELOCITY,(DOF.SURGE,DOF.SWAY,DOF.HEAVE),(-1.,-.5,.5,1.)).cases(HASH)
    assert len(cases)==12
    assert {case.dof for case in cases}=={DOF.SURGE,DOF.SWAY,DOF.HEAVE}
    assert all(len(case.velocity_vector())==6 for case in cases)


def test_rotation_and_forced_motion_are_type_checked():
    yaw=IdentificationCase(HASH,MotionType.STEADY_ROTATION,DOF.YAW,magnitude=-.3)
    forced=IdentificationCase(HASH,MotionType.FORCED_TRANSLATION,DOF.HEAVE,frequency_rad_s=2.,amplitude=.1)
    assert yaw.velocity_vector()[5]==-.3 and forced.case_id==forced.case_id
    with pytest.raises(ValueError): IdentificationCase(HASH,MotionType.STEADY_ROTATION,DOF.SURGE,magnitude=1.)


def test_case_hash_invalidates_every_material_input():
    base=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.)
    assert base.case_id!=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=-1.).case_id
    assert base.case_id!=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.,openfoam_version="12").case_id


def test_cache_reuses_only_accepted_results(tmp_path):
    case=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SWAY,magnitude=1.)
    cache=CaseCache(tmp_path);assert cache.load(case) is None
    with pytest.raises(ValueError): cache.store(case,{"accepted":False})
    cache.store(case,{"accepted":True,"wrench":[1,2,3,4,5,6]});assert cache.load(case)["accepted"]


def test_full_wrench_coupling_and_added_mass_fit():
    rng=np.random.default_rng(4);n=240;velocity=rng.uniform(-1,1,(n,6));acceleration=rng.uniform(-1,1,(n,6))
    added=np.diag(np.arange(1.,7.));added[0,1]=added[1,0]=.2
    linear=np.diag(np.arange(2.,8.));linear[5,1]=.3
    quadratic=np.arange(.1,.7,.1)
    wrench=acceleration@added.T+velocity@linear.T+np.abs(velocity)*velocity*quadratic
    fit=CoefficientFitter(max_condition=1e5,bounds=(-np.inf,np.inf)).fit_plant6(ForceMomentDataset(velocity,acceleration,wrench))
    assert np.allclose(fit.added_mass,added,atol=1e-10)
    assert np.allclose(fit.linear_damping,linear,atol=1e-10)
    assert np.allclose(fit.quadratic_damping,quadratic,atol=1e-10)
    assert Plant6ModelForm.discover().linear_damping_matrix


def test_forced_oscillation_recovers_added_mass_and_damping():
    t=np.linspace(0,12*np.pi,300);velocity=np.zeros((len(t),6));accel=np.zeros_like(velocity)
    velocity[:,2]=.2*np.cos(t);accel[:,2]=-.2*np.sin(t)
    rng=np.random.default_rng(7)
    # Excite other axes independently so the full supported Plant6 form remains identifiable.
    velocity[:,:]=rng.uniform(-1,1,velocity.shape);accel[:,:]=rng.uniform(-1,1,accel.shape)
    added=np.eye(6)*4;linear=np.eye(6)*3;quad=np.ones(6)*.5
    wrench=accel@added.T+velocity@linear.T+np.abs(velocity)*velocity*quad
    fit=CoefficientFitter(bounds=(-np.inf,np.inf)).fit_plant6(ForceMomentDataset(velocity,accel,wrench))
    assert fit.added_mass[2][2]==pytest.approx(4) and fit.linear_damping[2][2]==pytest.approx(3)


def test_quality_gates_reject_drift_and_convergence_comparison():
    # A 1 N drift against a 1000 N physical scale is deliberately near-zero;
    # exercise material loads here so the drift must remain rejectable.
    history=np.ones((20,6))*100;residual=np.geomspace(1,.000001,20)
    settings={"time_s":np.arange(20.),"force_scale_n":1000.,"moment_scale_nm":5000.,"window_s":8.}
    assert validate_run(history,residual,**settings).accepted
    history[-4:]*=2;assert not validate_run(history,residual,**settings).accepted
    assert convergence_check(np.ones(6)*1.2,np.ones(6)*1.01,np.ones(6))["accepted"]


def test_generic_openfoam_case_records_full_wrench_and_free_surface(tmp_path):
    geometry=tmp_path/"hull.stl"
    vertices=((-0.1,-0.1,-0.1),(0.1,-0.1,-0.1),(0,0.1,-0.1),(0,0,0.1))
    triangles=((0,1,2),(0,3,1),(1,3,2),(2,3,0))
    geometry.write_text("solid h\n"+"".join("facet normal 0 0 0\nouter loop\n"+
        "".join(f"vertex {x} {y} {z}\n" for x,y,z in (vertices[i] for i in face))+
        "endloop\nendfacet\n" for face in triangles)+"endsolid h\n")
    adapter=OpenFOAMAdapter(tmp_path/"cases")
    case=IdentificationCase(HASH,MotionType.STEADY_ROTATION,DOF.YAW,magnitude=-.2,fluid_model=FluidModel.FREE_SURFACE)
    root=adapter.generate_identification_case(SimpleNamespace(path=geometry,content_hash=HASH),case)
    metadata=json.loads((root/"case_metadata.json").read_text())
    assert metadata["measured_channels"]==["Fx","Fy","Fz","Mx","My","Mz"]
    assert metadata["fluid_model"]=="free_surface" and (root/"0/alpha.water").exists()
    assert not (root/"0/p").exists()
    control=(root/"system/controlDict").read_text()
    assert "adjustTimeStep yes" in control and "maxCo 0.5" in control
    assert "maxAlphaCo 0.5" in control and "maxDeltaT 0.01" in control
    block=(root/"system/blockMeshDict").read_text()
    assert all(name in block for name in ("inletWater","inletAir","outletWater","outletAir","atmosphere","bottom"))
    alpha=(root/"0/alpha.water").read_text()
    assert "inletWater {type fixedValue; value uniform 1;}" in alpha
    assert "inletAir {type fixedValue; value uniform 0;}" in alpha
    assert metadata["rotation_topology"]["method"]=="OpenFOAM11-solidBody-NCC"
    assert "motionSolver solidBody" in (root/"constant/dynamicMeshDict").read_text()
    assert not (root/"0/pointDisplacement").exists()
    topology=metadata["rotation_topology"]
    assert topology["axis_foam"]==[0,0,-1]
    assert topology["center_foam_m"]==[0,0,0]
    assert topology["cell_zone"]==topology["face_zone"]=="rotating"
    assert topology["ncc_patches"]==["nonConformalCyclic_on_nonCouple1","nonConformalCyclic_on_nonCouple2"]
    assert "cellZone rotating" in (root/"system/snappyHexMeshDict").read_text()
    assert "nonCouple1" in (root/"system/createBafflesDict").read_text()


def test_steady_yaw_rotation_sign_center_and_mesh_quality_gate():
    initial=np.array([[2.,3.,4.],[3.,3.,4.],[2.,4.,4.]])
    moved=np.array([[2.,3.,4.],[2.,2.,4.],[3.,3.,4.]])
    assert yaw_hull_point_error(initial,moved,np.array([0,1,2]),
        center=(2.,3.,4.),angle_rad=np.pi/2)<1e-14
    assert yaw_hull_point_error(initial,moved,np.array([0,1,2]),
        center=(0.,0.,0.),angle_rad=np.pi/2)>1
    log=("Cell volumes OK. Min volume = 0.001.\n"
         "Mesh non-orthogonality Max: 32 average: 5\nNon-orthogonality check OK.\n"
         "Max skewness = 1.9 OK.\nError in face tets: 0\n")
    assert validate_rotation_mesh_quality(log,log)["current"]["min_volume"]==.001
    with pytest.raises(ValueError,match="degraded"):
        validate_rotation_mesh_quality(log,log.replace("Min volume = 0.001", "Min volume = 0.0005"))
    with pytest.raises(ValueError,match="invalid"):
        validate_rotation_mesh_quality(log,log+"negative cell volume")


def test_surface_identification_defaults_to_explicit_sst_open_water(tmp_path):
    geometry=tmp_path/"hull.stl"; geometry.write_text("solid h\nendsolid h\n")
    adapter=OpenFOAMAdapter(tmp_path/"cases")
    case=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.,fluid_model=FluidModel.FREE_SURFACE)
    root=adapter.generate_identification_case(SimpleNamespace(path=geometry,content_hash=HASH),case)
    metadata=json.loads((root/"case_metadata.json").read_text())
    assert metadata["turbulence_model"]=="kOmegaSST"
    assert metadata["domain_settings"]["minimum_frd_m"]==[-12.,-10.,-5.]
    assert metadata["mesh_settings"]["boundary_layers"]>=3
    assert "model kOmegaSST" in (root/"constant/momentumTransport").read_text()
    assert all((root/f"0/{name}").exists() for name in ("k","omega","nut"))
    assert "addLayers true" in (root/"system/snappyHexMeshDict").read_text()
    assert "type yPlus" in (root/"system/controlDict").read_text()
    laminar=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.,
        fluid_model=FluidModel.FREE_SURFACE,turbulence_settings=TurbulenceSettings(model=TurbulenceModel.LAMINAR))
    assert laminar.case_id!=case.case_id


def test_source_waterline_is_aligned_without_changing_hull_shape(tmp_path):
    geometry=tmp_path/"hull.stl"
    geometry.write_text("solid h\n facet normal 0 0 1\n outer loop\n"
        " vertex 0 0 0.1\n vertex 1 0 0.2\n vertex 0 1 0.3\n"
        " endloop\n endfacet\nendsolid h\n")
    case=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.,
        fluid_model=FluidModel.FREE_SURFACE,waterline_z_m=.2)
    root=OpenFOAMAdapter(tmp_path/"cases").generate_identification_case(
        SimpleNamespace(path=geometry,content_hash=HASH),case)
    aligned=(root/"constant/triSurface/hull.stl").read_text()
    assert "vertex 0 0 -0.1" in aligned
    assert "vertex 1 0 0" in aligned
    assert "CofR (0.0 0.0 -0.2)" in (root/"system/controlDict").read_text()
    metadata=json.loads((root/"case_metadata.json").read_text())
    assert metadata["source_waterline_z_m"]==.2
    assert metadata["geometry_translation_z_m"]==-.2


def test_timestep_controls_are_case_hashed_and_configurable():
    a=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.,
        solver_settings=SolverSettings(max_courant=.4,max_alpha_courant=.2,timestep_s=.025))
    b=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.)
    assert a.case_id!=b.case_id


def test_parallel_completion_marker_survives_finalising_log_line(tmp_path):
    segments=tmp_path/"solver_segments";segments.mkdir()
    (segments/"from_0_1.log").write_text(
        "Time = 0.1s\nCourant Number mean: 0.1 max: 0.2\n"
        "deltaT = 0.1\nEnd\nFinalising parallel run\n")
    assert OpenFOAMAdapter.diagnose_case(tmp_path)["solver_completed"]


def test_campaign_runner_parallel_boundary_and_cache(tmp_path):
    class Backend:
        calls=0
        def execute_identification_case(self,case):
            self.calls+=1;return {"accepted":True,"case_id":case.case_id}
    backend=Backend();runner=CampaignRunner(backend,CaseCache(tmp_path),LocalDispatcher(1))
    case=IdentificationCase(HASH,MotionType.STEADY_VELOCITY,DOF.SURGE,magnitude=1.)
    assert runner.run((case,))[case.case_id]["accepted"]
    assert runner.run((case,))[case.case_id]["accepted"] and backend.calls==1


def test_scale_aware_stationarity_significant_and_near_zero_channels():
    t=np.linspace(0,20,1000);rng=np.random.default_rng(2)
    stable=np.column_stack([100+rng.normal(0,1,len(t)) for _ in range(6)])
    assert wrench_stationarity(t,stable,force_scale_n=1000,moment_scale_nm=5000,window_s=10).accepted
    drifting=stable.copy();drifting[:,0]+=4*t
    assert not wrench_stationarity(t,drifting,force_scale_n=1000,moment_scale_nm=5000,window_s=10).accepted
    zero=np.column_stack([rng.normal(0,.5,len(t)) for _ in range(6)])
    result=wrench_stationarity(t,zero,force_scale_n=1000,moment_scale_nm=5000,window_s=10)
    assert result.accepted and result.channels[0].regime=="near_zero"
    zero_drift=zero.copy();zero_drift[:,1]+=t
    assert not wrench_stationarity(t,zero_drift,force_scale_n=1000,moment_scale_nm=5000,window_s=10).accepted


def test_diagnose_case_joins_force_history_across_restarts(tmp_path):
    (tmp_path/"solver.log").write_text(
        "Time = 1s\nGAMG: Solving for p_rgh, Initial residual = 0.01, Final residual = 0.00004\n"
        "GAMG: Solving for p_rgh, Initial residual = 0.001, Final residual = 0.000001\n"
        "Time = 2s\nGAMG: Solving for p_rgh, Initial residual = 0.01, Final residual = 0.000002\nEnd\n")
    for restart_time, samples in ((0, ((1, 1), (2, 2))), (2, ((2, 20), (3, 3)))):
        directory=tmp_path/"postProcessing"/"forces"/str(restart_time)
        directory.mkdir(parents=True)
        body="".join(f"{time} ({force} 0 0) (0 0 0) (0 0 0) (0 0 0)\n" for time,force in samples)
        (directory/"forces.dat").write_text(body)
    diagnostics=OpenFOAMAdapter.diagnose_case(tmp_path)
    history=diagnostics["force_moment_history"]
    assert [sample[0] for sample in history]==[1,2,3]
    assert [sample[1] for sample in history]==[1,20,3]
    assert diagnostics["step_final_residuals"]["p_rgh"]==[1e-6,2e-6]


def test_periodic_stationarity_rejects_mean_and_amplitude_transients():
    t=np.linspace(0,20,4001);phase=2*np.pi*.8*t
    assert periodic_stationarity(t,np.ones_like(t)*20,physical_scale=1000) is None
    stable=periodic_stationarity(t,20+4*np.sin(phase),physical_scale=1000)
    assert stable is not None and stable.accepted and stable.complete_cycles>=5
    assert stable.frequency_hz==pytest.approx(.8,abs=.02)
    drifting_mean=periodic_stationarity(t,20+.3*t+4*np.sin(phase),physical_scale=1000)
    assert drifting_mean is not None and not drifting_mean.accepted
    growing=periodic_stationarity(t,20+(3+.2*t)*np.sin(phase),physical_scale=1000)
    assert growing is not None and not growing.accepted
    noise=np.random.default_rng(3).normal(0,.4,len(t))
    assert periodic_stationarity(t,noise,physical_scale=1000) is None
    near_zero=np.zeros((len(t),6));near_zero[:,1]=noise
    report=wrench_stationarity(t,near_zero,force_scale_n=1000,moment_scale_nm=5000,window_s=10)
    assert report.accepted and report.channels[1].regime=="near_zero"


def test_multitone_periodic_gate_uses_five_cycle_block_statistics():
    t=np.linspace(0,30,6001);phase=2*np.pi*.8*t
    multi=20+4*np.sin(phase)+1.5*np.sin(2*np.pi*.2*t)
    stable=periodic_stationarity(t,multi,physical_scale=1000)
    assert stable is not None and stable.complete_cycles>=15
    assert stable.block_accepted and stable.accepted
    drift=periodic_stationarity(t,multi+.15*t,physical_scale=1000)
    assert drift is not None and not drift.accepted
    growth=periodic_stationarity(t,20+(4+.12*t)*np.sin(phase)+1.5*np.sin(2*np.pi*.2*t),physical_scale=1000)
    assert growth is not None and not growth.accepted


def test_solver_continuation_streams_durable_joined_logs(tmp_path,monkeypatch):
    from io import StringIO
    from bcod_sim.vessel_generation import cfd
    (tmp_path/"system").mkdir();(tmp_path/"30").mkdir()
    (tmp_path/"system/controlDict").write_text("startFrom latestTime; endTime 36.5;")
    (tmp_path/"case_metadata.json").write_text(json.dumps({"solver":"foamRun -solver incompressibleVoF"}))
    (tmp_path/"solver.log").write_text("Time = 30s\nEnd\n")
    class FakeProcess:
        stdout=StringIO("Time = 36.5s\nCourant Number mean: 0.01 max: 0.4\nEnd\n")
        def wait(self): return 0
    monkeypatch.setattr(cfd.subprocess,"Popen",lambda *args,**kwargs: FakeProcess())
    adapter=OpenFOAMAdapter(tmp_path)
    monkeypatch.setattr(adapter,"verify_runtime",lambda: {"version":"OpenFOAM 11","image_identity":"test-image"})
    result=adapter.solve_case(tmp_path,resume=True)
    assert result["converged"]
    joined=(tmp_path/"solver.log").read_text()
    assert "Time = 30s" in joined and "Time = 36.5s" in joined
    assert "Courant Number" in Path(result["segment_log"]).read_text()


def test_numerical_history_gate_requires_complete_auditable_coverage():
    diagnostic={"time":[.001,1.,2.],"courant_max":[.4,.5],
                "alpha_courant_max":[.1,.2],"timestep":[.001,.002],
                "water_volume_fraction":[.5,.5001,.4999],
                "residuals":{"p_rgh":[(.01,1e-6)]},"solver_completed":True}
    limits={"max_courant":.5,"max_alpha_courant":.5,"max_timestep_s":.005}
    assert numerical_history_gate(diagnostic,expected_start_s=0,expected_end_s=2,**limits)["accepted"]
    assert not numerical_history_gate(diagnostic,expected_start_s=0,expected_end_s=30,**limits)["gates"]["coverage"]
    diagnostic["courant_max"]=[.4,.6]
    assert not numerical_history_gate(diagnostic,expected_start_s=0,expected_end_s=2,**limits)["gates"]["max_courant"]
    diagnostic["courant_max"]=[.4,.5]
    diagnostic["residuals"]={"p_rgh":[(.01,4e-5)]+[(.01,1e-6)]*200}
    assert numerical_history_gate(diagnostic,expected_start_s=0,expected_end_s=2,**limits)["gates"]["residual"]
    diagnostic["residuals"]["p_rgh"][-1]=(.01,4e-5)
    assert not numerical_history_gate(diagnostic,expected_start_s=0,expected_end_s=2,**limits)["gates"]["residual"]


def test_solver_diagnosis_discards_overlapped_restart_interval(tmp_path):
    segments=tmp_path/"solver_segments";segments.mkdir()
    (segments/"from_0_100.log").write_text("Time = 1s\ndeltaT = 1\nTime = 2s\ndeltaT = 1\nTime = 3s\ndeltaT = 9\nEnd\n")
    (segments/"from_2_200.log").write_text("Time = 2s\ndeltaT = 0.5\nTime = 2.5s\ndeltaT = 0.5\nTime = 3s\ndeltaT = 0.5\nEnd\n")
    (tmp_path/"solver.log").write_text("not canonical")
    diagnosis=OpenFOAMAdapter.diagnose_case(tmp_path)
    assert diagnosis["time"]==[1,2,2.5,3]
    assert diagnosis["timestep"]==[1,.5,.5,.5]
    assert diagnosis["solver_completed"]
