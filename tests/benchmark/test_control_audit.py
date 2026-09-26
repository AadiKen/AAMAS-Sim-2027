import math

import pytest
import torch

from bcod_sim.benchmark.control import integrate_gyro, common_heading_rate
from bcod_sim.benchmark.bcod_adapter import BCODAdapter, _otter_parameters
from bcod_sim.benchmark.core import BenchmarkConfig, NAMES
from bcod_sim.benchmark.qualify import clear_scenario, coordinates
from bcod_sim.frames.transforms import rpy_to_quaternion
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.dynamics.plant6 import EXTERNAL_TERMS


def test_full_imu_orientation_matches_native_kinematics_with_tilt():
    adapter=BCODAdapter()
    adapter.reset(clear_scenario(),123)
    plant=adapter.engine.vessels[NAMES[0]].plant
    q=torch.tensor(rpy_to_quaternion(.3,-.25,.7),dtype=torch.float64)
    omega=torch.tensor([.2,.1,-.3],dtype=torch.float64)
    # Native kinematic integration with imposed constant body angular velocity,
    # independent of hydrostatic accelerations (which are not under test).
    original=plant.acceleration
    plant.acceleration=lambda *a,**k:torch.zeros(6,dtype=torch.float64)
    try:
        nu=torch.cat((torch.zeros(3,dtype=torch.float64),omega))
        state=VesselState(torch.zeros(3,dtype=torch.float64),q,nu)
        estimated=q.clone()
        for _ in range(20):
            state=plant.step(state,{term:torch.zeros(6,dtype=torch.float64) for term in EXTERNAL_TERMS},.02).state
            estimated=integrate_gyro(estimated,omega,omega,.02)
        assert torch.allclose(estimated,state.q_body_to_ned,atol=1e-8)
        heading,rate=common_heading_rate(estimated,omega)
        assert abs(rate+float(omega[2]))>.01
        assert heading==pytest.approx(common_heading_rate(state.q_body_to_ned,omega)[0],abs=1e-8)
    finally:
        plant.acceleration=original
        adapter.close()


@pytest.mark.parametrize('reduced',[False,True])
def test_bcod_controller_keeps_native_limits_and_reports_saturation(reduced):
    adapter=BCODAdapter(BenchmarkConfig(half_width_m=500),reduced_fidelity=reduced)
    try:
        adapter.reset(clear_scenario(),123)
        for _ in range(1):adapter.step({n:(1.,1.) for n in NAMES})
        assert any(adapter.diagnostics['saturated'].values())
        for (env,owner,key),command in adapter.engine.held_commands.items():
            thruster=next(t for t in adapter.engine.vessels[NAMES[owner-1]].actuators if t.config.instance_id==key)
            assert thruster.config.thrust_bounds_n.minimum <= command.thrust_n <= thruster.config.thrust_bounds_n.maximum
        assert _otter_parameters()['mass_kg']==55
    finally:adapter.close()


def test_planar_yaw_crosses_pi_without_stalling():
    adapter=BCODAdapter(BenchmarkConfig(half_width_m=500),reduced_fidelity=True)
    try:
        adapter.reset(clear_scenario(heading=-math.pi/2+.05),123)
        previous=adapter.engine.states[NAMES[0]].q_body_to_ned
        for _ in range(30):
            readings,truth=adapter.step({n:(.8,-.5) for n in NAMES})
        error=readings[NAMES[0]].heading_rad-truth[NAMES[0]].heading_rad
        assert abs(math.atan2(math.sin(error),math.cos(error)))<.01
        assert readings[NAMES[0]].yaw_rps<-.1
    finally:adapter.close()


@pytest.mark.parametrize('reduced',[False,True])
def test_cardinal_diagonal_all_observation_fields(reduced):
    config=BenchmarkConfig(half_width_m=500)
    adapter=BCODAdapter(config,reduced_fidelity=reduced)
    try:
        assert all(row['pass'] for row in coordinates(adapter,config))
    finally:adapter.close()


@pytest.mark.parametrize('kind',['bcod','bcod-reduced','pyquaticus'])
@pytest.mark.parametrize('distance,visible',[(29.9,True),(30.,True),(30.1,False)])
def test_center_visibility_boundary_for_agents_and_obstacles(kind,distance,visible):
    import os
    from bcod_sim.benchmark.runner import make_adapter
    from bcod_sim.benchmark.core import Scenario,VesselPose,Obstacle,observation
    executable=os.environ.get('PYQUATICUS_BENCHMARK_PYTHON')
    if kind=='pyquaticus' and not executable:pytest.skip('isolated interpreter required')
    config=BenchmarkConfig(half_width_m=100)
    adapter=make_adapter(kind,config,pyquaticus_python=executable)
    try:
        s=Scenario('nominal',123,(VesselPose(0,0,math.pi/4),VesselPose(0,distance,0),
                    VesselPose(-60,-60,0),VesselPose(60,60,0)),((20,0),)*4,(Obstacle(distance,0,1),))
        readings,_=adapter.reset(s,123)
        obs=observation(readings[NAMES[0]],s.goals[0],config)
        assert (obs[8]>0)==visible
        assert (obs[17]>0)==visible
        assert all(obs[11:17]==0) and all(obs[20:]==0)
    finally:adapter.close()


def test_full_adapter_heading_estimator_under_native_roll_pitch_motion():
    adapter=BCODAdapter(BenchmarkConfig(half_width_m=500))
    try:
        adapter.reset(clear_scenario(),123)
        name=NAMES[0]
        q=torch.tensor(rpy_to_quaternion(.12,-.08,math.pi/2),dtype=torch.float64)
        state=adapter.engine.states[name]
        adapter.engine.states[name]=VesselState(state.position_ned,q,state.nu_body)
        adapter.orientation[name]=q.clone()  # known initialized attitude
        errors=[]
        for _ in range(60):
            readings,truth=adapter.step({n:(.4,.5) for n in NAMES})
            delta=readings[name].heading_rad-truth[name].heading_rad
            errors.append(abs(math.atan2(math.sin(delta),math.cos(delta))))
        assert max(errors)<.02
    finally:adapter.close()


def test_shared_yaw_ceiling_and_sweep_gate():
    from bcod_sim.benchmark.qualify import control_checks
    config = BenchmarkConfig()
    assert config.max_yaw_rps == .25
    assert [y*config.max_yaw_rps for y in (-1.,-.5,0.,.5,1.)] == [-.25,-.125,0.,.125,.25]
    rows = [{'action':[u,y], 'metrics':{'yaw_rps':{'final_mean':y*.25}},
             'final_saturation_fraction':0.} for u in (.5,.8,1.) for y in (-1.,-.5,0.,.5,1.)]
    checks = control_checks(rows)
    assert checks['yaw_sign_pass'] and checks['yaw_monotonic_pass'] and checks['normal_saturation_pass']
    assert max(checks['yaw_symmetry_error']) == 0
    rows[9]['final_saturation_fraction'] = 1.
    assert not control_checks(rows)['normal_saturation_pass']
    rows[9]['metrics']['yaw_rps']['final_mean'] = -.25
    assert not control_checks(rows)['yaw_sign_pass']
    assert not control_checks(rows)['yaw_monotonic_pass']
