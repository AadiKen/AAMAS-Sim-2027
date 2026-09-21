import math
import numpy as np
import pytest
import torch
from bcod_sim.actuators.base import ActuatorConfig,ActuatorState,Bounds
from bcod_sim.actuators.propulsion import LinearThrustMap,PiecewiseLinearThrustMap,PolynomialThrustMap,RPMThruster,RpmCommand
from bcod_sim.config.hashing import content_hash
from bcod_sim.config.models import IrregularWaves,RegularWaves
from bcod_sim.config.vessel_compiler import compile_vessel
from bcod_sim.core.errors import CommandBoundsError,OperatingEnvelopeError,PhysicalValidationError
from bcod_sim.dynamics.damping import CoupledDampingTerm,Damping
from bcod_sim.dynamics.environmental import RelativeWindLoads,WindCoefficientPoint,KinematicWaveLoads
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.mesh_buoyancy import MeshBuoyancy,TriangleMesh,linearize_mesh_hydrostatics
from bcod_sim.dynamics.reference_points import parallel_axis_inertia,transform_added_mass,translate_wrench,transform_point
from bcod_sim.dynamics.validity import ModelValidityEnvelope,ValidityBound,ValidityMonitor
from bcod_sim.frames.transforms import rpy_to_quaternion
from bcod_sim.scenario.uncertainty import Uncertainty,resolve_uncertain
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.vessel_generation.campaign_design import design_cfd_campaign
from bcod_sim.vessel_generation.payload import Payload,install_payload
from bcod_sim.world.waves import WaveField
from bcod_sim.world.world import WorldSample

D=torch.float64
def t(x): return torch.tensor(x,dtype=D)
def state(nu=(0,0,0,0,0,0),position=(0,0,0),rpy=(0,0,0)): return VesselState(t(position),t(rpy_to_quaternion(*rpy)),t(nu))

def box_mesh():
 v=np.array([[-1,-.5,-.5],[1,-.5,-.5],[1,.5,-.5],[-1,.5,-.5],[-1,-.5,.5],[1,-.5,.5],[1,.5,.5],[-1,.5,.5]])
 f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
 return TriangleMesh.from_arrays(v,f)

def sample(*,wind=(0,0,0),orbital=(0,0,0),acceleration=(0,0,0)):
 z=t([0.]); z3=t([[0.,0,0]])
 return WorldSample(z3,t([wind]),z3,z,t([orbital]),t([acceleration]),t([1025]),t([1.225]),t([1000]),z,z,None,None)

def test_reference_transforms_roundtrip_parallel_axis_force_and_added_mass():
 p=t((1,2,3)); r=t((.2,-.1,.3)); assert transform_point(transform_point(p,r),-r).tolist()==pytest.approx(p.tolist())
 inertia=torch.diag(t((2,3,4))); shifted=parallel_axis_inertia(inertia,5,r)
 assert torch.allclose(shifted,shifted.T) and torch.linalg.eigvalsh(shifted)[0]>0
 tau=translate_wrench(t((10,0,0,0,0,0)),t((0,2,0))); assert tau.tolist()==pytest.approx([10,0,0,0,0,-20])
 ma=torch.diag(t((1,2,3,4,5,6))); transformed=transform_added_mass(ma,r); assert torch.allclose(transformed,transformed.T)

def test_full_and_coupled_damping_relative_flow_and_dissipation():
 matrix=torch.diag(t((2,3,4,5,6,7))); matrix[1,5]=matrix[5,1]=.2
 model=Damping(torch.zeros(6,dtype=D),torch.ones(6,dtype=D)*.1,matrix,(CoupledDampingTerm(1,(1,),.3,(1,)),))
 model.validate(dtype=D,device=torch.device('cpu')); nu=t((1,.2,0,0,0,.1)); result=model.evaluate(nu)
 assert float(nu@result.tau_body)<=0 and result.diagnostics['tau_linear_damping'].shape==(6,)

def test_thrust_maps_and_rpm_dynamics_advance_once():
 assert LinearThrustMap(2).thrust(3)==6
 table=PiecewiseLinearThrustMap((-1,0,1),(-4,0,5)); assert table.thrust(.5)==pytest.approx(2.5)
 with pytest.raises(CommandBoundsError): table.thrust(2)
 assert PolynomialThrustMap((0,1,1),(-2,2)).thrust(2)==6
 config=ActuatorConfig('rpm',0,1,(0,.5,0),(1,0,0,0),Bounds(-100,100))
 actuator=RPMThruster(config,LinearThrustMap(1),rpm_bounds=(-100,100),rpm_time_constant_s=1)
 first=actuator.step(RpmCommand(10),ActuatorState(),.1); second=actuator.step(RpmCommand(10),first.state,.1)
 assert 0<first.state.rpm<second.state.rpm<10

def test_relative_wind_and_wave_loading_invariants():
 points=(WindCoefficientPoint(-math.pi,-1,0,0,0),WindCoefficientPoint(-math.pi/2,0,-1,-.2,-.3),WindCoefficientPoint(0,1,0,0,0),WindCoefficientPoint(math.pi/2,0,1,.2,.3),WindCoefficientPoint(math.pi,-1,0,0,0))
 wind=RelativeWindLoads(2,3,1,points)
 zero=wind.evaluate(state(),sample()).tau_body; head=wind.evaluate(state(),sample(wind=(4,0,0))).tau_body; head2=wind.evaluate(state(),sample(wind=(8,0,0))).tau_body
 assert zero.tolist()==pytest.approx([0]*6) and head2[0].item()==pytest.approx(4*head[0].item())
 port=wind.evaluate(state(),sample(wind=(0,4,0))).tau_body; starboard=wind.evaluate(state(),sample(wind=(0,-4,0))).tau_body
 assert port[1].item()==pytest.approx(-starboard[1].item()) and port[5].item()==pytest.approx(-starboard[5].item())
 wave=KinematicWaveLoads((1,1,1),(1,1,1),(1,1,1)); assert wave.evaluate(state(),sample()).tau_body.tolist()==pytest.approx([0]*6)

def test_regular_and_irregular_wave_determinism_phase_and_statistics():
 positions=t([[0,0,0],[1,0,0]])
 regular=WaveField(RegularWaves(kind='regular',height_m=2,period_s=4,direction_rad=0,phase_rad=0))
 surface,velocity,accel=regular.sample_kinematics(positions,0); assert surface[0].item()==pytest.approx(-1) and velocity.shape==accel.shape==(2,3)
 spec=IrregularWaves(kind='irregular',spectrum='jonswap',significant_height_m=1.5,peak_period_s=5,direction_rad=.2,component_count=64,seed=9)
 a=WaveField(spec); b=WaveField(spec)
 times=np.linspace(0,300,1200); series=np.array([a.sample_kinematics(positions[:1],float(x))[0].item() for x in times])
 assert np.array_equal(series,np.array([b.sample_kinematics(positions[:1],float(x))[0].item() for x in times]))
 assert 3.5*np.std(series)==pytest.approx(1.5,rel=.25)

def test_validity_payload_uncertainty_compiler_and_cfd_design():
 monitor=ValidityMonitor(ModelValidityEnvelope('demo',{'speed':ValidityBound(0,2,'manual')},False)); monitor.observe({'speed':3},1.2)
 assert not monitor.validated and monitor.first_exceedance['speed']==1.2
 strict=ValidityMonitor(ModelValidityEnvelope('demo',{'speed':ValidityBound(0,2,'manual')},True))
 with pytest.raises(OperatingEnvelopeError): strict.observe({'speed':3},0)
 base=MassProperties(10,t((0,0,0)),torch.diag(t((2,3,4))),torch.zeros((6,6),dtype=D))
 centered,_=install_payload(base,Payload(2,(0,0,0),((1,0,0),(0,1,0),(0,0,1))))
 off,data=install_payload(base,Payload(2,(0,1,0),((1,0,0),(0,1,0),(0,0,1))))
 assert centered.cg_frd_m.tolist()==pytest.approx([0,0,0]) and off.cg_frd_m[1]>0 and data['fingerprint']
 u=Uncertainty('normal',{'sigma':1}); a,pa=resolve_uncertain(10,u,master_seed=4,parameter_path='mass'); b,pb=resolve_uncertain(10,u,master_seed=4,parameter_path='mass')
 assert a==b and pa['substream_seed']==pb['substream_seed']
 design=design_cfd_campaign(['Y_v','Y_vv','N_r','N_rr','Y_r','N_v']); assert design.identifiable and design.design_rank==6
 vessel={k:{} for k in ('damping','hydrostatics','crossflow','equilibrium','validity','provenance')}; vessel.update({'canonical_origin_frd_m':[0,0,0],'canonical_frame':'FRD','mass':10,'cg':[0,0,0],'inertia':[], 'added_mass':[],'coriolis_model':'derived','actuators':[],'sensors':[],'asset_hashes':{'mesh':'a'*64}}); vessel['hydrostatics']={'model':'linear_matrix'}
 compiled=compile_vessel(vessel); changed={**vessel,'damping':{'linear':1}}
 assert compiled.plant_fingerprint!=compile_vessel(changed).plant_fingerprint

def test_mesh_local_linearization_provenance():
 model=MeshBuoyancy(box_mesh(),1000,9.81,0,(-2,2),.5,.5)
 result=linearize_mesh_hydrostatics(model,1000,t((0,0,0)),(0,0,0),(0,0,0))
 assert np.asarray(result['stiffness_6x6']).shape==(6,6) and result['geometry_hash']==model.mesh.content_hash
