import math
import numpy as np
import pytest
import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.config.hashing import content_hash
from bcod_sim.dynamics.crossflow import StripTheoryCrossflow
from bcod_sim.dynamics.mesh_buoyancy import MeshBuoyancy, TriangleMesh, solve_equilibrium_waterline
from bcod_sim.dynamics.restoring import LinearHydrostatics, reference_point_transform
from bcod_sim.frames.transforms import rpy_to_quaternion
from bcod_sim.state.vessel_state import VesselState

DTYPE=torch.float64
def t(x): return torch.tensor(x,dtype=DTYPE)
def state(position=(0,0,0),rpy=(0,0,0),nu=(0,0,0,0,0,0)):
    return VesselState(t(position),t(rpy_to_quaternion(*rpy)),t(nu))

def box_mesh(length=2.,beam=1.,height=1.):
    x,y,z=length/2,beam/2,height/2
    vertices=np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return TriangleMesh.from_arrays(vertices,faces)

def linear(stiffness,equilibrium=(0,0,0)):
    return LinearHydrostatics(t(stiffness),t(equilibrium),t((0,0,0)),.5,.5)

def test_linear_zero_diagonal_coupling_sign_and_equilibrium():
    diagonal=np.diag([0,0,10,2,3,0]); model=linear(diagonal,(0,0,1))
    model.validate(dtype=DTYPE,device=torch.device('cpu'))
    assert model.evaluate(state(position=(0,0,1)),10,t((0,0,0))).tau_body.tolist()==pytest.approx([0]*6)
    assert model.evaluate(state(position=(0,0,1.2)),10,t((0,0,0))).tau_body.tolist()==pytest.approx([0,0,-2,0,0,0])
    coupled=diagonal.copy(); coupled[2,4]=coupled[4,2]=1
    wrench=linear(coupled).evaluate(state(position=(0,0,.2)),10,t((0,0,0))).tau_body
    assert wrench.tolist()==pytest.approx([0,0,-2,0,-.2,0])

def test_reference_point_transform_matches_direct_matrix_relation():
    source=t(np.diag([0,0,10,2,3,0])); r=t((-.2,0,0)); resolved=reference_point_transform(source,r)
    eta=t((0,0,.3,.1,-.2,0)); x,y,z=r; skew=t(((0,-z,y),(z,0,-x),(-y,x,0)))
    h=torch.cat((torch.cat((torch.eye(3,dtype=DTYPE),skew.T),1),torch.cat((torch.zeros((3,3),dtype=DTYPE),torch.eye(3,dtype=DTYPE)),1)),0)
    assert resolved@eta==pytest.approx((h.T@source@h@eta).tolist())

def test_negative_stiffness_fails_closed():
    with pytest.raises(PhysicalValidationError): linear(np.diag([0,0,-1,1,1,0])).validate(dtype=DTYPE,device=torch.device('cpu'))

def test_box_mesh_analytic_dry_partial_full_and_equilibrium():
    mesh=box_mesh(); model=MeshBuoyancy(mesh,1000,10,0,(-2,2),.8,.8)
    model.validate(dtype=DTYPE,device=torch.device('cpu'))
    dry=model.evaluate(state(position=(0,0,-1)),500,t((0,0,0)))
    half=model.evaluate(state(),500,t((0,0,0)))
    full=model.evaluate(state(position=(0,0,1)),500,t((0,0,0)))
    assert dry.diagnostics['submerged_volume_m3']==pytest.approx(0,abs=1e-12)
    assert half.diagnostics['submerged_volume_m3']==pytest.approx(1,abs=1e-12)
    assert half.diagnostics['center_of_buoyancy_frd_m'].tolist()==pytest.approx([0,0,.25],abs=1e-12)
    assert half.tau_body[2].item()==pytest.approx(5000-10000)
    assert full.diagnostics['submerged_volume_m3']==pytest.approx(2,abs=1e-12)
    solved=solve_equilibrium_waterline(model,500)
    assert solved['equilibrium_heave_ned_m']==pytest.approx(-.25,abs=1e-9)
    assert abs(solved['residual_kg'])<1e-8

def test_mesh_symmetry_roll_pitch_and_no_level_lateral_moment():
    model=MeshBuoyancy(box_mesh(),1000,9.81,0,(-2,2),.8,.8); cg=t((0,0,0))
    level=model.evaluate(state(),1000,cg).tau_body
    plus=model.evaluate(state(rpy=(.1,0,0)),1000,cg).tau_body
    minus=model.evaluate(state(rpy=(-.1,0,0)),1000,cg).tau_body
    assert level[[3,4]].tolist()==pytest.approx([0,0],abs=1e-9)
    assert plus[3].item()==pytest.approx(-minus[3].item(),rel=1e-9,abs=1e-9)
    pplus=model.evaluate(state(rpy=(0,.1,0)),1000,cg).tau_body
    pminus=model.evaluate(state(rpy=(0,-.1,0)),1000,cg).tau_body
    assert pplus[4].item()==pytest.approx(-pminus[4].item(),rel=1e-9,abs=1e-9)

def crossflow(strips=20,vertical=False):
    return StripTheoryCrossflow.constant_section(2,.25,.13414634146341461,strips,include_vertical=vertical,dtype=DTYPE)

def test_crossflow_invariants_and_energy():
    model=crossflow(); model.validate(dtype=DTYPE,device=torch.device('cpu'))
    assert model.evaluate(state()).tau_body.tolist()==pytest.approx([0]*6)
    positive=model.evaluate(state(nu=(0,.4,0,0,0,0))).tau_body
    negative=model.evaluate(state(nu=(0,-.4,0,0,0,0))).tau_body
    yaw=model.evaluate(state(nu=(0,0,0,0,0,.3))).tau_body
    assert positive[1]<0 and positive[5]==pytest.approx(0,abs=1e-12)
    assert negative.tolist()==pytest.approx((-positive).tolist())
    assert yaw[5]<0 and yaw[1]==pytest.approx(0,abs=1e-12)
    mixed=state(nu=(0,.4,0,0,0,.3)); tau=model.evaluate(mixed).tau_body
    assert float(mixed.nu_body@tau)<=1e-12
    matched=model.evaluate(state(nu=(0,.4,0,0,0,0)),t((0,.4,0))).tau_body
    assert matched.tolist()==pytest.approx([0]*6)

def test_crossflow_resolution_converges():
    values=[abs(crossflow(n).evaluate(state(nu=(0,0,0,0,0,.3))).tau_body[5].item()) for n in (10,20,40,80)]
    assert abs(values[-1]-values[-2])<abs(values[-2]-values[-3])<abs(values[-3]-values[-4])

def test_hydrodynamic_configuration_fingerprint_and_determinism():
    base={"hydrostatics":{"model":"linear_matrix","K":[[1.0]]},"crossflow":{"model":"strip_theory","strips":20},"mesh_sha256":"a"*64}
    changed={**base,"crossflow":{**base["crossflow"],"strips":40}}
    assert content_hash(base)!=content_hash(changed)
    model=crossflow(); vessel=state(nu=(0,.3,.1,0,.1,.2))
    assert torch.equal(model.evaluate(vessel).tau_body,model.evaluate(vessel).tau_body)
    mesh_model=MeshBuoyancy(box_mesh(),1000,9.81,0,(-2,2),.8,.8)
    assert torch.equal(mesh_model.evaluate(state(),500,t((0,0,0))).tau_body,
                       mesh_model.evaluate(state(),500,t((0,0,0))).tau_body)
