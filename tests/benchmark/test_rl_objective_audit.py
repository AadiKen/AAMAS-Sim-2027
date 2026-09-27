import numpy as np
import math
import pytest
import torch
from bcod_sim.benchmark.runner import SharedActorCritic, _action, _return_targets, _update
from bcod_sim.benchmark.core import NAMES, VesselReading, observation, BenchmarkConfig


def record(env,reward,boundary=None):
    return (env,torch.zeros(1),torch.zeros(1),torch.tensor([float(reward)],requires_grad=True),
            None if boundary is None else torch.tensor([float(boundary)],requires_grad=True))


def test_exact_returns_and_episode_environment_isolation():
    records=[record(0,1),record(1,10,0),record(0,1),record(0,1,0),record(0,7,0)]
    targets=_return_targets(records,{0:torch.tensor([99.]),1:torch.tensor([99.])},.5)
    assert [r.item() for r in targets]==[1.75,10.,1.5,1.,7.]
    assert all(not r.requires_grad for r in targets)
    assert _return_targets([record(0,1,4)],{0:torch.tensor([99.])},.5)[0].item()==3
    assert _return_targets([record(0,1)],{0:torch.tensor([4.])},.5)[0].item()==3


def test_squashed_score_gradient_mean_std_and_detached_advantage(monkeypatch):
    model=SharedActorCritic()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    monkeypatch.setattr(torch.distributions.Normal,'sample',lambda self:torch.ones_like(self.loc)*.5)
    obs={n:np.zeros(47,dtype=np.float32) for n in NAMES}
    actions,logp,value=_action(model,obs,'cpu')
    raw=torch.full((4,2),.5)
    expected=(torch.distributions.Normal(torch.zeros_like(raw),torch.ones_like(raw)).log_prob(raw)-torch.log(1-torch.tanh(raw)**2)).sum(-1)-math.log(.5)
    assert torch.allclose(logp,expected)
    actor_loss=-(logp*(torch.ones(4)-value).detach()).mean()
    actor_loss.backward()
    assert torch.allclose(model.actor.bias.grad,torch.tensor([-.5,-.5]))
    assert torch.allclose(model.log_std.grad,torch.tensor([.75,.75]))
    assert model.critic.weight.grad is None
    assert actions[NAMES[0]][0]==pytest.approx((np.tanh(.5)+1)/2)


def test_single_frame_aliases_other_agent_motion():
    config=BenchmarkConfig()
    # Heading/velocity of the other vessel never appear in VesselReading's proximity slots.
    readings=[VesselReading(0,0,0,1,0,nearby_agents=[(10.,0.)]) for _ in range(3)]
    obs=[observation(r,(20,0),config) for r in readings]
    assert all(np.array_equal(obs[0],o) for o in obs[1:])


def test_collision_terminates_fleet_and_penalizes_every_agent():
    from bcod_sim.benchmark.qualify_scoring import reward_audit
    rows=reward_audit()
    collision=next(r for r in rows if r['trajectory']=='progress_then_collision')
    assert collision['colliding_agents']==['vessel_0']
    assert collision['physical_colliders']==['vessel_0']
    assert collision['fleet_failure_due_to_collision']
    assert collision['terminated'] and not collision['truncated']
    assert all(collision['reward_components'][n]['collision']==-50 for n in NAMES)
    assert collision['steps'] < BenchmarkConfig().max_steps


def test_no_collision_has_no_penalty():
    from bcod_sim.benchmark.qualify_scoring import reward_audit
    rows=reward_audit()
    safe=next(r for r in rows if r['trajectory']=='safe_detour_success')
    assert not safe['fleet_failure_due_to_collision']
    assert safe['physical_colliders']==[]
    assert all(safe['reward_components'][n]['collision']==0 for n in NAMES)


def test_time_based_discount():
    assert BenchmarkConfig().gamma == pytest.approx(0.99 ** BenchmarkConfig().dt_s)


def test_diagnostics_do_not_change_optimizer_update():
    import copy
    torch.manual_seed(3)
    first=SharedActorCritic(); second=copy.deepcopy(first)
    obs={n:np.zeros(47,dtype=np.float32) for n in NAMES}
    for model,diagnostics in ((first,None),(second,{})):
        torch.manual_seed(17)
        records=[]
        for index in range(3):
            _,logp,value=_action(model,obs,'cpu')
            records.append((0,logp,value,torch.ones(4),torch.zeros(4) if index==2 else None))
        _update(model,torch.optim.Adam(model.parameters(),lr=3e-4),records,{0:torch.zeros(4)},diagnostics=diagnostics)
    assert all(torch.equal(a,b) for a,b in zip(first.parameters(),second.parameters()))
    assert diagnostics['gradient_norms_before_clip']['actor_head']>0
    assert len(diagnostics['per_agent_actor_grad_norm'])==4


def test_sparse_diagnostic_scenarios_require_explicit_opt_in():
    from bcod_sim.benchmark.core import Benchmark, Scenario, VesselPose
    from bcod_sim.benchmark.qualify_scoring import PrescribedTrajectory, path
    starts=tuple(VesselPose(0.,float(i*10),0.) for i in range(4))
    scenario=Scenario('nominal',1,starts,tuple((20.,p.y_m) for p in starts),())
    adapter=PrescribedTrajectory(path([],1.))
    with pytest.raises(ValueError,match='6–10 obstacles'):
        Benchmark(adapter).reset(scenario)
    assert len(Benchmark(adapter,diagnostic_allow_sparse=True).reset(scenario))==4
