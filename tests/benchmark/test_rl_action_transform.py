"""The shared Gaussian transform has a one-to-one forward-speed mapping."""
import math

import numpy as np
import pytest
import torch

from bcod_sim.benchmark.core import NAMES, validate_actions
from bcod_sim.benchmark.audit_learning import (initialize_single_agent_policy,
    initialize_case_a_yaw_raw_std, surge_only_action)
from bcod_sim.benchmark.runner import SharedActorCritic, _action


def transformed(z_surge, z_yaw):
    model=SharedActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.actor.bias.copy_(torch.tensor([z_surge,z_yaw]))
    observations={NAMES[0]:np.zeros(47,dtype=np.float32)}
    actions,logp,_=_action(model,observations,'cpu',deterministic=True,agent_names=(NAMES[0],))
    return actions[NAMES[0]],float(logp[0].detach())


def test_action_ranges_monotonicity_and_no_negative_surge_clipping():
    values=[transformed(z,0)[0][0] for z in (-8,-.9,-.5,-.1,0,.1,.5,.9,8)]
    assert all(a<b for a,b in zip(values,values[1:]))
    assert values[0]<1e-6 and values[-1]>1-1e-6
    assert values[4]==pytest.approx(.5)
    assert [transformed(0,z)[0][1] for z in (-2,0,2)]==pytest.approx(
        [math.tanh(-2),0,math.tanh(2)])


def test_transformed_log_probability_includes_half_scale_jacobian():
    z_surge,z_yaw=.4,-.7
    _,actual=transformed(z_surge,z_yaw)
    # Deterministic action evaluates the density at the Gaussian mean.
    normal_logp=-math.log(2*math.pi)
    epsilon=1e-4
    surge=lambda z:(math.tanh(z)+1)/2
    yaw=math.tanh
    surge_jac=(surge(z_surge+epsilon)-surge(z_surge-epsilon))/(2*epsilon)
    yaw_jac=(yaw(z_yaw+epsilon)-yaw(z_yaw-epsilon))/(2*epsilon)
    assert actual==pytest.approx(normal_logp-math.log(surge_jac*yaw_jac),abs=2e-6)


def test_common_action_contract_rejects_negative_surge():
    valid={name:(0.,0.) for name in NAMES}
    validate_actions(valid)
    valid[NAMES[0]]=(-.1,0.)
    with pytest.raises(ValueError,match='surge in'):
        validate_actions(valid)
    valid[NAMES[0]]=(.5,1.1)
    with pytest.raises(ValueError,match='yaw in'):
        validate_actions(valid)


def test_single_agent_forward_has_one_log_probability_and_one_value(monkeypatch):
    model=SharedActorCritic()
    seen=[]
    original=torch.distributions.Normal.sample
    def sample(distribution, *args, **kwargs):
        seen.append(tuple(distribution.loc.shape))
        return original(distribution,*args,**kwargs)
    monkeypatch.setattr(torch.distributions.Normal,'sample',sample)
    obs={NAMES[0]:np.zeros(47,dtype=np.float32)}
    actions,logp,value=_action(model,obs,'cpu',agent_names=(NAMES[0],))
    assert set(actions)=={NAMES[0]}
    assert logp.shape==value.shape==(1,)
    assert seen==[(1,2)]


def test_diagnostic_initial_bias_requests_five_percent_surge():
    model=SharedActorCritic()
    own=np.zeros(47,dtype=np.float32)
    initialize_single_agent_policy(model,own)
    actions,_,_=_action(model,{NAMES[0]:own},'cpu',deterministic=True,
                        agent_names=(NAMES[0],))
    assert actions[NAMES[0]]==pytest.approx((.05,0.),abs=1e-6)


def test_surge_only_diagnostic_never_samples_or_scores_yaw(monkeypatch):
    model=SharedActorCritic()
    seen=[]
    original=torch.distributions.Normal.sample
    def sample(distribution,*args,**kwargs):
        seen.append(tuple(distribution.loc.shape))
        return original(distribution,*args,**kwargs)
    monkeypatch.setattr(torch.distributions.Normal,'sample',sample)
    actions,logp,value=surge_only_action(model,np.zeros(47,dtype=np.float32))
    assert actions[NAMES[0]][1]==0.
    assert logp.shape==value.shape==(1,)
    assert seen==[(1,)]
    (-logp.mean()).backward()
    assert model.actor.bias.grad[1]==0.
    assert model.log_std.grad[1]==0.


def test_reduced_initial_yaw_std_preserves_surge_std_and_yaw_gradients():
    model=SharedActorCritic()
    surge_std=float(model.log_std[0].exp().detach())
    initialize_case_a_yaw_raw_std(model,0.10)
    assert float(model.log_std[0].exp().detach())==pytest.approx(surge_std)
    assert float(model.log_std[1].exp().detach())==pytest.approx(0.10)
    obs={NAMES[0]:np.zeros(47,dtype=np.float32)}
    _,logp,_=_action(model,obs,'cpu',agent_names=(NAMES[0],))
    (-logp.mean()).backward()
    assert model.actor.bias.grad[1]!=0.
    assert model.log_std.grad[1]!=0.
