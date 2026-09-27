"""Repeated Case A policy passes use stored actions and a clipped ratio."""
import numpy as np
import pytest
import torch

from bcod_sim.benchmark.audit_learning import (case_a_rollout_action,
    initialize_case_a_yaw_raw_std, initial, ppo_case_a_update,
    transformed_log_probability)


def test_case_a_rollout_keeps_both_action_dimensions_trainable():
    model=initial(11)
    initialize_case_a_yaw_raw_std(model,.1)
    action,obs,latent,old_logp,old_value=case_a_rollout_action(
        model,np.zeros(47,dtype=np.float32))
    assert 0<=action[0]<=1 and -1<=action[1]<=1
    assert obs.shape==(47,) and latent.shape==(2,)
    assert old_value.ndim==old_logp.ndim==0
    with torch.no_grad():
        mean,_=model(obs.unsqueeze(0))
        new_logp=transformed_log_probability(model,mean,latent.unsqueeze(0))[0]
    assert float(new_logp)==pytest.approx(float(old_logp),abs=1e-6)


def test_case_a_ppo_repeated_passes_update_yaw_with_fixed_targets():
    model=initial(11)
    initialize_case_a_yaw_raw_std(model,.1)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    collected=[case_a_rollout_action(model,np.zeros(47,dtype=np.float32)) for _ in range(10)]
    observations=torch.stack([row[1] for row in collected])
    latents=torch.stack([row[2] for row in collected])
    old_logp=torch.stack([row[3] for row in collected])
    targets=torch.ones(10)
    advantages=torch.ones(10)
    yaw_before=model.actor.bias[1].detach().clone()
    diagnostics=ppo_case_a_update(model,optimizer,observations,latents,old_logp,
                                  targets,advantages,epochs=3)
    assert len(diagnostics)==3
    assert [row['optimization_epoch'] for row in diagnostics]==[1,2,3]
    assert diagnostics[0]['policy_ratio_mean']==pytest.approx(1.,abs=1e-6)
    assert diagnostics[0]['ppo_clip_fraction']==0.
    assert all(row['actor_gradient_norm']>0 for row in diagnostics)
    assert model.actor.bias[1].detach()!=yaw_before
    assert torch.equal(targets,torch.ones(10))
