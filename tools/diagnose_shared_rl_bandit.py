"""One-step, one-agent surge bandit using the unchanged shared actor-critic update."""
import json
from pathlib import Path

import numpy as np
import torch

from bcod_sim.benchmark.audit_learning import initialize_single_agent_policy
from bcod_sim.benchmark.core import BenchmarkConfig, NAMES
from bcod_sim.benchmark.runner import SharedActorCritic, _action, _update


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    np.random.seed(11)
    model=SharedActorCritic()
    obs=np.zeros(47,dtype=np.float32)
    initialize_single_agent_policy(model,obs)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    observations={NAMES[0]:obs}
    def deterministic_surge():
        with torch.no_grad():
            actions,_,_=_action(model,observations,'cpu',True,agent_names=(NAMES[0],))
        return actions[NAMES[0]][0]
    rows=[{'environment_steps':0,'deterministic_surge':deterministic_surge()}]
    for update in range(1,21):
        records=[];rewards=[]
        for _ in range(250):
            actions,logp,value=_action(model,observations,'cpu',agent_names=(NAMES[0],))
            # Stateless terminal reward; optimal deterministic surge is 1.
            reward=actions[NAMES[0]][0]
            rewards.append(reward)
            records.append((0,logp,value,torch.tensor([reward]),torch.zeros_like(value)))
        assert all(r[1].shape==r[2].shape==r[3].shape==(1,) for r in records)
        diagnostics={}
        _update(model,optimizer,records,{0:torch.zeros(1)},BenchmarkConfig().gamma,diagnostics)
        rows.append({'environment_steps':update*250,'deterministic_surge':deterministic_surge(),
                     'mean_sampled_reward':float(np.mean(rewards)),
                     'actor_loss':diagnostics['actor_loss'],'critic_loss':diagnostics['critic_loss'],
                     'clip_scale':diagnostics['clip_scale']})
    destination=Path('artifacts/shared-rl-diagnosis/synthetic-bandit.json')
    destination.write_text(json.dumps({'task':'one-step surge reward r=surge; optimal surge=1',
                                       'seed':11,'steps':5000,'update_interval':250,'rows':rows},indent=2)+'\n')
    print('initial',rows[0]['deterministic_surge'],'final',rows[-1]['deterministic_surge'])


if __name__=='__main__':
    main()
