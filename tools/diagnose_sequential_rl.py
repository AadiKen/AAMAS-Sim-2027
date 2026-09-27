"""Long-horizon 1D actor-critic diagnostics without BCOD dynamics."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from bcod_sim.benchmark.audit_learning import initialize_single_agent_policy
from bcod_sim.benchmark.core import BenchmarkConfig, NAMES, VesselReading, observation
from bcod_sim.benchmark.runner import SharedActorCritic, _action, _update


def main():
    torch.set_num_threads(1)
    config=BenchmarkConfig()
    goal=37.4
    horizon=600
    output=[]
    for variant in ('dense','benchmark'):
        torch.manual_seed(11)
        np.random.seed(11)
        model=SharedActorCritic()
        def obs(x,previous_surge):
            reading=VesselReading(x,0.,0.,previous_surge*config.max_surge_mps,0.)
            return {NAMES[0]:observation(reading,(goal,0.),config)}
        initialize_single_agent_policy(model,obs(0.,0.)[NAMES[0]])
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        x=0.;previous_surge=0.;episode_steps=0
        records=[];evaluations=[];updates=[]
        def evaluate(step):
            position=0.;last_surge=0.;surges=[]
            for tick in range(horizon):
                with torch.no_grad():
                    action,_,_=_action(model,obs(position,last_surge),'cpu',True,agent_names=(NAMES[0],))
                surge=action[NAMES[0]][0]
                surges.append(surge)
                position+=surge*config.max_surge_mps*config.dt_s
                last_surge=surge
                if variant=='benchmark' and goal-position<=config.goal_radius_m:break
            return {'environment_steps':step,'deterministic_surge':float(np.mean(surges)),
                    'success':goal-position<=config.goal_radius_m,
                    'final_distance_m':max(0.,goal-position),'episode_length':tick+1}
        evaluations.append(evaluate(0))
        for step in range(1,5001):
            action,logp,value=_action(model,obs(x,previous_surge),'cpu',agent_names=(NAMES[0],))
            surge=action[NAMES[0]][0]
            new_x=x+surge*config.max_surge_mps*config.dt_s
            episode_steps+=1
            reached=goal-new_x<=config.goal_radius_m
            done=reached if variant=='benchmark' else False
            truncated=episode_steps>=horizon and not done
            if variant=='dense':
                reward=new_x-x
            else:
                phi_before=-(goal-x)
                phi_after=0. if done else -(goal-new_x)
                reward=config.gamma*phi_after-phi_before
                reward+=config.goal_bonus*done-config.step_penalty
            next_observation=obs(new_x,surge)
            boundary=None
            if done:boundary=torch.zeros_like(value)
            elif truncated:
                with torch.no_grad():boundary=model(torch.as_tensor(next_observation[NAMES[0]]).unsqueeze(0))[1]
            records.append((0,logp,value,torch.tensor([reward]),boundary))
            x=new_x;previous_surge=surge
            if done or truncated:x=0.;previous_surge=0.;episode_steps=0
            if step%250==0:
                with torch.no_grad():bootstrap=model(torch.as_tensor(obs(x,previous_surge)[NAMES[0]]).unsqueeze(0))[1]
                diagnostics={}
                _update(model,optimizer,records,{0:bootstrap},config.gamma,diagnostics)
                updates.append({'environment_steps':step,'actor_loss':diagnostics['actor_loss'],
                    'critic_loss':diagnostics['critic_loss'],'clip_scale':diagnostics['clip_scale']})
                records=[]
            if step in (1000,2500,5000):evaluations.append(evaluate(step))
        output.append({'variant':variant,'seed':11,'goal_m':goal,'horizon':horizon,
                       'gamma':config.gamma,'evaluations':evaluations,'updates':updates})
    destination=Path('artifacts/shared-rl-diagnosis/synthetic-sequential.json')
    destination.write_text(json.dumps(output,indent=2)+'\n')
    for result in output:
        print(result['variant'],[(r['environment_steps'],round(r['deterministic_surge'],3),r['success'])
                                 for r in result['evaluations']])


if __name__=='__main__':main()
