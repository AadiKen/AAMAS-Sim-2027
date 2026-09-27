"""Bounded diagnostic experiments; never changes production reward or simulator physics."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .core import (Benchmark, BenchmarkConfig, NAMES, Scenario, VesselPose,
                   Obstacle, VesselReading, generate_scenario, observation)
from .runner import SharedActorCritic, make_adapter, _action, _update, _return_targets, _append
from .rl_diagnostics import EpisodeDiagnostics, action_summary


def scenario(kind, seed):
    s=generate_scenario(seed,split='nominal')
    if kind=='F':return s
    if kind=='E':return replace(s,obstacles=())
    starts=(VesselPose(-20,0,math.pi/2 if kind=='B' else 0),
            VesselPose(-40,40,0),VesselPose(0,40,0),VesselPose(40,40,0))
    obstacles=() if kind in 'AB' else (Obstacle(0,0,2),)
    if kind=='D':obstacles=tuple(Obstacle(x,y,2) for x,y in ((-8,1),(4,-2),(13,2),(-12,-12),(0,15),(20,-15)))
    return Scenario('nominal',seed,starts,((20,0),(-40,40),(0,40),(40,40)),obstacles)


def initial(seed=11):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    return SharedActorCritic()


def initialize_single_agent_policy(model, own_observation, target_surge=0.05):
    """Set the diagnostic start to 5% throttle and zero yaw at case A's initial state."""
    if not 0 < target_surge < 1:
        raise ValueError("Initial surge must be strictly between zero and one")
    desired=torch.tensor([math.atanh(2*target_surge-1),0.],dtype=torch.float32)
    with torch.no_grad():
        tensor=torch.as_tensor(own_observation,dtype=torch.float32).unsqueeze(0)
        current,_=model(tensor)
        model.actor.bias.add_(desired-current[0])


def initialize_case_a_yaw_raw_std(model, raw_std):
    """Set only the diagnostic starting spread; yaw remains trainable."""
    # _action maps latent index 0 to surge and index 1 to yaw.
    with torch.no_grad():model.log_std[1]=math.log(raw_std)


def surge_only_action(model, own_observation, deterministic=False, diagnostics=None):
    """Diagnostic policy: sample surge only; yaw is exactly zero and absent from logp."""
    tensor=torch.as_tensor(own_observation,dtype=torch.float32).unsqueeze(0)
    mean,value=model(tensor)
    distribution=torch.distributions.Normal(mean[:,0],model.log_std[0].exp())
    latent=mean[:,0] if deterministic else distribution.sample()
    surge=(torch.tanh(latent)+1)/2
    correction=2*(math.log(2.)-latent-torch.nn.functional.softplus(-2*latent))+math.log(.5)
    logp=distribution.log_prob(latent)-correction
    action={NAMES[0]:(float(surge[0].detach()),0.)}
    if diagnostics is not None:
        diagnostics.update(actor_mean=[[(float((torch.tanh(mean[0,0])+1).detach()/2),0.)]],
                           sampled=[[(action[NAMES[0]][0],0.)]],
                           log_std=model.log_std.detach().cpu().tolist(),
                           raw_std=model.log_std.exp().detach().cpu().tolist())
    return action,logp,value


def case_a_rollout_action(model, own_observation):
    """Sample both transformed actions and retain the latent for PPO ratios."""
    tensor=torch.as_tensor(own_observation,dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean,value=model(tensor)
        latent=torch.distributions.Normal(mean,model.log_std.exp()).sample()
        surge=(torch.tanh(latent[0,0])+1)/2
        yaw=torch.tanh(latent[0,1])
        old_logp=transformed_log_probability(model,mean,latent)[0]
    return (float(surge),float(yaw)),tensor[0],latent[0],old_logp,value[0]


def transformed_log_probability(model, mean, latent):
    """Exact density for surge=(tanh(z0)+1)/2 and yaw=tanh(z1)."""
    distribution=torch.distributions.Normal(mean,model.log_std.exp())
    correction=2*(math.log(2.)-latent-torch.nn.functional.softplus(-2*latent))
    return (distribution.log_prob(latent)-correction).sum(-1)-math.log(.5)


def ppo_case_a_update(model, optimizer, observations, latents, old_logp,
                      targets, advantages, *, epochs=25, epsilon=.2):
    """Diagnostic PPO-style repeated optimization on fixed on-policy targets."""
    def norm(grads):
        return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))
    diagnostics=[]
    body=list(model.body.parameters())
    for epoch in range(epochs):
        mean,value=model(observations)
        new_logp=transformed_log_probability(model,mean,latents)
        ratio=torch.exp(new_logp-old_logp)
        clipped=ratio.clamp(1-epsilon,1+epsilon)
        actor_loss=-torch.minimum(ratio*advantages,clipped*advantages).mean()
        critic_loss=.5*(targets-value).square().mean()
        actor_body=torch.autograd.grad(actor_loss,body,retain_graph=True,allow_unused=True)
        critic_body=torch.autograd.grad(critic_loss,body,retain_graph=True,allow_unused=True)
        actor_head=torch.autograd.grad(actor_loss,list(model.actor.parameters())+[model.log_std],
                                       retain_graph=True,allow_unused=True)
        critic_head=torch.autograd.grad(critic_loss,model.critic.parameters(),retain_graph=True,
                                        allow_unused=True)
        optimizer.zero_grad()
        (actor_loss+critic_loss).backward()
        combined_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        optimizer.step()
        diagnostics.append({'optimization_epoch':epoch+1,
            'actor_loss':float(actor_loss.detach()),'critic_loss':float(critic_loss.detach()),
            'actor_gradient_norm':math.hypot(norm(actor_body),norm(actor_head)),
            'critic_gradient_norm':math.hypot(norm(critic_body),norm(critic_head)),
            'combined_gradient_norm':combined_norm,
            'global_clip_coefficient':min(1.,1./(combined_norm+1e-6)),
            'policy_ratio_mean':float(ratio.detach().mean()),
            'ppo_clip_fraction':float(((ratio.detach()<1-epsilon)|(ratio.detach()>1+epsilon)).float().mean()),
            'raw_surge_std':float(model.log_std[0].exp().detach()),
            'raw_yaw_std':float(model.log_std[1].exp().detach())})
    return diagnostics


def summary(rows):
    collisions=[r for r in rows if r['collision_count']]
    return {'episodes':len(rows),'fleet_success':float(np.mean([r['fleet_success'] for r in rows])),
            'per_agent_success':{n:float(np.mean([r['per_agent_success'][n] for r in rows])) for n in NAMES},
            'collision_rate':len(collisions)/len(rows),'mean_return':float(np.mean([r['fleet_return'] for r in rows])),
            'mean_discounted_return':float(np.mean([r['discounted_fleet_return'] for r in rows])),
            'mean_episode_length':float(np.mean([r['steps'] for r in rows])),
            'mean_distance_to_goal_m':float(np.mean([r['mean_distance_to_goal_m'] for r in rows])),
            'median_episode_length':float(np.median([r['steps'] for r in rows])),
            'path_efficiency':float(np.mean([v for r in rows for v in r['path_efficiency'].values()])),
            'collision_positive_fleet_fraction':float(np.mean([r['fleet_return']>0 for r in collisions])) if collisions else None,
            'collision_positive_discounted_fleet_fraction':float(np.mean([r['discounted_fleet_return']>0 for r in collisions])) if collisions else None,
            'collision_three_positive_agents_fraction':float(np.mean([r['positive_return_agents']>=3 for r in collisions])) if collisions else None,
            'reached_then_left_episodes':sum(bool(r['reached_then_left']) for r in rows),
            'post_goal_collision_episodes':sum(bool(r['post_goal_collisions']) for r in rows)}


def geometric(obs,env,truth_access=False):
    # Reactive goal/repulsion baseline; deliberately no hidden state for obs variant.
    actions={}
    for i,n in enumerate(NAMES):
        o=obs[n]; x,y=o[:2]*env.config.half_width_m
        heading=math.atan2(o[2],o[3]);gx,gy=o[6:8]*(2*env.config.half_width_m)
        d=math.hypot(gx,gy)
        direction=np.array([gx,gy])/max(d,1e-9)
        objects=[]
        if truth_access:
            objects=[(p.x_m-x,p.y_m-y,3.2) for p in env.scenario.obstacles]
            objects += [(p.x_m-x,p.y_m-y,3.) for key,p in env.truth.items() if key!=n]
        else:
            for slot in range(13):
                r,si,co=o[8+3*slot:11+3*slot]
                if r:
                    angle=heading+math.atan2(si,co)
                    objects.append((r*30*math.cos(angle),r*30*math.sin(angle),3.2))
        clearance=30.
        for dx,dy,radius in objects:
            dist=math.hypot(dx,dy)
            if dist<12:
                unit=np.array([dx,dy])/max(dist,1e-9)
                # Fixed handedness to escape head-on local minima.
                direction -= unit*2*max(0.,(12-dist)/12)
                direction += np.array([-unit[1],unit[0]])*max(0.,(10-dist)/10)
                if dx*math.cos(heading)+dy*math.sin(heading)>0:clearance=min(clearance,dist-radius)
        error=math.atan2(direction[1],direction[0])-heading
        error=math.atan2(math.sin(error),math.cos(error))
        speed=.5*max(.1,math.cos(error))*max(.1,min(1.,clearance/5))
        actions[n]=(0.,0.) if d<=2 else (speed,max(-1.,min(1.,error/.25)))
    return actions


def evaluate(model,env,kind,count,deterministic,destination,step=0,controller=None,surge_only=False):
    rows=[]; stats=[]
    for index in range(count):
        torch.manual_seed(800000+index)
        obs=env.reset(scenario(kind,700000+index));episode=EpisodeDiagnostics()
        for _ in range(env.config.max_steps):
            with torch.no_grad():
                row={}
                if controller:
                    actions=geometric(obs,env,controller=='truth')
                elif surge_only:
                    actions,_,_=surge_only_action(model,obs[NAMES[0]],deterministic,row)
                elif kind in 'ABCD':
                    actions,_,_=_action(model,{NAMES[0]:obs[NAMES[0]]},'cpu',deterministic,
                                        diagnostics=row,agent_names=(NAMES[0],))
                else:
                    actions,_,_=_action(model,obs,'cpu',deterministic,diagnostics=row)
                if kind in 'ABCD':
                    for n in NAMES[1:]:actions[n]=(0.,0.)
                if not controller:
                    if surge_only:
                        row['actor_mean']=row['actor_mean'][0]
                        row['sampled']=row['sampled'][0]
                    stats.append(row)
            obs,reward,done,truncated,info=env.step(actions)
            if kind in 'ABCD':
                assert set(info['physical_colliders']).issubset({NAMES[0]})
                assert not done or info['per_agent_success'][NAMES[0]] or NAMES[0] in info['physical_colliders']
            episode.add(reward,info,env)
            if done or truncated:break
        active=range(1) if kind in 'ABCD' else range(4)
        final_distance=float(np.mean([math.dist((env.truth[NAMES[i]].x_m,env.truth[NAMES[i]].y_m),
                                               env.scenario.goals[i]) for i in active]))
        result={'scenario_seed':700000+index,'mean_distance_to_goal_m':final_distance,**episode.result(info)}
        rows.append(result);_append(destination,result)
        if (index+1)%10==0:print(f'eval {kind} {step} {index+1}/{count}',flush=True)
    return {'environment_steps':step,'deterministic':deterministic,'controller':controller,
            **summary(rows),**(action_summary(stats) if controller is None else {})}


def diagnostic_update(model, optimizer, records, bootstrap, gamma, mode):
    """Diagnostic-only alternatives to the production joint global clip."""
    targets=torch.stack(_return_targets(records,bootstrap,gamma))
    values=torch.stack([r[2] for r in records])
    logp=torch.stack([r[1] for r in records])
    advantage=targets-values
    actor_advantage=advantage.detach()
    if mode=='normalized':
        actor_advantage=(actor_advantage-actor_advantage.mean())/(actor_advantage.std(unbiased=False)+1e-8)
    actor_loss=-(logp*actor_advantage).mean()
    critic_loss=.5*advantage.square().mean()
    optimizer.zero_grad()
    if mode=='normalized':
        (actor_loss+critic_loss).backward()
        combined_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        optimizer.step()
        return {'actor_loss':float(actor_loss.detach()),'critic_loss':float(critic_loss.detach()),
                'mean_advantage':float(advantage.detach().mean()),
                'std_advantage':float(advantage.detach().std(unbiased=False)),
                'global_clip_coefficient':min(1.,1./(combined_norm+1e-6))}
    actor_parameters=list(model.body.parameters())+list(model.actor.parameters())+[model.log_std]
    critic_parameters=list(model.body.parameters())+list(model.critic.parameters())
    actor_grads=torch.autograd.grad(actor_loss,actor_parameters,retain_graph=True,allow_unused=True)
    critic_grads=torch.autograd.grad(critic_loss,critic_parameters,allow_unused=True)
    def norm(grads):
        return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))
    actor_norm,critic_norm=norm(actor_grads),norm(critic_grads)
    actor_scale=min(1.,1./(actor_norm+1e-6))
    critic_scale=min(1.,1./(critic_norm+1e-6))
    gradient_by_id={id(p):torch.zeros_like(p) for p in model.parameters()}
    for p,g in zip(actor_parameters,actor_grads):
        if g is not None:gradient_by_id[id(p)].add_(g.detach(),alpha=actor_scale)
    for p,g in zip(critic_parameters,critic_grads):
        if g is not None:gradient_by_id[id(p)].add_(g.detach(),alpha=critic_scale)
    for p in model.parameters():p.grad=gradient_by_id[id(p)]
    optimizer.step()
    return {'actor_loss':float(actor_loss.detach()),'critic_loss':float(critic_loss.detach()),
            'mean_advantage':float(advantage.detach().mean()),
            'std_advantage':float(advantage.detach().std(unbiased=False)),
            'actor_grad_norm':actor_norm,'critic_grad_norm':critic_norm,
            'actor_clip_coefficient':actor_scale,'critic_clip_coefficient':critic_scale}


def reinforce_update(model,optimizer,records,gamma):
    """Full observed episode returns, normalized; no critic target or baseline."""
    assert records and records[-1][4] is not None
    rewards=[float(record[3][0]) for record in records]
    targets=[]
    for start in range(len(records)):
        targets.append(sum(gamma**offset*reward for offset,reward in enumerate(rewards[start:])))
    returns=torch.tensor(targets,dtype=torch.float32)
    advantages=(returns-returns.mean())/(returns.std(unbiased=False)+1e-8)
    logp=torch.stack([record[1][0] for record in records])
    actor_loss=-(logp*advantages.detach()).mean()
    optimizer.zero_grad()
    actor_loss.backward()
    actor_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    optimizer.step()
    return {'actor_loss':float(actor_loss.detach()),'critic_loss':None,
            'actor_grad_norm':actor_norm,'critic_grad_norm':0.,
            'return_mean':float(returns.mean()),'return_std':float(returns.std(unbiased=False)),
            'optimizer_records':len(records)}


def reference_targets(records,bootstrap,gamma):
    """Independent forward-sum target calculation, stopping at each boundary."""
    result=[]
    for start in range(len(records)):
        total=0.;discount=1.
        for _,_,_,reward,boundary in records[start:]:
            total+=discount*float(reward[0])
            discount*=gamma
            if boundary is not None:
                total+=discount*float(boundary[0])
                break
        else:
            total+=discount*float(bootstrap[0])
        result.append(total)
    return result


class SyntheticSequentialEnvironment:
    """The benchmark-like 1D environment from diagnose_sequential_rl.py."""
    def __init__(self, config):
        self.config=config
        self.goal_m=37.4
        self.horizon=600
        self.reset()

    def reset(self):
        self.x_m=0.
        self.previous_surge=0.
        self.episode_steps=0
        return self.observation()

    def observation(self):
        reading=VesselReading(self.x_m,0.,0.,
                              self.previous_surge*self.config.max_surge_mps,0.)
        return {NAMES[0]:observation(reading,(self.goal_m,0.),self.config)}

    def step(self, surge):
        new_x=self.x_m+surge*self.config.max_surge_mps*self.config.dt_s
        self.episode_steps+=1
        done=self.goal_m-new_x<=self.config.goal_radius_m
        truncated=self.episode_steps>=self.horizon and not done
        phi_before=-(self.goal_m-self.x_m)
        phi_after=0. if done else -(self.goal_m-new_x)
        reward=self.config.gamma*phi_after-phi_before
        reward+=self.config.goal_bonus*done-self.config.step_penalty
        self.x_m=new_x
        self.previous_surge=surge
        return self.observation(),reward,done,truncated


def run_update_interval_sweep(intervals=(250,100,50,20,10), total_steps=5000,
                              seed=11, output_path='artifacts/shared-rl-diagnosis/update-interval-sweep.json'):
    """Change only update frequency in the existing benchmark-like synthetic task."""
    if total_steps<1 or any(interval<1 for interval in intervals):
        raise ValueError('Steps and update intervals must be positive')
    config=BenchmarkConfig()
    torch.set_num_threads(1)
    results=[]
    for interval in intervals:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model=SharedActorCritic()
        env=SyntheticSequentialEnvironment(config)
        initialize_single_agent_policy(model,env.observation()[NAMES[0]])
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        records=[]
        sampled_window=[]
        measurements=[]
        optimizer_updates=0
        last_diagnostics=None

        def measure(step):
            evaluation=SyntheticSequentialEnvironment(config)
            surges=[]
            for _ in range(evaluation.horizon):
                with torch.no_grad():
                    actions,_,_=_action(model,evaluation.observation(),'cpu',True,
                                        agent_names=(NAMES[0],))
                surge=actions[NAMES[0]][0]
                surges.append(surge)
                _,_,done,truncated=evaluation.step(surge)
                if done or truncated:
                    break
            diagnostics=last_diagnostics
            if diagnostics is None:
                actor_norm=critic_norm=None
            else:
                gradients=diagnostics['gradient_norms_before_clip']
                actor_norm=math.sqrt(diagnostics['actor_body_grad_norm']**2+
                    gradients['actor_head']**2+gradients['log_std']**2)
                critic_norm=math.sqrt(diagnostics['critic_body_grad_norm']**2+
                    gradients['critic_head']**2)
            return {'update_interval':interval,'environment_steps':step,
                'optimizer_updates':optimizer_updates,
                'deterministic_surge':float(np.mean(surges)),
                'mean_sampled_surge':float(np.mean(sampled_window)) if sampled_window else None,
                'sample_window_start':step-len(sampled_window)+1 if sampled_window else None,
                'actor_loss':diagnostics['actor_loss'] if diagnostics else None,
                'critic_loss':diagnostics['critic_loss'] if diagnostics else None,
                'actor_gradient_norm':actor_norm,'critic_gradient_norm':critic_norm,
                'combined_gradient_norm':diagnostics['total_grad_norm_before_clip'] if diagnostics else None,
                'global_clip_coefficient':diagnostics['clip_scale'] if diagnostics else None,
                'log_std_surge':float(model.log_std[0].detach()),
                'log_std_yaw':float(model.log_std[1].detach()),
                'goal_success':evaluation.goal_m-evaluation.x_m<=config.goal_radius_m,
                'final_position':evaluation.x_m,
                'final_distance_to_goal':max(0.,evaluation.goal_m-evaluation.x_m),
                'episode_length':evaluation.episode_steps}

        measurements.append(measure(0))
        checkpoints={1000,2500,total_steps}
        for step in range(1,total_steps+1):
            actions,logp,value=_action(model,env.observation(),'cpu',agent_names=(NAMES[0],))
            surge=actions[NAMES[0]][0]
            sampled_window.append(surge)
            next_observation,reward,done,truncated=env.step(surge)
            boundary=None
            if done:
                boundary=torch.zeros_like(value)
            elif truncated:
                with torch.no_grad():
                    boundary=model(torch.as_tensor(next_observation[NAMES[0]]).unsqueeze(0))[1]
            records.append((0,logp,value,torch.tensor([reward]),boundary))
            if done or truncated:
                env.reset()
            if step%interval==0:
                with torch.no_grad():
                    bootstrap=model(torch.as_tensor(env.observation()[NAMES[0]]).unsqueeze(0))[1]
                last_diagnostics={}
                _update(model,optimizer,records,{0:bootstrap},config.gamma,last_diagnostics)
                optimizer_updates+=1
                records=[]
            if step in checkpoints:
                measurements.append(measure(step))
                sampled_window=[]
        results.append({'update_interval':interval,'seed':seed,
                        'total_steps':total_steps,'gamma':config.gamma,
                        'measurements':measurements,
                        'optimizer_updates':optimizer_updates,
                        'goal_success':measurements[-1]['goal_success'],
                        'final_position':measurements[-1]['final_position'],
                        'final_distance_to_goal':measurements[-1]['final_distance_to_goal']})
    destination=Path(output_path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(results,indent=2)+'\n')
    print('interval | updates | surge@1k | surge@2.5k | surge@5k | success')
    for result in results:
        by_step={row['environment_steps']:row for row in result['measurements']}
        def surge_at(step):
            return f"{by_step[step]['deterministic_surge']:.4f}" if step in by_step else '—'
        print(f"{result['update_interval']:>8} | {result['optimizer_updates']:>7} | "
              f"{surge_at(1000):>8} | {surge_at(2500):>10} | "
              f"{surge_at(5000):>8} | {result['goal_success']}")
    return results


def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=BenchmarkConfig();model=initial(args.seed)
    if args.initial_yaw_raw_std is not None:
        initialize_case_a_yaw_raw_std(model,args.initial_yaw_raw_std)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    env=Benchmark(make_adapter(args.sim,config,pyquaticus_python=args.pyquaticus_python),config,diagnostic_allow_sparse=True)
    eval_env=Benchmark(make_adapter(args.sim,config,pyquaticus_python=args.pyquaticus_python),config,diagnostic_allow_sparse=True)
    count=args.eval_count
    manifest=vars(args).copy();manifest['benchmark']=vars(config);manifest['gamma']=config.gamma
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    evaluations=[];optimizer_updates=0;last_update_diagnostics=None
    def check(step):
        torch_rng=torch.get_rng_state()
        numpy_rng=np.random.get_state()
        python_rng=random.getstate()
        try:
            result=evaluate(model,eval_env,args.kind,count,True,output/f'eval-{step}.jsonl',step,
                            surge_only=args.surge_only)
        finally:
            torch.set_rng_state(torch_rng)
            np.random.set_state(numpy_rng)
            random.setstate(python_rng)
        diagnostics=last_update_diagnostics
        actor_norm=critic_norm=None
        if diagnostics is not None:
            gradients=diagnostics.get('gradient_norms_before_clip')
            if gradients is not None:
                actor_norm=math.sqrt(diagnostics['actor_body_grad_norm']**2+
                    gradients['actor_head']**2+gradients['log_std']**2)
                critic_norm=math.sqrt(diagnostics['critic_body_grad_norm']**2+
                    gradients['critic_head']**2)
            else:
                actor_norm=diagnostics.get('actor_grad_norm')
                critic_norm=diagnostics.get('critic_grad_norm')
        result.update({'optimizer_updates':optimizer_updates,
                       'success':result['per_agent_success'][NAMES[0]] if args.kind in 'ABCD' else result['fleet_success'],
                       'success_count':round(result['per_agent_success'][NAMES[0]]*count) if args.kind in 'ABCD' else round(result['fleet_success']*count),
                       'collision':result['collision_rate'],
                       'collision_count':round(result['collision_rate']*count),
                       'final_distance_to_goal_m':result['mean_distance_to_goal_m'],
                       'episode_length':result['mean_episode_length'],
                       'deterministic_surge':result.get('actor_mean_surge'),
                       'deterministic_signed_yaw':result.get('actor_mean_yaw'),
                       'deterministic_abs_yaw':result.get('actor_mean_abs_yaw'),
                       'action_statistics_basis':'deterministic evaluation rollout average',
                       'raw_surge_std':float(model.log_std[0].exp().detach()),
                       'raw_yaw_std':float(model.log_std[1].exp().detach()),
                       'deterministic_yaw_near_saturation_fraction':result.get('near_limit_fraction',[None,None])[1],
                       'actor_loss':diagnostics.get('actor_loss') if diagnostics else None,
                       'critic_loss':diagnostics.get('critic_loss') if diagnostics else None,
                       'actor_gradient_norm':actor_norm,
                       'critic_gradient_norm':critic_norm})
        evaluations.append(result)
        (output/'evaluations.json').write_text(json.dumps(evaluations,indent=2))
    try:
        obs=env.reset(scenario(args.kind,1100000))
        if args.kind in 'ABCD':
            initialize_single_agent_policy(model,obs[NAMES[0]])
            manifest['initial_deterministic_surge']=0.05
            manifest['initial_deterministic_yaw']=0.0
            manifest['initial_raw_surge_std']=float(model.log_std[0].exp().detach())
            manifest['initial_raw_yaw_std']=float(model.log_std[1].exp().detach())
            manifest['learning_agents']=[NAMES[0]]
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        check(0)
        episode=EpisodeDiagnostics(config.gamma);records=[];action_rows=[];episodes=0
        trace=[];trace_episode=0
        for step in range(1,args.steps+1):
            row={}
            if args.kind in 'ABCD':
                if args.surge_only:
                    actions,logp,value=surge_only_action(model,obs[NAMES[0]],diagnostics=row)
                    row['actor_mean']=row['actor_mean'][0]
                    row['sampled']=row['sampled'][0]
                else:
                    actions,logp,value=_action(model,{NAMES[0]:obs[NAMES[0]]},'cpu',diagnostics=row,
                                                agent_names=(NAMES[0],))
                for n in NAMES[1:]:actions[n]=(0.,0.)
            else:
                actions,logp,value=_action(model,obs,'cpu',diagnostics=row)
            action_rows.append(row)
            obs,reward,done,truncated,info=env.step(actions)
            if args.kind in 'ABCD':
                assert set(info['physical_colliders']).issubset({NAMES[0]})
                assert not done or info['per_agent_success'][NAMES[0]] or NAMES[0] in info['physical_colliders']
            episode.add(reward,info,env)
            r=torch.tensor([reward[NAMES[0]]] if args.kind in 'ABCD' else
                           [reward[n] for n in NAMES],dtype=torch.float32)
            boundary=None
            if done:boundary=torch.zeros_like(value)
            elif truncated:
                names=(NAMES[0],) if args.kind in 'ABCD' else NAMES
                with torch.no_grad():boundary=model(torch.tensor(np.stack([obs[n] for n in names])))[1]
            if args.kind in 'ABCD':
                assert logp.shape==value.shape==r.shape==(1,)
                assert boundary is None or boundary.shape==(1,)
            records.append((0,logp,value,r,boundary))
            if args.surge_only and trace_episode==0:
                trace.append({'environment_steps':step,'surge':actions[NAMES[0]][0],
                              'reward':float(r[0]),'critic_prediction':float(value[0].detach()),
                              'boundary_value':None if boundary is None else float(boundary[0].detach()),
                              'episode_step':episode.steps if hasattr(episode,'steps') else info['steps']})
            if done or truncated:
                _append(output/'episodes.jsonl',{'environment_steps':step,**episode.result(info)})
                episodes+=1;obs=env.reset(scenario(args.kind,1100000+episodes));episode=EpisodeDiagnostics(config.gamma)
                if trace_episode==0:trace_episode=1
                if args.actor_only:
                    diag=reinforce_update(model,optimizer,records,config.gamma)
                    optimizer_updates+=1
                    last_update_diagnostics=diag
                    _append(output/'updates.jsonl',{'environment_steps':step,'optimizer_updates':optimizer_updates,
                        'agent_transitions_per_environment_step':1,**diag,**action_summary(action_rows)})
                    records=[];action_rows=[]
            if not args.actor_only and (step%args.update_interval==0 or step==args.steps):
                names=(NAMES[0],) if args.kind in 'ABCD' else NAMES
                with torch.no_grad():bootstrap=model(torch.tensor(np.stack([obs[n] for n in names])))[1]
                if args.kind in 'ABCD':
                    assert len(records)==(step-1)%args.update_interval+1
                    assert bootstrap.shape==(1,)
                    assert all(logp.shape==value.shape==reward.shape==(1,)
                               for _,logp,value,reward,_ in records)
                    assert torch.stack([record[1] for record in records]).shape==(len(records),1)
                    assert torch.stack([record[2] for record in records]).shape==(len(records),1)
                if args.surge_only and trace:
                    targets=_return_targets(records,{0:bootstrap},config.gamma)
                    reference=reference_targets(records,bootstrap,config.gamma)
                    assert np.allclose([float(t[0]) for t in targets],reference,atol=3e-4,rtol=1e-5)
                    for offset,target in enumerate(targets):
                        index=step-len(records)+offset
                        if index<len(trace):
                            trace[index]['return_target']=float(target[0])
                            trace[index]['advantage']=float((target[0]-records[offset][2][0]).detach())
                            trace[index]['update_boundary_bootstrap']=float(bootstrap[0])
                diag={}
                if args.update_mode=='global':
                    _update(model,optimizer,records,{0:bootstrap},gamma=config.gamma,diagnostics=diag)
                else:
                    diag=diagnostic_update(model,optimizer,records,{0:bootstrap},config.gamma,args.update_mode)
                optimizer_updates+=1
                last_update_diagnostics=diag
                _append(output/'updates.jsonl',{'environment_steps':step,'optimizer_updates':optimizer_updates,
                    'agent_transitions_per_environment_step':len(names),**diag,**action_summary(action_rows)})
                records=[];action_rows=[]
            if step in (1000,2500,args.steps):
                torch.save({'model':model.state_dict(),'environment_steps':step,'manifest':manifest},output/f'checkpoint-{step}.pt')
                check(step)
        if trace:
            (output/'first_episode_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    finally:env.close();eval_env.close()


def train_case_a_rollout_ppo(args):
    """Case A only: 250-step on-policy collection, then repeated clipped passes."""
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=BenchmarkConfig();model=initial(args.seed)
    initialize_case_a_yaw_raw_std(model,args.initial_yaw_raw_std)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    env=Benchmark(make_adapter('bcod-reduced',config),config,diagnostic_allow_sparse=True)
    eval_env=Benchmark(make_adapter('bcod-reduced',config),config,diagnostic_allow_sparse=True)
    manifest={k:v for k,v in vars(args).items() if k!='update_interval'}
    manifest.update({'algorithm':'PPO-style long-horizon diagnostic',
              'ppo_clip_epsilon':args.ppo_clip_epsilon,'benchmark':vars(config),
              'gamma':config.gamma,'learning_agents':[NAMES[0]],
              'initial_deterministic_surge':.05,'initial_deterministic_yaw':0.,
              'initial_raw_surge_std':float(model.log_std[0].exp().detach()),
              'initial_raw_yaw_std':float(model.log_std[1].exp().detach()),
              'target_convention':'fixed once per collected rollout; zero true-terminal bootstrap; final-state timeout bootstrap'})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rollouts=actor_passes=critic_passes=0
    last_diagnostic=None
    evaluations=[]
    def check(step):
        torch_rng=torch.get_rng_state();numpy_rng=np.random.get_state();python_rng=random.getstate()
        try:
            result=evaluate(model,eval_env,'A',args.eval_count,True,output/f'eval-{step}.jsonl',step)
        finally:
            torch.set_rng_state(torch_rng);np.random.set_state(numpy_rng);random.setstate(python_rng)
        rows=[json.loads(line) for line in (output/f'eval-{step}.jsonl').read_text().splitlines()]
        result.update({'success_count':sum(row['per_agent_success'][NAMES[0]] for row in rows),
                       'collision_count':sum(NAMES[0] in row['physical_colliders'] for row in rows),
                       'final_distance_to_goal_m':result['mean_distance_to_goal_m'],
                       'episode_length':result['mean_episode_length'],
                       'deterministic_surge':result['actor_mean_surge'],
                       'deterministic_signed_yaw':result['actor_mean_yaw'],
                       'deterministic_abs_yaw':result['actor_mean_abs_yaw'],
                       'deterministic_yaw_near_saturation_fraction':result['near_limit_fraction'][1],
                       'action_statistics_basis':'deterministic evaluation rollout average',
                       'raw_surge_std':float(model.log_std[0].exp().detach()),
                       'raw_yaw_std':float(model.log_std[1].exp().detach()),
                       'collected_rollouts':rollouts,'actor_optimizer_passes':actor_passes,
                       'critic_optimizer_passes':critic_passes,
                       'actor_loss':last_diagnostic['actor_loss'] if last_diagnostic else None,
                       'critic_loss':last_diagnostic['critic_loss'] if last_diagnostic else None,
                       'actor_gradient_norm':last_diagnostic['actor_gradient_norm'] if last_diagnostic else None,
                       'critic_gradient_norm':last_diagnostic['critic_gradient_norm'] if last_diagnostic else None,
                       'return_scope':'vessel_0_only',
                       'mean_return':float(np.mean([row['per_agent_return'][NAMES[0]] for row in rows])),
                       'mean_discounted_return':float(np.mean([row['per_agent_discounted_return'][NAMES[0]] for row in rows]))})
        evaluations.append(result)
        (output/'evaluations.json').write_text(json.dumps(evaluations,indent=2)+'\n')
        torch.save({'model':model.state_dict(),'environment_steps':step,'manifest':manifest},
                   output/f'checkpoint-{step}.pt')
    try:
        obs=env.reset(scenario('A',1100000))
        initialize_single_agent_policy(model,obs[NAMES[0]])
        check(0)
        episode=EpisodeDiagnostics(config.gamma);episode_index=0
        for start in range(0,args.steps,args.rollout_horizon):
            collected=[];records=[];action_rows=[]
            length=min(args.rollout_horizon,args.steps-start)
            assert length==250==args.rollout_horizon
            for offset in range(length):
                step=start+offset+1
                action,own_obs,latent,old_logp,old_value=case_a_rollout_action(model,obs[NAMES[0]])
                actions={name:(0.,0.) for name in NAMES};actions[NAMES[0]]=action
                obs,reward,done,truncated,info=env.step(actions)
                assert set(info['physical_colliders']).issubset({NAMES[0]})
                assert not done or info['per_agent_success'][NAMES[0]] or NAMES[0] in info['physical_colliders']
                episode.add(reward,info,env)
                boundary=None
                if done:boundary=torch.zeros(1)
                elif truncated:
                    with torch.no_grad():boundary=model(torch.as_tensor(obs[NAMES[0]],dtype=torch.float32).unsqueeze(0))[1]
                records.append((0,old_logp.unsqueeze(0),old_value.unsqueeze(0),
                                torch.tensor([reward[NAMES[0]]],dtype=torch.float32),boundary))
                collected.append((own_obs,latent,old_logp,old_value))
                action_rows.append({'actor_mean':[[float((torch.tanh(model(own_obs.unsqueeze(0))[0][0,0])+1).detach()/2),
                                                  float(torch.tanh(model(own_obs.unsqueeze(0))[0][0,1]).detach())]],
                                    'sampled':[[action[0],action[1]]],
                                    'log_std':model.log_std.detach().tolist(),
                                    'raw_std':model.log_std.exp().detach().tolist()})
                if done or truncated:
                    outcome=episode.result(info)
                    _append(output/'episodes.jsonl',{'environment_steps':step,
                        'vessel_0_return':outcome['per_agent_return'][NAMES[0]],
                        'vessel_0_discounted_return':outcome['per_agent_discounted_return'][NAMES[0]],
                        'vessel_0_success':info['per_agent_success'][NAMES[0]],
                        'vessel_0_collision':NAMES[0] in info['physical_colliders'],
                        'episode_length':info['steps']})
                    episode_index+=1;obs=env.reset(scenario('A',1100000+episode_index))
                    episode=EpisodeDiagnostics(config.gamma)
            with torch.no_grad():
                bootstrap=model(torch.as_tensor(obs[NAMES[0]],dtype=torch.float32).unsqueeze(0))[1]
            targets=torch.stack(_return_targets(records,{0:bootstrap},config.gamma)).squeeze(-1).detach()
            old_values=torch.stack([item[3] for item in collected]).detach()
            advantages=(targets-old_values).detach()
            observations=torch.stack([item[0] for item in collected]).detach()
            latents=torch.stack([item[1] for item in collected]).detach()
            old_logp=torch.stack([item[2] for item in collected]).detach()
            assert observations.shape==(length,47)
            assert latents.shape==(length,2)
            assert targets.shape==advantages.shape==old_logp.shape==(length,)
            rollouts+=1
            diagnostics=ppo_case_a_update(model,optimizer,observations,latents,old_logp,
                targets,advantages,epochs=args.optimization_epochs,epsilon=args.ppo_clip_epsilon)
            assert len(diagnostics)==25==args.optimization_epochs
            for diagnostic in diagnostics:
                actor_passes+=1;critic_passes+=1;last_diagnostic=diagnostic
                _append(output/'updates.jsonl',{'environment_steps':start+length,
                    'collected_rollouts':rollouts,'actor_optimizer_passes':actor_passes,
                    'critic_optimizer_passes':critic_passes,'agent_transitions_per_environment_step':1,
                    'rollout_records':length,'episode_boundaries':sum(r[4] is not None for r in records),
                    'mean_advantage':float(advantages.mean()),'std_advantage':float(advantages.std(unbiased=False)),
                    **diagnostic,**action_summary(action_rows)})
            if start+length in (1000,2500,args.steps):check(start+length)
        assert rollouts==20==args.steps//args.rollout_horizon
        assert actor_passes==critic_passes==rollouts*args.optimization_epochs==500
        (output/'sanity.json').write_text(json.dumps({
            'environment_steps':args.steps,'rollout_horizon':args.rollout_horizon,
            'collected_rollouts':rollouts,'optimization_epochs_per_rollout':args.optimization_epochs,
            'actor_optimizer_passes':actor_passes,'critic_optimizer_passes':critic_passes,
            'sanity_assertions_passed':True},indent=2)+'\n')
    finally:env.close();eval_env.close()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',nargs='?',choices=['train','evaluate']);p.add_argument('--sim',default='pyquaticus')
    p.add_argument('--pyquaticus-python',default='.venv-pyquaticus/bin/python');p.add_argument('--kind',default='F',choices=list('ABCDEF'))
    p.add_argument('--output');p.add_argument('--seed',type=int,default=11);p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--update-interval',type=int,default=250)
    p.add_argument('--diagnostic',choices=['update-interval-sweep','case-a-rollout-ppo'])
    p.add_argument('--rollout-horizon',type=int,default=250)
    p.add_argument('--optimization-epochs',type=int,default=25)
    p.add_argument('--ppo-clip-epsilon',type=float,default=.2)
    p.add_argument('--checkpoint');p.add_argument('--count',type=int,default=100)
    p.add_argument('--eval-count',type=int,default=3)
    p.add_argument('--update-mode',choices=['global','separate','normalized'],default='global')
    p.add_argument('--surge-only',action='store_true')
    p.add_argument('--initial-yaw-raw-std',type=float,
                   help='Case A full-action diagnostic initial latent yaw standard deviation')
    p.add_argument('--actor-only',action='store_true')
    p.add_argument('--stochastic',action='store_true');p.add_argument('--controller',choices=['observation','truth'])
    args=p.parse_args();torch.set_num_threads(1)
    if args.diagnostic:
        if args.mode is not None:
            p.error('Do not combine a diagnostic with train/evaluate mode')
        if args.diagnostic=='update-interval-sweep':
            return run_update_interval_sweep(total_steps=args.steps,seed=args.seed,
                output_path=args.output or 'artifacts/shared-rl-diagnosis/update-interval-sweep.json')
        if args.diagnostic=='case-a-rollout-ppo':
            if (args.sim!='bcod-reduced' or args.kind!='A' or args.output is None or
                    args.initial_yaw_raw_std is None or args.surge_only or args.actor_only or
                    args.update_mode!='global' or args.steps!=5000 or
                    args.rollout_horizon!=250 or args.optimization_epochs!=25 or
                    args.ppo_clip_epsilon!=.2 or args.eval_count!=3 or
                    not math.isfinite(args.initial_yaw_raw_std) or
                    args.initial_yaw_raw_std<=0):
                p.error('case-a-rollout-ppo requires BCOD-reduced full-action Case A, '
                        '5,000 steps, 250-step rollouts, 25 epochs, epsilon 0.2, '
                        '3 evaluation scenarios, and a positive initial yaw raw std')
            return train_case_a_rollout_ppo(args)
    if args.mode is None or args.output is None:
        p.error('train/evaluate require a mode and --output')
    if args.steps<1 or args.update_interval<1:
        p.error('--steps and --update-interval must be positive')
    if args.actor_only and (not args.surge_only or args.kind!='A'):
        p.error('--actor-only requires --surge-only --kind A')
    if args.initial_yaw_raw_std is not None and (
            args.mode!='train' or args.kind!='A' or args.surge_only or
            not math.isfinite(args.initial_yaw_raw_std) or args.initial_yaw_raw_std<=0):
        p.error('--initial-yaw-raw-std requires full-action Case A training and a positive finite value')
    if args.mode=='train':return train(args)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False);model=initial(args.seed);step=0
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);step=saved['environment_steps']
    env=Benchmark(make_adapter(args.sim,BenchmarkConfig(),pyquaticus_python=args.pyquaticus_python),BenchmarkConfig(),diagnostic_allow_sparse=True)
    try:result=evaluate(model,env,args.kind,args.count,not args.stochastic,output/'episodes.jsonl',step,args.controller,args.surge_only)
    finally:env.close()
    (output/'summary.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
