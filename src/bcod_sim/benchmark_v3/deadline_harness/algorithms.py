"""Small self-contained MAPPO and TD3 learners for the deadline tasks."""
from __future__ import annotations

from collections import deque
import random
import math
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch import nn


def mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers.extend((nn.Linear(a, b), nn.ReLU()))
    return nn.Sequential(*layers[:-1])


class MAPPO(nn.Module):
    def __init__(self, obs_dim=36, state_dim=89, action_dim=2, initial_action_std=.10):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim,128), nn.ReLU(), nn.Linear(128,128),
                                   nn.ReLU(), nn.Linear(128,action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(initial_action_std)))
        self.critic = nn.Sequential(nn.Linear(state_dim,256), nn.ReLU(), nn.Linear(256,256),
                                    nn.ReLU(), nn.Linear(256,1))
        self.register_buffer("obs_mean",torch.zeros(obs_dim))
        self.register_buffer("obs_std",torch.ones(obs_dim))
        self.register_buffer("value_mean",torch.zeros(()))
        self.register_buffer("value_std",torch.ones(()))

    def set_observation_normalizer(self, observations):
        x=torch.as_tensor(observations,dtype=self.obs_mean.dtype,device=self.obs_mean.device)
        self.obs_mean.copy_(x.mean(0)); self.obs_std.copy_(x.std(0,unbiased=False).clamp_min(.05))

    def normalize_observation(self, obs):
        return ((obs-self.obs_mean)/self.obs_std).clamp(-10.,10.)

    def mean_action(self, obs):
        return self.actor(self.normalize_observation(obs))

    def act(self, obs, deterministic=False):
        mean = self.mean_action(obs)
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        raw = mean if deterministic else dist.rsample()
        action = torch.stack((torch.sigmoid(raw[..., 0]), torch.tanh(raw[..., 1])), -1)
        logp = dist.log_prob(raw).sum(-1)
        return action, logp, raw

    def value(self, state):
        return self.critic(state).squeeze(-1)*self.value_std+self.value_mean


class TD3Actor(nn.Module):
    def __init__(self, obs_dim=10, action_dim=2):
        super().__init__(); self.register_buffer("obs_mean",torch.zeros(obs_dim)); self.register_buffer("obs_std",torch.ones(obs_dim))
        self.net = nn.Sequential(nn.Linear(obs_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(), nn.Linear(256,action_dim), nn.Tanh())
    def set_observation_normalizer(self, observations):
        x=torch.as_tensor(observations,dtype=self.obs_mean.dtype,device=self.obs_mean.device)
        self.obs_mean.copy_(x.mean(0)); self.obs_std.copy_(x.std(0,unbiased=False).clamp_min(.05))
    def forward(self, obs): return self.net(((obs-self.obs_mean)/self.obs_std).clamp(-10.,10.))


class TwinCritic(nn.Module):
    def __init__(self, obs_dim=10, action_dim=2):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim+action_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(), nn.Linear(256,1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim+action_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(), nn.Linear(256,1))
    def forward(self, obs, act):
        x = torch.cat((obs,act),-1); return self.q1(x), self.q2(x)


class Replay:
    def __init__(self, capacity=1_000_000): self.rows=deque(maxlen=capacity)
    def add(self,*row): self.rows.append(tuple(np.asarray(x).copy() if hasattr(x,"shape") else x for x in row))
    def sample(self,batch,device="cpu"):
        rows=random.sample(self.rows,batch)
        return [torch.as_tensor(np.stack([r[i] for r in rows]),device=device,dtype=torch.float32)
                for i in range(5)]


def clone_actor(model, observations, actions, *, epochs=20, batch_size=256, lr=1e-3,
                marl=False, seed=11):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    x=torch.as_tensor(observations,dtype=torch.float32); y=torch.as_tensor(actions,dtype=torch.float32)
    model.set_observation_normalizer(x)
    optimizer=torch.optim.Adam(model.actor.parameters() if marl else model.parameters(),lr=lr)
    losses=[]
    for _ in range(epochs):
        epoch=[]
        for idx in np.array_split(rng.permutation(len(x)),max(1,(len(x)+batch_size-1)//batch_size)):
            pred=model.mean_action(x[idx]) if marl else model(x[idx])
            if marl: pred=torch.stack((torch.sigmoid(pred[:,0]),torch.tanh(pred[:,1])),-1)
            loss=(pred-y[idx]).square().mean(); optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); epoch.append(loss.item())
        losses.append(float(np.mean(epoch)))
    return losses


def action_error_metrics(predicted, target, dimension_names):
    error=np.asarray(predicted,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    absolute=np.abs(error)
    if error.ndim!=2 or error.shape[1]!=len(dimension_names):
        raise ValueError("Action metric dimensions disagree")
    return {"sample_count":len(error),"normalized_mae":float(absolute.mean()),
        "dimension_mae":{name:float(absolute[:,i].mean()) for i,name in enumerate(dimension_names)},
        "dimension_max_error":{name:float(absolute[:,i].max()) for i,name in enumerate(dimension_names)},
        "dimension_quantiles":{name:{str(q):float(np.quantile(absolute[:,i],q))
            for q in (.5,.9,.95,.99)} for i,name in enumerate(dimension_names)}}


def train_td3(env, *, steps=400_000, seed=11, warmup=0, batch=256,
              output="sarl-td3.pt", device="cpu", initial_actor=None,
              initial_replay=None, validate=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    started=time.perf_counter()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    actor=TD3Actor().to(device); critic=TwinCritic().to(device)
    if initial_actor is not None: actor.load_state_dict(initial_actor)
    at=TD3Actor().to(device); ct=TwinCritic().to(device); at.load_state_dict(actor.state_dict()); ct.load_state_dict(critic.state_dict())
    ao=torch.optim.Adam(actor.parameters(),lr=3e-4); co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    replay=Replay(); obs,_=env.reset(seed=seed); episode=0; updates=0
    if initial_replay is not None:
        for row in zip(*initial_replay): replay.add(*row)
    torch.save({"actor":actor.state_dict(),"critic":critic.state_dict(),"steps":0,"seed":seed},output+".step-0.pt")
    rewards=[]; actor_losses=[]; critic_losses=[]; q_means=[]; saturation=[]; best_validation=None
    for t in range(steps):
        frac=min(1.,t/max(steps,1)); noise=.10+frac*(.03-.10)
        with torch.no_grad():
            action=actor(torch.as_tensor(obs,device=device).float()).cpu().numpy()
        if t<warmup: action=np.random.uniform(-1,1,2).astype(np.float32)
        else: action=np.clip(action+np.random.normal(0,noise,2),-1,1).astype(np.float32)
        nxt,r,term,trunc,info=env.step(action); replay.add(obs,action,r,nxt,float(term))
        rewards.append(float(r)); saturation.extend(info.get("saturation",{}).values())
        obs=nxt
        if term or trunc: episode+=1; obs,_=env.reset(seed=seed+episode)
        if len(replay.rows)>=batch:
            s,a,rw,ns,d=replay.sample(batch,device)
            with torch.no_grad():
                na=(at(ns)+torch.randn_like(a).mul(.1).clamp(-.2,.2)).clamp(-1,1)
                q1t,q2t=ct(ns,na); target=rw[:,None]+.995*(1-d[:,None])*torch.minimum(q1t,q2t)
            q1,q2=critic(s,a); loss=(q1-target).square().mean()+(q2-target).square().mean()
            if not all(torch.isfinite(x).all() for x in (loss,q1,q2,target)) or max(float(q1.detach().abs().max()),float(q2.detach().abs().max()))>1e6:
                raise FloatingPointError("Nonfinite or exploding TD3 critic values")
            co.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),1.); co.step()
            critic_losses.append(float(loss.detach())); q_means.append(float(torch.minimum(q1,q2).detach().mean()))
            updates+=1
            if updates%2==0:
                aloss=-critic.q1(torch.cat((s,actor(s)),-1)).mean()
                if not torch.isfinite(aloss): raise FloatingPointError("Nonfinite TD3 actor loss")
                ao.zero_grad(); aloss.backward(); nn.utils.clip_grad_norm_(actor.parameters(),1.); ao.step()
                actor_losses.append(float(aloss.detach()))
                with torch.no_grad():
                    for p,tp in zip(actor.parameters(),at.parameters()): tp.mul_(.995).add_(p,alpha=.005)
                    for p,tp in zip(critic.parameters(),ct.parameters()): tp.mul_(.995).add_(p,alpha=.005)
        if (t+1)%25_000==0 or t+1==steps:
            payload={"actor":actor.state_dict(),"critic":critic.state_dict(),"steps":t+1,"seed":seed,
                "replay_size":len(replay.rows)}
            torch.save(payload,output); torch.save(payload,output+f".step-{t+1}.pt")
            if validate is not None:
                result=validate(actor,t+1)
                with open(output+".validation.jsonl","a") as stream:
                    stream.write(json.dumps({"step":t+1,"validation":result})+"\n")
                score=(-result.get("speed_rmse_mps",float("inf")),
                       -result.get("heading_rmse_deg",float("inf")),
                       -result.get("saturation_fraction",1.))
                if best_validation is None or score>best_validation:
                    best_validation=score
                    torch.save(payload,output+".best-tracking.pt")
            with open(output+".metrics.jsonl","a") as stream:
                stream.write(json.dumps({"step":t+1,"elapsed_s":time.perf_counter()-started,
                    "steps_per_s":(t+1)/(time.perf_counter()-started),
                    "actor_loss":float(np.mean(actor_losses)) if actor_losses else None,
                    "critic_loss":float(np.mean(critic_losses)) if critic_losses else None,
                    "q_mean":float(np.mean(q_means)) if q_means else None,
                    "return_mean":float(np.mean(rewards[-25_000:])),
                    "replay_size":len(replay.rows),"pid_replay_transitions":len(initial_replay[0]) if initial_replay is not None else 0,
                    "saturation_fraction":float(np.mean(saturation[-50_000:])) if saturation else 0.})+"\n")
    return actor,replay


def train_mappo(env, *, steps=400_000, seed=11, output="marl-mappo.pt",
                device="cpu", initial_state=None, validate=None,
                anchor_observations=None, anchor_actions=None, anchor_weight=0.):
    """Shared-actor centralized-critic PPO. Steps count complete fleet ticks."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    started=time.perf_counter()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    model=MAPPO().to(device)
    if initial_state is not None: model.load_state_dict(initial_state)
    torch.save({"model":model.state_dict(),"steps":0,"seed":seed},output+".step-0.pt")
    actor_opt=torch.optim.Adam(list(model.actor.parameters())+[model.log_std],lr=1e-4)
    critic_opt=torch.optim.Adam(model.critic.parameters(),lr=3e-4)
    if anchor_weight < 0 or (anchor_observations is None) != (anchor_actions is None):
        raise ValueError("BC anchor requires paired observations/actions and nonnegative weight")
    anchor_x=(torch.as_tensor(anchor_observations,dtype=torch.float32,device=device)
              if anchor_observations is not None else None)
    anchor_y=(torch.as_tensor(anchor_actions,dtype=torch.float32,device=device)
              if anchor_actions is not None else None)
    if anchor_weight and (anchor_x is None or anchor_x.ndim!=2 or anchor_y.shape!=(len(anchor_x),2)):
        raise ValueError("BC anchor has invalid observation/action shape")
    obs,_=env.reset(seed=seed); names=list(env.possible_agents)
    horizon=128; gamma=.995; lam=.95; best=None; best_collision=None
    if validate is not None:
        baseline=validate(model,0)
        best=(baseline.get("success_rate",0.),-baseline.get("collision_rate",1.),
              -baseline.get("completion_time_s",float("inf")))
        best_collision=(-baseline.get("collision_rate",1.),baseline.get("success_rate",0.),
                        -baseline.get("completion_time_s",float("inf")))
        record={"step":0,"validation":baseline}
        with open(output+".validation.jsonl","a") as stream: stream.write(json.dumps(record)+"\n")
        payload={"model":model.state_dict(),"steps":0,"validation":baseline}
        torch.save(payload,output+".best-success.pt")
        torch.save(payload,output+".best-low-collision.pt")
    total_steps=(steps//horizon)*horizon
    for update in range(total_steps//horizon):
        ro=[]; rs=[]; raws=[]; logps=[]; vals=[]; nextvals=[]; states=[]; dones=[]; truncs=[]; action_rows=[]
        component_rows=[]
        for _ in range(horizon):
            local=torch.as_tensor(np.stack([obs[n] for n in names]),device=device)
            state=torch.as_tensor(env.state(),device=device)
            with torch.no_grad():
                action,logp,raw=model.act(local)
                value=model.value(state)
            action=action.cpu().numpy(); actions={n:action[i] for i,n in enumerate(names)}
            action_rows.extend(action)
            nxt,rewards,terms,truncations,infos=env.step(actions)
            with torch.no_grad():
                nextvals.append(float(model.value(torch.as_tensor(env.state(),device=device))))
            ro.append(local.cpu().numpy()); states.append(state.cpu().numpy())
            raws.append(raw.cpu().numpy()); logps.append(logp.cpu().numpy()); vals.append(float(value))
            rs.append(float(rewards[names[0]])); done=all(terms.values()) or all(truncations.values())
            component_rows.append(dict(infos[names[0]].get("shared_fleet_reward_components",{})))
            dones.append(float(all(terms.values()))); truncs.append(done)
            obs=nxt
            if done: obs,_=env.reset(seed=seed+update*horizon+_+1)
        adv=np.zeros(horizon,np.float32); gae=0.
        for i in reversed(range(horizon)):
            next_value=nextvals[i]
            continuation=0. if truncs[i] else 1.
            delta=rs[i]+gamma*next_value*(1.-dones[i])-vals[i]
            gae=delta+gamma*lam*continuation*gae; adv[i]=gae
        targets=adv+np.asarray(vals,np.float32); adv=(adv-adv.mean())/(adv.std()+1e-8)
        target_mean=float(targets.mean()); target_std=max(float(targets.std()),.1)
        model.value_mean.fill_(target_mean); model.value_std.fill_(target_std)
        flat_obs=torch.as_tensor(np.concatenate(ro),device=device)
        flat_raw=torch.as_tensor(np.concatenate(raws),device=device)
        oldlp=torch.as_tensor(np.concatenate(logps),device=device)
        flat_adv=torch.as_tensor(np.repeat(adv, len(names)),device=device)
        flat_states=torch.as_tensor(np.repeat(np.stack(states),len(names),axis=0),device=device)
        flat_targets=torch.as_tensor(np.repeat(targets,len(names)),device=device)
        n=len(flat_obs); ids=np.arange(n)
        epoch_kl=[]; policy_losses=[]; value_losses=[]; entropies=[]; grad_norms=[]; anchor_losses=[]
        entropy_coef=.002+(.0005-.002)*min(1.,update/max(total_steps//horizon-1,1))
        for _epoch in range(5):
            epoch_kls=[]
            np.random.shuffle(ids)
            for batch_ids in np.array_split(ids,max(1,math.ceil(n/512))):
                ix=torch.as_tensor(batch_ids,device=device)
                mean=model.mean_action(flat_obs[ix]); dist=torch.distributions.Normal(mean,model.log_std.exp().expand_as(mean))
                lp=dist.log_prob(flat_raw[ix]).sum(-1); logratio=lp-oldlp[ix]; ratio=logratio.exp()
                a=flat_adv[ix]; entropy=dist.entropy().sum(-1).mean()
                loss_actor=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()-entropy_coef*entropy
                if anchor_weight:
                    anchor_ix=torch.randint(len(anchor_x),(len(ix),),device=device)
                    anchor_raw=model.mean_action(anchor_x[anchor_ix])
                    anchor_pred=torch.stack((torch.sigmoid(anchor_raw[:,0]),torch.tanh(anchor_raw[:,1])),-1)
                    anchor_loss=(anchor_pred-anchor_y[anchor_ix]).square().mean()
                    loss_actor=loss_actor+anchor_weight*anchor_loss
                    anchor_losses.append(float(anchor_loss.detach()))
                normalized_targets=(flat_targets[ix]-model.value_mean)/model.value_std
                normalized_values=model.critic(flat_states[ix]).squeeze(-1)
                loss_value=(normalized_values-normalized_targets).square().mean()
                actor_opt.zero_grad(); loss_actor.backward(); actor_norm=nn.utils.clip_grad_norm_(list(model.actor.parameters())+[model.log_std],.5); actor_opt.step()
                model.log_std.data.clamp_(math.log(.03),math.log(.3))
                critic_opt.zero_grad(); loss_value.backward(); critic_norm=nn.utils.clip_grad_norm_(model.critic.parameters(),.5); critic_opt.step()
                approx_kl=((ratio-1.)-logratio).mean()
                if not all(torch.isfinite(x).all() for x in (loss_actor,loss_value,approx_kl,model.log_std)):
                    raise FloatingPointError("Nonfinite MAPPO loss, KL, or action standard deviation")
                epoch_kls.append(float(approx_kl.detach())); policy_losses.append(float(loss_actor.detach()))
                value_losses.append(float(loss_value.detach())); entropies.append(float(entropy.detach()))
                grad_norms.append(max(float(actor_norm),float(critic_norm)))
            if not all(np.isfinite(row).all() for row in ro) or not np.isfinite(rs).all():
                raise FloatingPointError("Nonfinite MAPPO rollout observation or reward")
            epoch_kl.append(float(np.mean(epoch_kls)))
            if epoch_kl[-1]>.035: break
        step=(update+1)*horizon
        target_np=np.repeat(targets,len(names)); value_np=np.repeat(vals,len(names))
        explained=1.-float(np.var(target_np-value_np)/(np.var(target_np)+1e-8))
        action_array=np.asarray(action_rows)
        logs={"step":step,"elapsed_s":time.perf_counter()-started,
            "steps_per_s":step/(time.perf_counter()-started),"policy_loss":float(np.mean(policy_losses)),
            "value_loss":float(np.mean(value_losses)),"entropy":float(np.mean(entropies)),
            "entropy_coef":entropy_coef,"action_std":model.log_std.detach().exp().cpu().tolist(),
            "bc_anchor_weight":anchor_weight,
            "bc_anchor_loss":float(np.mean(anchor_losses)) if anchor_losses else None,
            "approx_kl_by_epoch":epoch_kl,"explained_variance":explained,
            "gradient_norm_max":float(np.max(grad_norms)),"return_mean":float(np.mean(rs)),
            "zero_speed_action_fraction":float(np.mean(action_array[:,0]<.01)),
            "saturated_action_fraction":float(np.mean(np.max(np.abs(action_array),axis=1)>.98)),
            "reward_components":{key:float(np.mean([row.get(key,0.) for row in component_rows]))
                                 for key in sorted({k for row in component_rows for k in row})}}
        with open(output+".metrics.jsonl","a") as stream: stream.write(__import__("json").dumps(logs)+"\n")
        if step//25_000>(step-horizon)//25_000 or step>=total_steps:
            payload={"model":model.state_dict(),"steps":step,"seed":seed}
            torch.save(payload,output); torch.save(payload,output+f".step-{step}.pt")
            if validate is not None:
                result=validate(model,step)
                score=(result.get("success_rate",0.),-result.get("collision_rate",1.),
                       -result.get("completion_time_s",float("inf"))) if isinstance(result,dict) else result
                record={"step":step,"validation":result}
                with open(output+".validation.jsonl","a") as stream: stream.write(json.dumps(record)+"\n")
                if best is None or score>best:
                    best=score; torch.save({"model":model.state_dict(),"steps":step,"validation":result},output+".best-success.pt")
                collision_score=(-result.get("collision_rate",1.),result.get("success_rate",0.),-result.get("completion_time_s",float("inf")))
                if best_collision is None or collision_score>best_collision:
                    best_collision=collision_score
                    torch.save({"model":model.state_dict(),"steps":step,"validation":result},output+".best-low-collision.pt")
    return model
