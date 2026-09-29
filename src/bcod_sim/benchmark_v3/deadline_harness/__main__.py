"""Command line for teacher datasets and guarded deadline training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .algorithms import MAPPO, TD3Actor, action_error_metrics, clone_actor, train_mappo, train_td3
from .data import collect_marl_teacher, collect_sarl_pid, load_dataset
from .marl import GeometricCoordinator, ResidualCoordinatorEnv, make_four_vessel_scenario
from .sarl import PIDThrustTeacher, SARLTrackingEnv
from .evaluation import evaluate_sarl, evaluate_marl
from ..config import TaskConfig


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def finite_training_logs(rows, policy):
    for row in rows:
        if policy=="marl":
            values=[row[k] for k in ("policy_loss","value_loss","entropy","entropy_coef",
                "gradient_norm_max","return_mean","explained_variance","zero_speed_action_fraction",
                "saturated_action_fraction")]+row["action_std"]+row["approx_kl_by_epoch"]+list(row["reward_components"].values())
        else:
            values=[row[k] for k in ("return_mean","saturation_fraction")]
            values += [row[k] for k in ("actor_loss","critic_loss","q_mean") if row[k] is not None]
        if not np.isfinite(np.asarray(values,dtype=float)).all(): return False
    return bool(rows)


def marl_validation(model, step, episodes=32, seed=81001):
    device=next(model.parameters()).device
    """Frozen small validation mix; uses deterministic actor means."""
    rng=np.random.default_rng(seed); rows=[]; family_rows={}
    for i in range(episodes):
        family=("easy","pair","fourway","random")[i%4]
        scenario=make_four_vessel_scenario(rng,family,split="marl-validation")
        env=ResidualCoordinatorEnv(scenario=scenario); obs,_=env.reset(seed=scenario.seed)
        for tick in range(env.config.deadline_steps):
            names=list(env.possible_agents)
            with torch.no_grad():
                actions_np=model.act(torch.as_tensor(np.stack([obs[n] for n in names]),device=device),deterministic=True)[0].cpu().numpy()
            actions={n:actions_np[j].astype(np.float32) for j,n in enumerate(names)}
            obs,_,terms,truncs,infos=env.step(actions)
            if all(terms.values()) or all(truncs.values()): break
        reason=infos[env.possible_agents[0]]["terminal_reason"]
        rows.append((reason, tick*env.config.dt_s)); family_rows.setdefault(family,[]).append(reason); env.close()
    return {"step":step,"episodes":len(rows),"success_rate":sum(x[0]=="success" for x in rows)/len(rows),
        "collision_rate":sum(x[0]=="collision" for x in rows)/len(rows),
        "timeout_rate":sum(x[0]=="deadline" for x in rows)/len(rows),
        "completion_time_s":float(np.mean([x[1] for x in rows])),
        "by_family":{name:{"episodes":len(reasons),"success_rate":reasons.count("success")/len(reasons),
            "collision_rate":reasons.count("collision")/len(reasons),"timeout_rate":reasons.count("deadline")/len(reasons)}
            for name,reasons in family_rows.items()}}


def sarl_validation(actor, step, episodes=12, seed=82001, steps=300):
    device=next(actor.parameters()).device
    """Nominal held-out command tracking score for TD3 checkpoint selection."""
    from ..control.high_level import wrap_angle
    speed_rmse=[]; heading_rmse=[]; saturation=[]; actions=[]
    for i in range(episodes):
        env=SARLTrackingEnv(episode_steps=steps); obs,_=env.reset(seed=seed+i,command_family="combined")
        env._maybe_new_command=lambda:None
        if i%2: env.desired_heading=wrap_angle(env.desired_heading+np.deg2rad(45.))
        se=[]; he=[]; sat=[]
        for _ in range(steps):
            with torch.no_grad(): action=actor(torch.as_tensor(obs,device=device).float()).cpu().numpy()
            actions.append(action)
            obs,_,term,trunc,info=env.step(action); se.append(info["speed_error"]); he.append(info["heading_error"])
            sat.extend(info["saturation"].values())
            if term or trunc: break
        speed_rmse.append(float(np.sqrt(np.mean(np.square(se[-150:])))))
        heading_rmse.append(float(np.degrees(np.sqrt(np.mean(np.square(he[-150:]))))))
        saturation.extend(sat); env.close()
    return {"step":step,"episodes":episodes,"speed_rmse_mps":float(np.mean(speed_rmse)),
        "heading_rmse_deg":float(np.mean(heading_rmse)),"saturation_fraction":float(np.mean(saturation)),
        "action_std":np.asarray(actions).std(axis=0).tolist()}


def preflight(output, seed=11, easy_cases=50, marl_bc=None, sarl_bc=None, policy="both"):
    rng=np.random.default_rng(seed); successes=collisions=timeouts=0; easy_scenarios=[]
    if policy in ("both","marl"):
        for case in range(easy_cases):
            scenario=make_four_vessel_scenario(rng,"easy",split="preflight")
            easy_scenarios.append(scenario)
            env=ResidualCoordinatorEnv(scenario=scenario); obs,_=env.reset(seed=scenario.seed)
            for _ in range(env.config.deadline_steps):
                obs,_,terms,truncs,infos=env.step(GeometricCoordinator().actions(env))
                if all(terms.values()) or all(truncs.values()): break
            reason=infos[env.possible_agents[0]]["terminal_reason"]
            successes+=reason=="success"; collisions+=reason=="collision"; timeouts+=reason=="deadline"
            env.close()
    # The plant sign check uses a fresh direct-thrust episode.
    sarl=SARLTrackingEnv(episode_steps=2); sarl.reset(seed=seed)
    sarl.step(np.array([0.,.5],np.float32)); positive_turn=sarl.reading.yaw_rps>0; sarl.close()
    reward_scores={"teacher":[],"nominal":[],"degenerate":[]}
    if policy in ("both","marl"):
        rr=np.random.default_rng(seed+1000)
        reward_config=TaskConfig(agent_count=4,dynamics="bcod-reduced",action_mode="high_level",
                                 dt_s=.5,deadline_steps=120)
        for family in ("pair","corner","fourway","random"):
            scenario=make_four_vessel_scenario(rr,family,split="preflight-reward")
            for reward_policy in reward_scores:
                env=ResidualCoordinatorEnv(config=reward_config,scenario=scenario); obs,_=env.reset(seed=scenario.seed); total=0.
                for _ in range(env.config.deadline_steps):
                    if reward_policy=="teacher": actions=GeometricCoordinator().actions(env)
                    elif reward_policy=="nominal": actions={n:np.array([1.,0.],np.float32) for n in env.possible_agents}
                    else: actions={n:np.array([0.,0.],np.float32) for n in env.possible_agents}
                    obs,rewards,terms,truncs,_=env.step(actions); total+=rewards[env.possible_agents[0]]
                    if all(terms.values()) or all(truncs.values()): break
                reward_scores[reward_policy].append(total); env.close()
    means={name:(float(np.mean(scores)) if scores else None) for name,scores in reward_scores.items()}
    ordering=(means["teacher"]>means["nominal"]>means["degenerate"]
              if policy in ("both","marl") else True)
    import math
    from ..control.high_level import wrap_angle
    pid_metrics={}
    cases=[("speed_step",0.,1.5),*(('heading_'+str(angle),angle,None) for angle in (15,45,90)),
           ("combined_90",90.,1.5)]
    for index,(family,angle,speed) in enumerate(cases):
        env=SARLTrackingEnv(episode_steps=300); obs,_=env.reset(seed=seed+7+index)
        env._maybe_new_command=lambda:None
        if speed is not None: env.desired_speed=speed
        env.desired_heading=wrap_angle(env.reading.heading_rad+math.radians(angle))
        teacher=PIDThrustTeacher(); speed_errors=[]; heading_errors=[]; actions=[]
        for _ in range(300):
            action,_=teacher.predict(obs,env=env); obs,_,term,trunc,info=env.step(action)
            speed_errors.append(info["command"][0]-info["actual_speed"])
            heading_errors.append(info["heading_error"]); actions.append(action)
            if term or trunc: break
        pid_metrics[family]={"settled_speed_rmse_mps":float(np.sqrt(np.mean(np.square(speed_errors[-150:])))),
            "settled_heading_rmse_deg":float(np.degrees(np.sqrt(np.mean(np.square(heading_errors[-150:]))))),
            "finite":bool(np.isfinite(actions).all()),
            "commands_achievable":bool(0<=env.desired_speed<=2 and np.isfinite(env.desired_heading))}
        env.close()
    pid_ok=all(m["finite"] and m["commands_achievable"] and
               m["settled_speed_rmse_mps"]<.5 and m["settled_heading_rmse_deg"]<5 for m in pid_metrics.values())
    marl_bc_ok=sarl_bc_ok=False
    if policy in ("both","marl") and marl_bc and Path(marl_bc).is_file():
        payload=torch.load(marl_bc,map_location="cpu",weights_only=True)
        x,y,_=load_dataset(payload["dataset_path"])
        with np.load(payload["dataset_path"],allow_pickle=False) as ds: ids=ds["scenario_ids"]
        heldout=set(np.unique(ids)[-max(1,len(np.unique(ids))//5):].tolist())
        test_mask=np.asarray([int(i) in heldout for i in ids])
        with torch.no_grad():
            model=MAPPO(); model.load_state_dict(payload["model"])
            raw=model.mean_action(torch.as_tensor(x[test_mask])).numpy()
            pred=np.stack((1/(1+np.exp(-raw[:,0])),np.tanh(raw[:,1])),-1)
        imitation=action_error_metrics(pred,y[test_mask],("speed_multiplier","heading_residual"))
        class ClonedCoordinator:
            def actions(self, observations):
                names=list(observations)
                with torch.no_grad():
                    raw=model.mean_action(torch.as_tensor(np.stack([observations[n] for n in names]))).numpy()
                action=np.stack((1/(1+np.exp(-raw[:,0])),np.tanh(raw[:,1])),-1)
                return {n:action[i].astype(np.float32) for i,n in enumerate(names)}
        bc_success=0
        bc_collision=0
        for scenario in easy_scenarios:
            env=ResidualCoordinatorEnv(scenario=scenario); obs,_=env.reset(seed=scenario.seed)
            for _ in range(env.config.deadline_steps):
                obs,_,terms,truncs,infos=env.step(ClonedCoordinator().actions(obs))
                if all(terms.values()) or all(truncs.values()): break
            bc_success+=infos[env.possible_agents[0]]["terminal_reason"]=="success"; env.close()
            bc_collision+=infos[env.possible_agents[0]]["terminal_reason"]=="collision"
        conflict_results={"teacher_success":0,"teacher_collision":0,"bc_success":0,"bc_collision":0}
        families=("pair","corner","fourway","random")
        for family_index,family in enumerate(families):
            for case in range(5):
                scenario=make_four_vessel_scenario(np.random.default_rng(seed+50000+family_index*100+case),family,split="bc-conflict")
                for label,policy_impl in (("teacher",GeometricCoordinator()),("bc",ClonedCoordinator())):
                    env=ResidualCoordinatorEnv(scenario=scenario); obs,_=env.reset(seed=scenario.seed)
                    for _ in range(env.config.deadline_steps):
                        obs,_,terms,truncs,infos=env.step(policy_impl.actions(env) if label=="teacher" else policy_impl.actions(obs))
                        if all(terms.values()) or all(truncs.values()): break
                    reason=infos[env.possible_agents[0]]["terminal_reason"]
                    conflict_results[label+"_success"]+=reason=="success"
                    conflict_results[label+"_collision"]+=reason=="collision"
                    env.close()
        denom=20
        easy_success_rate=bc_success/easy_cases; easy_collision_rate=bc_collision/easy_cases
        marl_bc_ok=(imitation["normalized_mae"]<=.10 and easy_success_rate>=.90 and
            easy_collision_rate<=.05 and conflict_results["bc_success"]/denom+1e-9 >=
            conflict_results["teacher_success"]/denom-.10 and
            conflict_results["bc_collision"]/denom <= conflict_results["teacher_collision"]/denom+.10+1e-9)
        pid_metrics["marl_bc"]={"open_loop_heldout":imitation,
            "easy_success_rate":easy_success_rate,"easy_collision_rate":easy_collision_rate,
            "conflict_results":conflict_results}
    if policy in ("both","sarl") and sarl_bc and Path(sarl_bc).is_file():
        payload=torch.load(sarl_bc,map_location="cpu",weights_only=True)
        x,y,_=load_dataset(payload["dataset_path"])
        with np.load(payload["dataset_path"],allow_pickle=False) as ds: ids=ds["episode_ids"]
        heldout=set(np.unique(ids)[-max(1,len(np.unique(ids))//5):].tolist())
        test_mask=np.asarray([int(i) in heldout for i in ids])
        actor=TD3Actor(); actor.load_state_dict(payload["model"])
        with torch.no_grad(): pred=actor(torch.as_tensor(x[test_mask])).numpy()
        imitation=action_error_metrics(pred,y[test_mask],("common_thrust","differential_thrust"))
        suite=[]
        for index,(family,angle,speed) in enumerate(cases):
            paired={"pid":{},"bc":{}}
            for label in ("pid","bc"):
                env=SARLTrackingEnv(episode_steps=300); obs,_=env.reset(seed=seed+700+index)
                env._maybe_new_command=lambda:None
                if speed is not None: env.desired_speed=speed
                env.desired_heading=wrap_angle(env.reading.heading_rad+math.radians(angle))
                teacher=PIDThrustTeacher(); se=[]; he=[]; sat=[]
                for _ in range(300):
                    if label=="pid": action,_=teacher.predict(obs,env=env)
                    else:
                        with torch.no_grad(): action=actor(torch.as_tensor(obs).float()).cpu().numpy()
                    obs,_,term,trunc,info=env.step(action)
                    se.append(info["command"][0]-info["actual_speed"]); he.append(info["heading_error"])
                    sat.extend(info["saturation"].values())
                    if term or trunc: break
                paired[label]={"speed_rmse":float(np.sqrt(np.mean(np.square(se[-150:])))),
                    "heading_rmse_deg":float(np.degrees(np.sqrt(np.mean(np.square(he[-150:]))))),
                    "heading_mae_deg":float(np.degrees(np.mean(np.abs(he[-150:])))),
                    "saturation_fraction":float(np.mean(sat))}
                env.close()
            suite.append({"family":family,**paired})
        track_ok=all(row["bc"]["speed_rmse"]<=max(1.5*row["pid"]["speed_rmse"],.15) and
                     row["bc"]["heading_rmse_deg"]<=max(1.5*row["pid"]["heading_rmse_deg"],5.) and
                     row["bc"]["saturation_fraction"]<.95 for row in suite)
        sarl_bc_ok=imitation["normalized_mae"]<=.10 and track_ok
        pid_metrics["sarl_bc"]={"open_loop_heldout":imitation,"paired_tracking":suite}
    ready=(bool(positive_turn) and ordering and pid_ok and
           ((policy in ("both","marl") and successes/easy_cases>=.95 and marl_bc_ok) if policy=="marl" else True) and
           ((policy in ("both","sarl") and sarl_bc_ok) if policy=="sarl" else True) and
           (marl_bc_ok and sarl_bc_ok if policy=="both" else True))
    report={"seed":seed,"teacher_easy_success_rate":successes/easy_cases if policy in ("both","marl") else None,
            "teacher_easy_success_count":successes,"easy_cases":easy_cases,
            "teacher_collision_count":collisions,"teacher_timeout_count":timeouts,
            "positive_differential_positive_yaw":bool(positive_turn),
            "reward_order_means":means,"teacher_reward_ordering":bool(ordering),
            "pid_tracking_metrics":pid_metrics,
            "marl_bc_reproduction":bool(marl_bc_ok),"sarl_bc_reproduction":bool(sarl_bc_ok),
            "sarl_command_family_tracking":bool(pid_ok),
            "production_ready":bool(ready)}
    write(output,report); return report


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("preflight"); c.add_argument("--output",default="runs/deadline/preflight.json"); c.add_argument("--seed",type=int,default=11); c.add_argument("--policy",choices=("both","marl","sarl"),default="both")
    c.add_argument("--easy-cases",type=int,default=50)
    c.add_argument("--marl-bc"); c.add_argument("--sarl-bc")
    for policy in ("marl","sarl"):
        c=sub.add_parser(policy+"-preflight"); c.add_argument("--output",default=f"runs/deadline/{policy}-preflight.json"); c.add_argument("--seed",type=int,default=11); c.add_argument("--bc",default=f"runs/deadline/{policy}-bc.pt")
    c=sub.add_parser("marl-eval"); c.add_argument("--device",default="cpu"); c.add_argument("--checkpoint",default="runs/deadline/marl-mappo.pt.best-success.pt"); c.add_argument("--output",default="runs/deadline/marl-eval"); c.add_argument("--per-family",type=int,default=50); c.add_argument("--seed",type=int,default=62001)
    c=sub.add_parser("sarl-eval"); c.add_argument("--device",default="cpu"); c.add_argument("--checkpoint",default="runs/deadline/sarl-td3.pt.best-tracking.pt"); c.add_argument("--output",default="runs/deadline/sarl-eval"); c.add_argument("--episodes",type=int,default=60); c.add_argument("--steps",type=int,default=500); c.add_argument("--seed",type=int,default=44001)
    for policy in ("marl","sarl"):
        c=sub.add_parser(policy+"-smoke"); c.add_argument("--preflight",default=f"runs/deadline/{policy}-preflight.json"); c.add_argument("--bc",default=f"runs/deadline/{policy}-bc.pt"); c.add_argument("--replay-data",default="runs/deadline/sarl-pid.npz"); c.add_argument("--steps",type=int,default=5000); c.add_argument("--device",default="cpu"); c.add_argument("--seed",type=int,default=11); c.add_argument("--output",default=f"runs/deadline/{policy}-smoke.pt")
    c=sub.add_parser("marl-data"); c.add_argument("--output",default="runs/deadline/marl-teacher.npz"); c.add_argument("--steps",type=int,default=50_000); c.add_argument("--seed",type=int,default=11)
    c=sub.add_parser("sarl-data"); c.add_argument("--output",default="runs/deadline/sarl-pid.npz"); c.add_argument("--transitions",type=int,default=100_000); c.add_argument("--seed",type=int,default=11)
    c=sub.add_parser("marl-bc"); c.add_argument("--data",default="runs/deadline/marl-teacher.npz"); c.add_argument("--output",default="runs/deadline/marl-bc.pt"); c.add_argument("--epochs",type=int,default=20)
    c=sub.add_parser("sarl-bc"); c.add_argument("--data",default="runs/deadline/sarl-pid.npz"); c.add_argument("--output",default="runs/deadline/sarl-bc.pt"); c.add_argument("--epochs",type=int,default=20)
    c=sub.add_parser("marl-train"); c.add_argument("--preflight",default="runs/deadline/marl-preflight.json"); c.add_argument("--bc",default="runs/deadline/marl-bc.pt"); c.add_argument("--steps",type=int,default=400_000); c.add_argument("--device",default="cpu"); c.add_argument("--seed",type=int,default=11); c.add_argument("--output",default="runs/deadline/marl-mappo.pt")
    c=sub.add_parser("sarl-train"); c.add_argument("--preflight",default="runs/deadline/sarl-preflight.json"); c.add_argument("--bc",default="runs/deadline/sarl-bc.pt"); c.add_argument("--replay-data",default="runs/deadline/sarl-pid.npz"); c.add_argument("--steps",type=int,default=400_000); c.add_argument("--device",default="cpu"); c.add_argument("--seed",type=int,default=11); c.add_argument("--output",default="runs/deadline/sarl-td3.pt")
    for name in ("marl-train","sarl-train"):
        sub.choices[name].add_argument("--smoke-report",default=f"runs/deadline/{name.split('-')[0]}-smoke.pt.smoke.json")
        sub.choices[name].add_argument("--init-checkpoint")
    sub.choices["marl-train"].add_argument("--scenario-mixture",choices=("mixture","pair_focus"),default="mixture")
    a=p.parse_args()
    if hasattr(a,"device") and a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA requested ({a.device}) but torch.cuda.is_available() is false")
    if a.command=="preflight": print(json.dumps(preflight(a.output,a.seed,a.easy_cases,a.marl_bc,a.sarl_bc,a.policy),indent=2)); return
    if a.command in ("marl-preflight","sarl-preflight"):
        policy=a.command.split("-")[0]; kwargs={"marl_bc":a.bc} if policy=="marl" else {"sarl_bc":a.bc}
        print(json.dumps(preflight(a.output,a.seed,50,policy=policy,**kwargs),indent=2)); return
    if a.command=="marl-eval": print(json.dumps(evaluate_marl(a.checkpoint,a.output,per_family=a.per_family,seed=a.seed,device=a.device),indent=2)); return
    if a.command=="sarl-eval": print(json.dumps(evaluate_sarl(a.checkpoint,a.output,episodes=a.episodes,steps=a.steps,seed=a.seed,device=a.device),indent=2)); return
    if a.command=="marl-data": print(json.dumps(collect_marl_teacher(a.output,fleet_steps=a.steps,seed=a.seed))); return
    if a.command=="sarl-data": print(json.dumps(collect_sarl_pid(a.output,transitions=a.transitions,seed=a.seed))); return
    if a.command in ("marl-bc","sarl-bc"):
        x,y,meta=load_dataset(a.data); marl=a.command=="marl-bc"
        with np.load(a.data,allow_pickle=False) as ds:
            ids=ds["scenario_ids" if marl else "episode_ids"]
        heldout_ids=set(np.unique(ids)[-max(1,len(np.unique(ids))//5):].tolist())
        train_mask=np.asarray([int(i) not in heldout_ids for i in ids])
        test_mask=~train_mask
        if not train_mask.any() or not test_mask.any():
            raise SystemExit("BC dataset needs multiple complete scenarios/episodes for a held-out split")
        torch.manual_seed(11)
        model=MAPPO() if marl else TD3Actor()
        losses=clone_actor(model,x[train_mask],y[train_mask],epochs=a.epochs,marl=marl)
        with torch.no_grad():
            if marl:
                raw=model.mean_action(torch.as_tensor(x[test_mask])).numpy()
                pred=np.stack((1/(1+np.exp(-raw[:,0])),np.tanh(raw[:,1])),-1)
                names=("speed_multiplier","heading_residual")
            else:
                pred=model(torch.as_tensor(x[test_mask])).numpy()
                names=("common_thrust","differential_thrust")
        errors=action_error_metrics(pred,y[test_mask],names)
        payload={"model":model.state_dict(),"losses":losses,"dataset_metadata":meta,
                 "dataset_path":str(Path(a.data).resolve()),"training_samples":int(train_mask.sum()),
                 "open_loop_heldout":errors,
                 "observation_normalization":"fixed documented clipping to [-1,1]; no fitted statistics"}
        torch.save(payload,a.output)
        Path(a.output+".metrics.json").write_text(json.dumps(payload|{"model":None},indent=2,default=str)+"\n")
        print(json.dumps({"samples":int(train_mask.sum()),"heldout_samples":int(test_mask.sum()),
            "final_bc_loss":losses[-1],"open_loop_heldout":errors,"output":a.output})); return
    if a.command.endswith("-smoke"):
        policy=a.command.split("-")[0]; gate=json.loads(Path(a.preflight).read_text())
        if not gate.get("production_ready") or not Path(a.bc).is_file():
            raise SystemExit(f"{policy.upper()} smoke refused: policy preflight and BC initialization must pass first.")
        if policy=="marl":
            env=ResidualCoordinatorEnv(scenario_sampler=lambda r:make_four_vessel_scenario(r,"mixture"),sampler_seed=a.seed)
            initial=torch.load(a.bc,map_location="cpu",weights_only=True)["model"]
            baseline_model=MAPPO(); baseline_model.load_state_dict(initial); baseline_model.eval()
            baseline=marl_validation(baseline_model,0,episodes=32,seed=81001)
            train_mappo(env,steps=a.steps,seed=a.seed,output=a.output,device=a.device,initial_state=initial,
                validate=lambda model,step:marl_validation(model,step,episodes=32,seed=81001))
            env.close()
            with open(a.output+".metrics.jsonl") as stream: logs=[json.loads(line) for line in stream if line.strip()]
            with open(a.output+".validation.jsonl") as stream: vals=[json.loads(line)["validation"] for line in stream if line.strip()]
            final=vals[-1] if vals else {}
            std_ranges=np.asarray([row["action_std"] for row in logs],dtype=float)
            smoke_ok=bool(logs and vals and finite_training_logs(logs,"marl") and np.isfinite(std_ranges).all() and
                std_ranges.min()>=.03 and std_ranges.max()<=.3 and
                logs[-1]["zero_speed_action_fraction"]<.95 and logs[-1]["saturated_action_fraction"]<.95 and
                final.get("success_rate",0.)>=baseline["success_rate"]-.15 and
                final.get("collision_rate",1.)<=baseline["collision_rate"]+.10+1e-9 and
                final.get("timeout_rate",1.)<=baseline["timeout_rate"]+.10+1e-9)
        else:
            env=SARLTrackingEnv(); initial=torch.load(a.bc,map_location="cpu",weights_only=True)["model"]
            with np.load(a.replay_data,allow_pickle=False) as data:
                terminal_rows=data["terminals"].copy(); episode_ids=data["episode_ids"]
                terminal_rows[:-1]=np.maximum(terminal_rows[:-1],episode_ids[1:]!=episode_ids[:-1])
                replay=(data["observations"],data["actions"],data["rewards"],
                    data["next_observations"],terminal_rows)
            train_td3(env,steps=a.steps,seed=a.seed,output=a.output,device=a.device,initial_actor=initial,
                initial_replay=replay,validate=lambda actor,step:sarl_validation(actor,step,episodes=4,steps=200))
            env.close()
            with open(a.output+".metrics.jsonl") as stream: logs=[json.loads(line) for line in stream if line.strip()]
            with open(a.output+".validation.jsonl") as stream: vals=[json.loads(line)["validation"] for line in stream if line.strip()]
            final=vals[-1] if vals else {}
            smoke_ok=bool(logs and vals and finite_training_logs(logs,"sarl") and
                final.get("speed_rmse_mps",float("inf"))<.5 and
                final.get("heading_rmse_deg",float("inf"))<20. and
                final.get("saturation_fraction",1.)<.95 and min(final.get("action_std",[0.]))>.01 and
                abs(logs[-1].get("q_mean",float("inf")) or 0.)<1e6 and
                logs[-1]["pid_replay_transitions"]>0)
        result={"policy":policy,"steps":a.steps,"pass":smoke_ok,"validation":final,
            "numerical_logs_finite":finite_training_logs(logs,policy)}
        if policy=="marl": result["bc_validation"]=baseline
        write(a.output+".smoke.json",result)
        print(json.dumps(result,indent=2))
        return
    gate=json.loads(Path(a.preflight).read_text()); policy=a.command.split("-")[0]
    key=f"{policy}_bc_reproduction"
    if not gate.get("production_ready") or not gate.get(key,False):
        raise SystemExit(f"Production mode refused: {policy.upper()} preflight has not passed.")
    smoke=json.loads(Path(a.smoke_report).read_text()) if Path(a.smoke_report).is_file() else {}
    if not smoke.get("pass"):
        raise SystemExit(f"Production mode refused: {policy.upper()} smoke report has not passed.")
    if not Path(a.bc).is_file(): raise SystemExit(f"Required BC initialization is missing: {a.bc}")
    if a.init_checkpoint and not Path(a.init_checkpoint).is_file():
        raise SystemExit(f"Initialization checkpoint is missing: {a.init_checkpoint}")
    if a.command=="marl-train":
        rng=np.random.default_rng(a.seed)
        env=ResidualCoordinatorEnv(scenario_sampler=lambda r: make_four_vessel_scenario(r,a.scenario_mixture),sampler_seed=a.seed)
        initial=torch.load(a.init_checkpoint or a.bc,map_location="cpu",weights_only=True)["model"]
        train_mappo(env,steps=a.steps,seed=a.seed,output=a.output,device=a.device,initial_state=initial,
            validate=lambda model,step:marl_validation(model,step)); env.close()
    else:
        env=SARLTrackingEnv(); source=torch.load(a.init_checkpoint or a.bc,map_location="cpu",weights_only=True)
        initial=source["actor"] if a.init_checkpoint else source["model"]
        with np.load(a.replay_data,allow_pickle=False) as data:
            terminal_rows=data["terminals"].copy(); episode_ids=data["episode_ids"]
            terminal_rows[:-1]=np.maximum(terminal_rows[:-1],episode_ids[1:]!=episode_ids[:-1])
            replay=(data["observations"],data["actions"],data["rewards"],
                    data["next_observations"],terminal_rows)
        train_td3(env,steps=a.steps,seed=a.seed,output=a.output,device=a.device,initial_actor=initial,
                  initial_replay=replay,validate=lambda actor,step:sarl_validation(actor,step)); env.close()


if __name__=="__main__": main()
