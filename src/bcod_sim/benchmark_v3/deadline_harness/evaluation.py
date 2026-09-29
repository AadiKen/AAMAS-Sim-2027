"""Paired deterministic PID/TD3 tracking evaluation with raw CSV output."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .algorithms import TD3Actor
from .sarl import PIDThrustTeacher,SARLTrackingEnv
from .marl import ResidualCoordinatorEnv,GeometricCoordinator,make_four_vessel_scenario
from .algorithms import MAPPO
from ..control.high_level import wrap_angle


def evaluate_sarl(checkpoint, output, *, episodes=6, seed=44001, steps=500):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    actor=TD3Actor(); actor.load_state_dict(payload.get("actor",payload.get("model"))); actor.eval()
    out=Path(output); tsdir=out/"timeseries"; tsdir.mkdir(parents=True,exist_ok=True)
    families=("nominal","heading","combined","current","wind","parameter")
    reports=[]
    for episode in range(episodes):
        family=families[episode%len(families)]; seed_i=seed+episode
        command_family="speed" if family=="nominal" else "heading" if family=="heading" else "combined"
        paired={}; all_rows=[]
        for controller in ("pid","td3"):
            disturbance=family if family in ("current","wind","parameter") else "nominal"
            env=SARLTrackingEnv(episode_steps=steps); obs,_=env.reset(seed=seed_i,
                command_family=command_family,disturbance=disturbance)
            env._maybe_new_command=lambda:None
            if family=="nominal":
                env.desired_speed=.9; env._maybe_new_command=lambda:None
            if family in ("current","wind","parameter"):
                env._maybe_new_command=lambda:None
            if family in ("heading","combined"):
                env.desired_heading=wrap_angle(env.desired_heading+math.radians(90.))
            if family=="combined": env.desired_speed=1.4
            teacher=PIDThrustTeacher(); rows=[]; prev_action=np.zeros(2); settling=None
            for tick in range(steps):
                if controller=="pid": action,_=teacher.predict(obs,env=env)
                else:
                    with torch.no_grad(): action=actor(torch.as_tensor(obs).float()).cpu().numpy()
                obs,_,term,trunc,info=env.step(action)
                row={"episode":episode,"seed":seed_i,"family":family,"controller":controller,
                    "step":tick,"time_s":info["time_s"],"command_speed_mps":info["command"][0],
                    "command_heading_rad":info["command"][1],"speed_mps":info["actual_speed"],
                    "speed_error_mps":info["speed_error"],"heading_error_rad":info["heading_error"],
                    "common_action":float(action[0]),"differential_action":float(action[1]),
                    "left_command":info["thrust"][0],"right_command":info["thrust"][1],
                    "saturated":any(info["saturation"].values())}
                rows.append(row); all_rows.append(row)
                prev_action=np.asarray(action)
                if term or trunc: break
            env.close()
            dt_s=env.config.dt_s
            se=np.asarray([r["speed_error_mps"] for r in rows]); he=np.asarray([r["heading_error_rad"] for r in rows])
            for index,row in enumerate(rows):
                suffix=rows[index:]
                if all(abs(item["speed_error_mps"])<.05 and abs(item["heading_error_rad"])<math.radians(5) for item in suffix):
                    settling=row["time_s"]; break
            actions=np.asarray([[r["common_action"],r["differential_action"]] for r in rows])
            overshoot=max(0.,max(r["speed_mps"]-r["command_speed_mps"] for r in rows))
            paired[controller]={"speed_rmse_mps":float(np.sqrt(np.mean(se**2))),
                "heading_rmse_deg":float(np.degrees(np.sqrt(np.mean(he**2)))),
                "heading_mae_deg":float(np.degrees(np.mean(np.abs(he)))),
                "integrated_abs_speed_error_m_s":float(np.abs(se).sum()*dt_s),
                "integrated_abs_heading_error_rad_s":float(np.abs(he).sum()*dt_s),
                "settling_time_s":settling,
                "speed_overshoot_mps":float(overshoot),
                "actuator_effort":float(np.square(actions).sum()),
                "actuator_slew":float(np.square(np.diff(actions,axis=0)).sum()) if len(actions)>1 else 0.,
                "saturation_fraction":float(np.mean([r["saturated"] for r in rows]))}
        reports.append({"episode":episode,"seed":seed_i,"family":family,
                        **{f"{c}_{k}":v for c,m in paired.items() for k,v in m.items()}})
        with (tsdir/f"episode-{episode:03d}.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    with (out/"per_episode.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(reports[0])); w.writeheader(); w.writerows(reports)
    metric_keys=[k for k in reports[0] if k not in ("episode","seed","family")]
    summary={"episodes":len(reports),"paired_identical_seeds":True,
        "disturbances":{"current_mps":.25,"wind_mps":3.,"damping_scale":1.05},
        "mean":{k:(float(np.mean(values)) if values else None) for k in metric_keys
            for values in [[r[k] for r in reports if r[k] is not None]]},
        "episodes_metrics":reports}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def evaluate_marl(checkpoint,output,*,per_family=50,seed=62001):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=MAPPO(); model.load_state_dict(payload.get("model",payload)); model.eval()
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[];rng=np.random.default_rng(seed)
    families=("fourway","pairwise_corner","random","mild_current")
    family_names=("four_way","pairwise_corner","randomized","mild_current")
    for family,family_name in zip(families,family_names):
        for index in range(per_family):
            if family == "pairwise_corner":
                family = "pair" if index % 2 == 0 else "corner"
            scenario=make_four_vessel_scenario(rng,family,split="frozen-test")
            for policy_name in ("nominal","scripted","mappo"):
                env=ResidualCoordinatorEnv(scenario=scenario);obs,_=env.reset(seed=scenario.seed)
                if family_name=="mild_current":
                    env.backend.engine.world.current.vector_ned_mps=(.15,0.,0.)
                minimum=float("inf")
                for step in range(env.config.deadline_steps):
                    if policy_name=="nominal": actions={n:np.array([1.,0.],np.float32) for n in env.possible_agents}
                    elif policy_name=="scripted": actions=GeometricCoordinator().actions(env)
                    else:
                        names=list(env.possible_agents)
                        with torch.no_grad():
                            matrix=model.act(torch.as_tensor(np.stack([obs[n] for n in names])),deterministic=True)[0].numpy()
                        actions={n:matrix[i].astype(np.float32) for i,n in enumerate(names)}
                    obs,_,terms,truncs,infos=env.step(actions)
                    minimum=min(minimum,env.task.min_separation_m)
                    if all(terms.values()) or all(truncs.values()):break
                reason=infos[env.possible_agents[0]]["terminal_reason"]
                path_eff=float(np.mean([min(1.,math.dist((scenario.starts[i].x_m,scenario.starts[i].y_m),scenario.goals[i])/
                    max(env.task.path_lengths[n],1e-9)) for i,n in enumerate(env.possible_agents)]))
                rows.append({"policy":policy_name,"family":family_name,"seed":scenario.seed,
                    "scenario_hash":scenario.geometry_hash(),"success":reason=="success",
                    "collision":reason=="collision","timeout":reason=="deadline",
                    "completion_time_s":step*env.config.dt_s,
                    "minimum_separation_m":None if math.isinf(minimum) else minimum,
                    "path_length_m":float(np.mean(list(env.task.path_lengths.values()))),
                    "path_efficiency":path_eff})
                env.close()
            if family_name == "pairwise_corner": family = "pairwise_corner"
    fields=list(rows[0]);
    with (out/"per_episode.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={}
    for name in ("nominal","scripted","mappo"):
        group=[r for r in rows if r["policy"]==name];n=len(group)
        summary[name]={"episodes":n,"fleet_success_rate":sum(r["success"] for r in group)/n,
            "collision_rate":sum(r["collision"] for r in group)/n,
            "timeout_rate":sum(r["timeout"] for r in group)/n,
            "mean_completion_time_s":float(np.mean([r["completion_time_s"] for r in group])),
            "mean_minimum_separation_m":float(np.nanmean([r["minimum_separation_m"] if r["minimum_separation_m"] is not None else np.nan for r in group])),
            "mean_path_length_m":float(np.mean([r["path_length_m"] for r in group])),
            "mean_path_efficiency":float(np.mean([r["path_efficiency"] for r in group]))}
    result={"test_episodes_per_policy":len(rows)//3,"families":family_names,"summary":summary}
    (out/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result
