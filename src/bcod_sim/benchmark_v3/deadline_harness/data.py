"""Deterministic teacher datasets with schema/version metadata and integrity hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .marl import ResidualCoordinatorEnv, GeometricCoordinator, make_four_vessel_scenario, OBS_VERSION, ACT_VERSION
from .sarl import SARLTrackingEnv, PIDThrustTeacher


def save_dataset(path, observations, actions, metadata, extra_arrays=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    arrays={"observations":np.asarray(observations,dtype=np.float32),
            "actions":np.asarray(actions,dtype=np.float32),
            "metadata":np.asarray(json.dumps(metadata,sort_keys=True))}
    if extra_arrays: arrays.update({k:np.asarray(v) for k,v in extra_arrays.items()})
    np.savez_compressed(path,**arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_marl_teacher(path, *, fleet_steps=50_000, seed=11):
    rng=np.random.default_rng(seed); obs_rows=[]; action_rows=[]; hashes=[]; scenario_ids=[]
    families=("easy","pair","corner","fourway","random")
    env=None
    try:
        done_steps=0
        while done_steps<fleet_steps:
            family=families[int(rng.choice(5,p=[.2,.3,.15,.2,.15]))]
            scenario=make_four_vessel_scenario(rng,family)
            scenario_index=len(hashes)
            if env is not None: env.close()
            env=ResidualCoordinatorEnv(scenario=scenario)
            obs,_=env.reset(seed=scenario.seed); teacher=GeometricCoordinator()
            hashes.append(scenario.geometry_hash())
            while done_steps<fleet_steps:
                actions=teacher.actions(env)
                obs_rows.extend(obs[n] for n in env.possible_agents)
                action_rows.extend(actions[n] for n in env.possible_agents)
                scenario_ids.extend([scenario_index]*len(env.possible_agents))
                obs,_,terms,truncs,_=env.step(actions); done_steps+=1
                if all(terms.values()) or all(truncs.values()): break
    finally:
        if env is not None: env.close()
    digest=save_dataset(path,obs_rows,action_rows,{"seed":seed,"fleet_steps":done_steps,
        "agent_samples":len(action_rows),"scenario_hashes":hashes,
        "observation_version":OBS_VERSION,"action_version":ACT_VERSION})
    # Store case IDs separately so every episode stays wholly in one split.
    with np.load(path,allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files};
    arrays["scenario_ids"]=np.asarray(scenario_ids,np.int32)
    np.savez_compressed(path,**arrays)
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"fleet_steps":done_steps,"agent_samples":len(action_rows),"sha256":digest}


def collect_sarl_pid(path, *, transitions=100_000, seed=11):
    rng=np.random.default_rng(seed)
    episode_steps=min(1000,max(100,transitions//20))
    env=SARLTrackingEnv(episode_steps=episode_steps)
    teacher=PIDThrustTeacher(); observation_rows=[]; action_rows=[]; next_rows=[]; reward_rows=[]; terminal_rows=[]; episode_ids=[]
    try:
        family="speed"
        episode_index=0
        obs,_=env.reset(seed=int(rng.integers(2**31-1)),command_family=family)
        while len(action_rows)<transitions:
            action,_=teacher.predict(obs,env=env)
            observation_rows.append(obs.copy()); action_rows.append(action.copy())
            episode_ids.append(episode_index)
            nxt,reward,term,trunc,_=env.step(action)
            next_rows.append(nxt.copy()); reward_rows.append(reward); terminal_rows.append(float(term))
            obs=nxt
            if term or trunc:
                episode_index+=1
                family=str(rng.choice(["speed","heading","combined"]))
                obs,_=env.reset(seed=int(rng.integers(2**31-1)),command_family=family)
    finally: env.close()
    digest=save_dataset(path,observation_rows,action_rows,{"seed":seed,
        "transitions":len(action_rows),"command_families":["speed","heading","combined"],
        "observation_version":"sarl-tracking-observation-v1",
        "action_version":"common-differential-thrust-v1"},
        {"next_observations":next_rows,"rewards":reward_rows,"terminals":terminal_rows,
         "episode_ids":episode_ids})
    return {"transitions":len(action_rows),"sha256":digest}


def load_dataset(path):
    with np.load(path,allow_pickle=False) as data:
        observations=data["observations"].copy(); actions=data["actions"].copy()
        metadata=json.loads(str(data["metadata"].item()))
    if len(observations)!=len(actions) or not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("Dataset shape or values are invalid")
    return observations,actions,metadata
