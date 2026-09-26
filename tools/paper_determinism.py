#!/usr/bin/env python3
"""100 same-seed multi-vessel replays with noise, environment, and contact scene."""
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/validation"))
from test_stage5a_marl import command, make_world
from bcod_sim.scenario.generator import ScenarioTemplate

OUT = ROOT / "paper_results/determinism"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = ("a", "b", "c")
ENVIRONMENT = {"current": {"kind": "uniform", "ned_mps": [0.02, 0, 0]}, "wind": {"kind": "uniform", "ned_mps": [0.01, 0.01, 0]}, "waves": {"kind": "irregular", "spectrum": "jonswap", "significant_height_m": 0.01, "peak_period_s": 5., "direction_rad": 0.2, "component_count": 8, "seed": 31}, "visibility_m": 1000}

def make(order):
    env = make_world(NAMES, order=order, max_steps=12, goal=(1e9,0,0),
        sensor_by_name={name: ("gps", 10, .01, 0, 5+i) for i,name in enumerate(NAMES)},
        environment_config=ENVIRONMENT,
        spawn_by_name={"a": (0,0,0), "b": (1,0,0), "c": (20,0,0)},
        static_entities=({"id": "marker", "position_ned_m": (0.5,0,0), "shape": {"kind": "sphere", "radius_m": .25}},), bottom=2.)
    env.engine.template = ScenarioTemplate.model_validate({"id":"random_spawn", "version":"1", "spawn_overrides":{"c":{"north_m":{"kind":"uniform", "low":18, "high":22}}}})
    return env

def snapshot(env, observations, rewards, terms, truncs, infos):
    parts = []
    numeric = []
    for name in NAMES:
        state = env.engine.states[name]
        for value in (state.position_ned, state.q_body_to_ned, state.nu_body):
            a = value.detach().cpu().numpy().astype("<f8", copy=False)
            parts.append(a.tobytes()); numeric.extend(a.tolist())
        for key in sorted(observations.get(name, {})):
            value = observations[name][key]
            if isinstance(value, torch.Tensor):
                a = value.detach().cpu().numpy().astype("<f8", copy=False)
                parts.append(key.encode()+a.tobytes()); numeric.extend(a.ravel().tolist())
        parts.append(np.float64(rewards[name]).tobytes()); numeric.append(float(rewards[name]))
        parts.append(bytes((terms[name], truncs[name])))
        parts.append(repr(infos[name].get("contact_events", ())).encode())
    return b"".join(parts), np.array(numeric)

def rollout(env, seed):
    env.reset(seed=seed)
    packets=[]; vectors=[]
    for step in range(12):
        values=(1, -1, 0) if step%2 else (0, 1, -1)
        obs, rew, term, trunc, info = env.step({name:command(value) for name,value in zip(NAMES,values)})
        packet, vector=snapshot(env,obs,rew,term,trunc,info)
        packets.append(packet); vectors.append(vector)
    digest=hashlib.sha256(b"".join(packets)).hexdigest()
    return digest, np.concatenate(vectors)

start=time.perf_counter(); env=make(NAMES); rows=[]; baseline=None
for i in range(100):
    digest, vector=rollout(env,73)
    if baseline is None: baseline=(digest,vector)
    rows.append({"repetition":i+1,"seed":73,"hash":digest,"hash_equal":digest==baseline[0],"maximum_numeric_deviation":float(np.max(np.abs(vector-baseline[1])))})
permuted=make(tuple(reversed(NAMES))); permuted_digest, permuted_vector=rollout(permuted,73)
different=[rollout(env,seed)[0] for seed in (74,75,76)]
with (OUT/"repetitions.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,rows[0].keys());writer.writeheader();writer.writerows(rows)
summary={"same_seed_identical":sum(row["hash_equal"] for row in rows),"repetitions":100,"maximum_numeric_deviation":max(row["maximum_numeric_deviation"] for row in rows),"agent_order_hash_equal":permuted_digest==baseline[0],"agent_order_maximum_numeric_deviation":float(np.max(np.abs(permuted_vector-baseline[1]))),"different_seed_distinct_hashes":len(set(different+[baseline[0]]))==4,"baseline_sha256":baseline[0],"different_seed_sha256":different}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
(OUT/"manifest.json").write_text(json.dumps({"command":f"{sys.executable} tools/paper_determinism.py","commit":commit,"dirty":dirty,"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,"device":"CPU","seed":73,"resolved_config":{"vessels":NAMES,"steps":12,"environment":ENVIRONMENT,"sensor":"GPS noise std 0.01","obstacle":"sphere at 0.5 m"},"runtime_s":time.perf_counter()-start,"status":"PASS" if summary["same_seed_identical"]==100 else "FAIL","evidence_class":"internal_consistency"},indent=2)+"\n")
print(json.dumps(summary))
