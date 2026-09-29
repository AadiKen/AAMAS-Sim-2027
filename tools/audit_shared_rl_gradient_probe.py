"""Frozen historical policy, native BCOD rollouts, diagnostic gradients only (lr=0)."""
import json
from pathlib import Path
import numpy as np
import torch
from bcod_sim.benchmark.core import Benchmark, BenchmarkConfig, NAMES, generate_scenario
from bcod_sim.benchmark.runner import SharedActorCritic, make_adapter, _action, _update, _append
from bcod_sim.benchmark.rl_diagnostics import EpisodeDiagnostics,action_summary

torch.set_num_threads(1);torch.manual_seed(909)
source=Path.home()/'Downloads'/'benchmark-bcod-50k-corrected'/'checkpoint-000038250.pt'
saved=torch.load(source,map_location='cpu',weights_only=False);model=SharedActorCritic();model.load_state_dict(saved['model'])
optimizer=torch.optim.SGD(model.parameters(),lr=0.);config=BenchmarkConfig(**saved['manifest']['benchmark'])
output=Path('artifacts/shared-rl-audit/bcod-frozen-gradient-probe');output.mkdir(exist_ok=False)
env=Benchmark(make_adapter('bcod',config),config);obs=env.reset(generate_scenario(800000,split='nominal'));episode=EpisodeDiagnostics();episode_index=0;records=[];actions_rows=[]
try:
 for step in range(1,1501):
  stats={};actions,logp,value=_action(model,obs,'cpu',diagnostics=stats);actions_rows.append(stats)
  obs,reward,done,truncated,info=env.step(actions);episode.add(reward,info,env)
  boundary=None
  if done:boundary=torch.zeros_like(value)
  elif truncated:
   with torch.no_grad():boundary=model(torch.tensor(np.stack([obs[n] for n in NAMES])))[1]
  records.append((0,logp,value,torch.tensor([reward[n] for n in NAMES]),boundary))
  if done or truncated:
   _append(output/'episodes.jsonl',{'environment_steps':step,**episode.result(info)})
   episode_index+=1;obs=env.reset(generate_scenario(800000+episode_index,split='nominal'));episode=EpisodeDiagnostics()
  if step%250==0:
   with torch.no_grad():bootstrap=model(torch.tensor(np.stack([obs[n] for n in NAMES])))[1]
   diag={};_update(model,optimizer,records,{0:bootstrap},diagnostics=diag)
   _append(output/'updates.jsonl',{'environment_steps':step,'frozen_policy':True,**diag,**action_summary(actions_rows)})
   records=[];actions_rows=[];print(step,flush=True)
 assert all(torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items())
finally:env.close()
