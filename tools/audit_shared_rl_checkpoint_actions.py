"""Fixed-reset observation probe of historical policy parameters (not rollouts)."""
import json
from pathlib import Path
import numpy as np
import torch
from bcod_sim.benchmark.core import *
from bcod_sim.benchmark.runner import SharedActorCritic
base=Path('/Users/aadikenchammanaold/Downloads/benchmark-bcod-50k-corrected')
torch.set_num_threads(1);torch.manual_seed(11);model=SharedActorCritic();config=BenchmarkConfig()
observations=[]
for index in range(100):
 s=generate_scenario(700000+index,split='nominal')
 for i,p in enumerate(s.starts):
  r=VesselReading(p.x_m,p.y_m,p.heading_rad,0,0,nearby_agents=[(q.x_m,q.y_m) for j,q in enumerate(s.starts) if j!=i],nearby_obstacles=[(o.x_m,o.y_m,o.radius_m) for o in s.obstacles])
  observations.append(observation(r,s.goals[i],config))
obs=torch.tensor(np.stack(observations));rows=[]
for checkpoint in [None]+sorted(base.glob('checkpoint-*.pt')):
 step=0
 if checkpoint:
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);step=saved['environment_steps']
 with torch.no_grad():action=torch.tanh(model(obs)[0])
 rows.append({'steps':step,'mean_deterministic_surge':float(action[:,0].mean()),'mean_abs_deterministic_yaw':float(action[:,1].abs().mean()),'raw_std':model.log_std.exp().detach().tolist(),'log_std':model.log_std.detach().tolist(),'negative_surge_fraction':float((action[:,0]<0).float().mean())})
Path('artifacts/shared-rl-audit/historical-action-probe.json').write_text(json.dumps(rows,indent=2))
print(rows[0]);print(rows[-1])
