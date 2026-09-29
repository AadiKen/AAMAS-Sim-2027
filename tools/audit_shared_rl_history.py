import json
from pathlib import Path
import numpy as np
base=Path.home()/'Downloads'/'benchmark-bcod-50k-corrected'
t=[json.loads(l) for l in (base/'training.jsonl').read_text().splitlines()]
e=[json.loads(l) for l in (base/'episodes.jsonl').read_text().splitlines()]
e=[r for r in e if r['environment_steps']<=t[-1]['environment_steps']]
for r in e:
 r['decomposition_reconstructed']={'collision':-50*r['collision_count'], 'goal':20*sum(r['per_agent_success'].values()), 'step':-.04*r['steps']}
 r['decomposition_reconstructed']['progress']=r['cumulative_reward']-sum(r['decomposition_reconstructed'].values())
windows=[]
for start in range(0,t[-1]['environment_steps'],5000):
 rows=[r for r in e if start<r['environment_steps']<=start+5000]
 windows.append({'mean_reward_components':{key:float(np.mean([r['decomposition_reconstructed'][key] for r in rows])) for key in ('progress','goal','collision','step')} if rows else {},'start':start,'end':min(start+5000,t[-1]['environment_steps']),'episodes':len(rows),'collision_rate':sum(r['collision_count']>0 for r in rows)/len(rows) if rows else None,'mean_episode_length':float(np.mean([r['steps'] for r in rows])) if rows else None,'mean_return':float(np.mean([r['cumulative_reward'] for r in rows])) if rows else None})
result={'source':str(base),'latest':t[-1],'single_run_ids':list({r['run_id'] for r in t+e}), 'schedule_correct':all(r['optimizer_updates']==r['environment_steps']//250 for r in t),'windows':windows,'mean_episode_length':float(np.mean([r['steps'] for r in e])),'median_episode_length':float(np.median([r['steps'] for r in e])), 'training':t}
Path('artifacts/shared-rl-audit/historical-training.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='training'},indent=2))
