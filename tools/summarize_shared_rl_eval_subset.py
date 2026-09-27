"""Summarize the common first 30 scenarios at each historical checkpoint."""
import json
from pathlib import Path
import numpy as np
root=Path('artifacts/shared-rl-audit')
for p in sorted(root.glob('bcod-eval-*/episodes.jsonl')):
 rows=[json.loads(l) for l in p.read_text().splitlines()][:30]
 if len(rows)<30:continue
 step=int(p.parent.name.split('-')[-1]);counts={n:float(np.mean([r['per_agent_success'][n] for r in rows])) for n in rows[0]['per_agent_success']}
 summary={'environment_steps':step,'episodes':30,'scenario_seeds':[r['scenario_seed'] for r in rows],
          'fleet_success':float(np.mean([r['fleet_success'] for r in rows])),
          'per_agent_success':counts,'collision_rate':float(np.mean([r['collision_count']>0 for r in rows])),
          'mean_return':float(np.mean([r['fleet_return'] for r in rows])),
          'mean_discounted_return':float(np.mean([r['discounted_fleet_return'] for r in rows])),
          'mean_episode_length':float(np.mean([r['steps'] for r in rows])),
          'path_efficiency':float(np.mean([v for r in rows for v in r['path_efficiency'].values()])),
          'subset':'first 30 of fixed nominal-100 bank'}
 assert summary['scenario_seeds']==list(range(700000,700030))
 (p.parent/'summary-first30.json').write_text(json.dumps(summary,indent=2))
 print(step,summary['collision_rate'],summary['fleet_success'])
