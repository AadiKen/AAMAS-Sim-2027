"""Constant-command case-A return landscape, using production scoring."""
import json
import math
from pathlib import Path

import numpy as np

from bcod_sim.benchmark.audit_learning import scenario
from bcod_sim.benchmark.core import Benchmark, BenchmarkConfig, NAMES
from bcod_sim.benchmark.runner import make_adapter


def main():
    config=BenchmarkConfig()
    speeds=(0.,.05,.10,.20,.30,.40,.50,.60,.70,.80,1.)
    rows=[]
    env=Benchmark(make_adapter('bcod-reduced',config),config,diagnostic_allow_sparse=True)
    try:
        for surge in speeds:
            episodes=[]
            for index in range(3):
                env.reset(scenario('A',700000+index))
                components=dict.fromkeys(('potential_shaping','goal','collision','step'),0.)
                discounted=0.;undiscounted=0.
                for step in range(config.max_steps):
                    actions={NAMES[0]:(surge,0.)}
                    actions.update({n:(0.,0.) for n in NAMES[1:]})
                    _,reward,done,truncated,info=env.step(actions)
                    discounted+=config.gamma**step*reward[NAMES[0]]
                    undiscounted+=reward[NAMES[0]]
                    for key,value in info['reward_components'][NAMES[0]].items():
                        components[key]+=value
                    if done or truncated:break
                p=env.truth[NAMES[0]]
                episodes.append({'seed':700000+index,'success':bool(info['per_agent_success'][NAMES[0]] and not info['fleet_failure_due_to_collision']),
                    'episode_length':info['steps'],'final_goal_distance_m':math.dist((p.x_m,p.y_m),env.scenario.goals[0]),
                    'undiscounted_return':undiscounted,'discounted_return':discounted,
                    'components':components})
            rows.append({'surge':surge,'success_rate':float(np.mean([e['success'] for e in episodes])),
                'mean_episode_length':float(np.mean([e['episode_length'] for e in episodes])),
                'mean_final_goal_distance_m':float(np.mean([e['final_goal_distance_m'] for e in episodes])),
                'mean_undiscounted_return':float(np.mean([e['undiscounted_return'] for e in episodes])),
                'mean_discounted_return':float(np.mean([e['discounted_return'] for e in episodes])),
                'mean_components':{key:float(np.mean([e['components'][key] for e in episodes])) for key in components},
                'episodes':episodes})
            print('surge',surge,'discounted',rows[-1]['mean_discounted_return'],flush=True)
    finally:
        env.close()
    destination=Path('artifacts/shared-rl-diagnosis/case-a-constant-surge-landscape.json')
    destination.write_text(json.dumps({'simulator':'bcod-reduced','seeds':[700000,700001,700002],
        'gamma':config.gamma,'rows':rows},indent=2)+'\n')
    print('wrote',destination)


if __name__=='__main__':main()
