"""Inspect first native surge-only episode and independently verify its targets."""
import json
from pathlib import Path

import numpy as np

from bcod_sim.benchmark.core import BenchmarkConfig


def main():
    source=Path('artifacts/shared-rl-diagnosis/A-surge-only-case-a-5000/first_episode_trace.json')
    trace=json.loads(source.read_text())
    assert len(trace)==600
    gamma=BenchmarkConfig().gamma
    rewards=np.asarray([row['reward'] for row in trace])
    surges=np.asarray([row['surge'] for row in trace])
    def correlation(x,y):
        return float(np.corrcoef(x,y)[0,1])
    horizons=(1,5,25,100)
    correlations={}
    index=np.arange(len(trace),dtype=float)
    detrend=np.column_stack((np.ones(len(trace)),index))
    surge_residual=surges-detrend@np.linalg.lstsq(detrend,surges,rcond=None)[0]
    for horizon in horizons:
        returns=np.asarray([sum(gamma**offset*rewards[t+offset]
                                for offset in range(min(horizon,len(trace)-t)))
                            for t in range(len(trace))])
        residual=returns-detrend@np.linalg.lstsq(detrend,returns,rcond=None)[0]
        correlations[str(horizon)]={'raw':correlation(surges,returns),
                                     'time_detrended':correlation(surge_residual,residual)}
    for field in ('return_target','advantage'):
        values=np.asarray([row[field] for row in trace])
        residual=values-detrend@np.linalg.lstsq(detrend,values,rcond=None)[0]
        correlations[field]={'raw':correlation(surges,values),
                             'time_detrended':correlation(surge_residual,residual)}
    errors=[]
    for t,row in enumerate(trace):
        last=249 if t<250 else 499 if t<500 else 599
        total=sum(gamma**(j-t)*rewards[j] for j in range(t,last+1))
        bootstrap=(trace[last]['boundary_value'] if last==599
                   else trace[last]['update_boundary_bootstrap'])
        total+=gamma**(last-t+1)*bootstrap
        errors.append(abs(total-row['return_target']))
    assert max(errors)<3e-4
    samples=[trace[i] for i in (0,124,248,249,250,374,498,499,500,549,598,599)]
    result={'horizon_correlations':correlations,'target_reference_max_abs_error':max(errors),
            'reference_verified_steps':len(trace),'representative_steps':samples,
            'timeout_boundary_index':600,
            'timeout_target_uses_final_observation_not_next_episode':True}
    destination=source.with_name('temporal-credit-analysis.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'horizon_correlations':correlations,
                      'target_reference_max_abs_error':max(errors)},indent=2))


if __name__=='__main__':main()
