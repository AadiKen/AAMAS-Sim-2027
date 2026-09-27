"""The synthetic sweep resets model state and changes only update frequency."""
import json

import pytest

from bcod_sim.benchmark.audit_learning import run_update_interval_sweep


def test_update_interval_sweep_has_fresh_identical_initial_policies(tmp_path):
    output=tmp_path/'sweep.json'
    results=run_update_interval_sweep(intervals=(20,10),total_steps=20,seed=11,
                                      output_path=output)
    assert [row['optimizer_updates'] for row in results]==[1,2]
    assert [row['update_interval'] for row in results]==[20,10]
    initial=[row['measurements'][0] for row in results]
    assert [row['environment_steps'] for row in initial]==[0,0]
    assert initial[0]['deterministic_surge']==pytest.approx(initial[1]['deterministic_surge'])
    assert initial[0]['log_std_surge']==initial[1]['log_std_surge']
    assert initial[0]['log_std_yaw']==initial[1]['log_std_yaw']
    assert all(row['measurements'][-1]['environment_steps']==20 for row in results)
    assert json.loads(output.read_text())==results
