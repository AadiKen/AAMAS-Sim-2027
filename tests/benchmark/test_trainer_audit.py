import json
from pathlib import Path

import pytest
import torch

from bcod_sim.benchmark import runner


def train(tmp_path, interval, *, overwrite=False):
    config=tmp_path/'config.json'
    config.write_text('{"max_steps": 2}')
    args=['train','--sim','bcod','--total-steps','6','--num-envs','2',
          '--update-interval','2','--checkpoint-interval',str(interval),
          '--checkpoint-dir',str(tmp_path/'run'),'--config',str(config)]
    if overwrite:args+=['--overwrite']
    runner.main(args)
    return torch.load(tmp_path/'run/checkpoint-000000006.pt',weights_only=False)


def test_checkpoint_schedule_does_not_change_learning_and_runs_do_not_merge(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    first=train(a,2);second=train(b,3)
    assert first['optimizer_updates']==second['optimizer_updates']==3
    assert first['manifest']['gamma_per_second'] == 0.99
    assert first['manifest']['gamma'] == pytest.approx(0.99 ** 0.2)
    initial=torch.load(a/'run/checkpoint-000000000.pt',weights_only=False)
    assert initial['environment_steps'] == initial['optimizer_updates'] == 0
    for key in first['model']:
        assert torch.equal(first['model'][key],second['model'][key])
    rows=[json.loads(line) for line in (a/'run/training.jsonl').read_text().splitlines()]
    assert [r['environment_steps'] for r in rows]==[2,4,6]
    assert [r['optimizer_updates'] for r in rows]==[1,2,3]
    assert all(r['run_id']==first['manifest']['run_id'] for r in rows)
    with pytest.raises(ValueError,match='not empty'):train(a,2)
    third=train(a,3,overwrite=True)
    assert third['manifest']['run_id']!=first['manifest']['run_id']
    archives=list(a.glob('run.archive-*'))
    assert len(archives)==1
    assert (archives[0]/'checkpoint-000000006.pt').exists()
    assert third['optimizer']['state'] and 'rng_states' in third
    assert third['resumable'] is False


@pytest.mark.parametrize('boundary,expected',[(0.,1.),(10.,10.),(None,19.)])
def test_terminal_truncated_and_rollout_bootstrap(boundary,expected):
    # gamma=.9: true terminal return 1; timeout 1+.9*10;
    # unfinished rollout 1+.9*20. A reset value must never leak into timeout.
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    value=model(torch.ones(1)).reshape(1)
    logp=value*0
    optimizer=torch.optim.SGD(model.parameters(),lr=0.)
    records=[(0,logp,value,torch.tensor([1.]),None if boundary is None else torch.tensor([boundary]))]
    loss=runner._update(model,optimizer,records,{0:torch.tensor([20.])},gamma=.9)
    assert loss==pytest.approx(.5*expected**2)


def test_independent_reference_targets_stop_at_episode_boundary():
    from bcod_sim.benchmark.audit_learning import reference_targets
    def record(reward,boundary=None):
        return (0,torch.zeros(1),torch.zeros(1),torch.tensor([reward]),
                None if boundary is None else torch.tensor([boundary]))
    records=[record(1.),record(2.,3.),record(100.),record(4.,0.)]
    reference=reference_targets(records,torch.tensor([999.]),.9)
    production=[float(x[0]) for x in runner._return_targets(records,{0:torch.tensor([999.])},.9)]
    assert reference==pytest.approx([1+.9*2+.9**2*3,2+.9*3,100+.9*4,4])
    assert production==pytest.approx(reference)
