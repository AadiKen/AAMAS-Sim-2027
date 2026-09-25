import json
import threading
import time

from bcod_sim.rl.training_control import TrainingControl, TrainingRun, append


def events(path):
    return [json.loads(x) for x in (path / 'events.jsonl').read_text().splitlines()]


def test_control_lifecycle(tmp_path):
    values = []
    def checkpoint(step):
        path = tmp_path / 'checkpoints' / f'step_{step}.pt'
        path.write_bytes(b'policy'); return path
    control = TrainingControl(tmp_path, {'learning_rate': .2}, set_learning_rate=values.append,
                              evaluate=lambda count: {'success_rate': .5, 'episodes': count},
                              checkpoint=checkpoint)
    run = TrainingRun(tmp_path)
    assert run.state()['status'] == 'initializing'
    control.progress(step=1, episode=4, metrics={'episode_return': 2.0})
    assert run.metrics()[0]['episode_return'] == 2.0
    good = run.set_learning_rate(.1)
    bad = run.set_learning_rate(-5)
    append(tmp_path / 'commands.jsonl', {'id': good, 'command': 'set_learning_rate', 'args': {'value': .9}})
    control.poll()
    assert values == [.1]
    assert run.state()['config_version'] == 1
    assert run.config()['learning_rate'] == .2
    assert any(e.get('command_id') == bad and e['type'] == 'command_rejected' for e in events(tmp_path))
    assert any(e.get('command_id') == good and e['type'] == 'command_duplicate' for e in events(tmp_path))
    run.pause(); assert not control.poll() and run.state()['status'] == 'paused'
    run.resume(); assert control.poll()
    run.evaluate(3); run.checkpoint(); control.poll()
    assert run.state()['last_evaluation_step'] == 1
    assert run.state()['latest_checkpoint'] == 'checkpoints/step_1.pt'
    run.stop(); assert not control.poll()
    assert run.state()['status'] == 'stopped'
    assert sum(e['type'] == 'checkpoint_saved' for e in events(tmp_path)) == 2


def test_sentinels_and_restart(tmp_path):
    control = TrainingControl(tmp_path, {'learning_rate': .2})
    run = TrainingRun(tmp_path)
    (tmp_path / 'control' / 'PAUSE').touch()
    assert not control.poll()
    (tmp_path / 'control' / 'PAUSE').unlink()
    assert control.poll()
    command = run.set_reward_weight('collision', 2.5)
    control.poll()
    restarted = TrainingControl(tmp_path, {'learning_rate': .2})
    restarted.poll()
    assert sum(e.get('command_id') == command and e['type'] == 'command_rejected' for e in events(tmp_path)) == 1
    run.kill(); assert not restarted.poll()
    assert run.state()['status'] == 'killed'


def test_external_supervisor_flow(tmp_path):
    seen = []
    def checkpoint(step):
        path = tmp_path / 'checkpoints' / f'{step}.pt'; path.write_bytes(b'x'); return path
    control = TrainingControl(tmp_path, {'learning_rate': .2}, set_learning_rate=seen.append, checkpoint=checkpoint)
    run = TrainingRun(tmp_path)
    def trainer():
        step = 0
        while control.wait_if_paused():
            step += 1
            control.progress(step=step, episode=step, metrics={'episode_return': step})
            time.sleep(.005)
        control.finish()
    thread = threading.Thread(target=trainer); thread.start()
    try:
        deadline = time.monotonic() + 3
        def until(predicate):
            while not predicate():
                assert time.monotonic() < deadline
                time.sleep(.01)
        until(lambda: run.state()['global_step'] > 0)
        run.set_learning_rate(.1); until(lambda: seen == [.1])
        run.pause(); until(lambda: run.state()['status'] == 'paused')
        run.resume(); until(lambda: run.state()['status'] == 'running')
        run.checkpoint(); until(lambda: run.state()['latest_checkpoint'] is not None)
        run.stop(); thread.join(3)
        assert not thread.is_alive() and run.state()['status'] == 'stopped'
    finally:
        run.kill(); thread.join(3)


def test_partial_line_validation_and_kill_priority(tmp_path):
    invoked = []
    control = TrainingControl(tmp_path, {'learning_rate': .2},
                              evaluate=lambda count: invoked.append(count))
    commands = tmp_path / 'commands.jsonl'
    with commands.open('ab') as file: file.write(b'{"id":"partial","command":"evaluate","args":{"episodes":2}')
    control.poll()
    assert not invoked
    with commands.open('ab') as file: file.write(b'}\nnot-json\n')
    control.poll()
    assert invoked == [2]
    assert any(e['type'] == 'command_rejected' for e in events(tmp_path))
    TrainingRun(tmp_path).evaluate(3)
    TrainingRun(tmp_path).kill()
    assert not control.poll() and invoked == [2]
