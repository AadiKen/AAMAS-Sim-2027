"""File-backed control for one active trainer per run directory."""
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import yaml


def stamp(): return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temp.open('w') as file:
            file.write(value); file.flush(); os.fsync(file.fileno())
        os.replace(temp, path)
    finally: temp.unlink(missing_ok=True)


def append(path, value):
    payload = (json.dumps(value, allow_nan=False, sort_keys=True) + '\n').encode()
    with path.open('ab', buffering=0) as file:
        os.write(file.fileno(), payload); os.fsync(file.fileno())


class TrainingRun:
    def __init__(self, path): self.path = Path(path)
    def state(self): return json.loads((self.path / 'state.json').read_text())
    def config(self): return yaml.safe_load((self.path / 'config.yaml').read_text())
    def recent_metrics(self, count=20):
        path = self.path / 'metrics.jsonl'
        return [json.loads(x) for x in path.read_text().splitlines()[-count:]] if path.exists() else []
    def metrics(self): return self.recent_metrics(10**9)
    def command(self, command_name, **args):
        identifier = uuid.uuid4().hex
        append(self.path / 'commands.jsonl', {'id': identifier, 'command': command_name, 'args': args})
        return identifier
    def set_learning_rate(self, value): return self.command('set_learning_rate', value=value)
    def set_reward_weight(self, name, value): return self.command('set_reward_weight', name=name, value=value)
    def set_curriculum_stage(self, stage): return self.command('set_curriculum_stage', stage=stage)
    def set_scenario_distribution(self, distribution): return self.command('set_scenario_distribution', distribution=distribution)
    def evaluate(self, episodes=20): return self.command('evaluate', episodes=episodes)
    def checkpoint(self): return self.command('checkpoint')
    def pause(self): return self.command('pause')
    def resume(self): return self.command('resume')
    def stop(self): return self.command('stop')
    def rollback(self, checkpoint): return self.command('rollback', checkpoint=checkpoint)
    def kill(self): (self.path / 'control' / 'KILL_NOW').touch()


class TrainingControl:
    def __init__(self, path, config, *, set_learning_rate=None, evaluate=None, checkpoint=None):
        self.path = Path(path); self.path.mkdir(parents=True, exist_ok=True)
        (self.path / 'control').mkdir(exist_ok=True)
        (self.path / 'checkpoints').mkdir(exist_ok=True)
        if not (self.path / 'config.yaml').exists():
            atomic(self.path / 'config.yaml', yaml.safe_dump(config, sort_keys=True))
        self.runtime_config = dict(config)
        self.set_learning_rate, self.evaluate_callback, self.checkpoint_callback = set_learning_rate, evaluate, checkpoint
        self.start = time.monotonic(); self.offset = 0; self.pending = b''; self.processed = set()
        self.step = self.episode = self.config_version = 0
        self.latest_checkpoint = self.last_evaluation_step = None
        self.status = 'initializing'; self.paused_by_command = self.terminal = False
        self._recover(); self._state(); self._event('initialized')

    def _recover(self):
        path = self.path / 'events.jsonl'
        if not path.exists(): return
        for line in path.read_text().splitlines():
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            if event.get('type') in ('command_applied', 'command_rejected'):
                self.processed.add(event.get('command_id'))
            if event.get('type') == 'command_applied' and event.get('command') == 'set_learning_rate':
                self.runtime_config['learning_rate'] = event['changes']['learning_rate']['new']
                self.config_version += 1

    def _event(self, kind, **fields):
        append(self.path / 'events.jsonl', {'timestamp': stamp(), 'step': self.step, 'type': kind, **fields})

    def _state(self):
        atomic(self.path / 'state.json', json.dumps({
            'status': self.status, 'global_step': self.step, 'episode': self.episode,
            'wall_time_s': time.monotonic() - self.start,
            'latest_checkpoint': self.latest_checkpoint,
            'last_evaluation_step': self.last_evaluation_step,
            'config_version': self.config_version, 'runtime_config': self.runtime_config,
            'updated_at': stamp()}, allow_nan=False, sort_keys=True) + '\n')

    def progress(self, *, step, episode, metrics=None):
        self.step, self.episode = step, episode
        if metrics is not None: append(self.path / 'metrics.jsonl', {'step': step, 'episode': episode, **metrics})
        self._state()

    def _checkpoint(self):
        if self.checkpoint_callback is None: raise ValueError('checkpoint unavailable')
        self.status = 'checkpointing'; self._state()
        result = Path(self.checkpoint_callback(self.step))
        if not result.resolve().is_relative_to((self.path / 'checkpoints').resolve()) or not result.is_file():
            raise ValueError('checkpoint must be a file inside run/checkpoints')
        self.latest_checkpoint = str(result.relative_to(self.path))
        self._event('checkpoint_saved', checkpoint=self.latest_checkpoint)
        self.status = 'running'; self._state()
        return self.latest_checkpoint

    def _apply(self, item):
        if not isinstance(item, dict): raise ValueError('command must be an object')
        identifier = item.get('id')
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            raise ValueError('id must be a nonempty string of at most 128 characters')
        if identifier in self.processed:
            self._event('command_duplicate', command_id=identifier); return
        name, args = item.get('command'), item.get('args', {})
        try:
            if not isinstance(args, dict): raise ValueError('args must be an object')
            changes = {}
            if name == 'set_learning_rate':
                if set(args) != {'value'}: raise ValueError('expected value')
                value = args['value']
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                    raise ValueError('learning_rate must be finite and in (0, 1]')
                if self.set_learning_rate is None: raise ValueError('learning_rate is not runtime mutable')
                old = self.runtime_config['learning_rate']; self.set_learning_rate(float(value))
                self.runtime_config['learning_rate'] = float(value); self.config_version += 1
                changes = {'learning_rate': {'old': old, 'new': float(value)}}
            elif name == 'evaluate':
                if set(args) != {'episodes'}: raise ValueError('expected episodes')
                count = args['episodes']
                if type(count) is not int or not 1 <= count <= 1000: raise ValueError('episodes must be 1..1000')
                if self.evaluate_callback is None: raise ValueError('evaluation unavailable')
                self.status = 'evaluating'; self._state()
                result = self.evaluate_callback(count)
                append(self.path / 'metrics.jsonl', {'step': self.step, 'evaluation': result})
                self.last_evaluation_step = self.step; changes = {'evaluation': result}; self.status = 'running'
            elif name == 'checkpoint':
                if args: raise ValueError('checkpoint takes no args')
                changes = {'checkpoint': self._checkpoint()}
            elif name == 'pause':
                if args: raise ValueError('pause takes no args')
                self.paused_by_command = True
            elif name == 'resume':
                if args: raise ValueError('resume takes no args')
                self.paused_by_command = False; (self.path / 'control' / 'PAUSE').unlink(missing_ok=True)
            elif name == 'stop':
                if args: raise ValueError('stop takes no args')
                (self.path / 'control' / 'STOP').touch()
            else: raise ValueError(f'{name!r} is not runtime mutable or supported by this trainer')
            self._event('command_applied', command_id=identifier, command=name, changes=changes)
        except Exception as error:
            self._event('command_rejected', command_id=identifier, command=name, reason=str(error))
        self.processed.add(identifier)

    def poll(self):
        if self.terminal: return False
        control = self.path / 'control'
        if (control / 'KILL_NOW').exists():
            self.status = 'killed'; self.terminal = True; self._event('killed'); self._state(); return False
        path = self.path / 'commands.jsonl'
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(self.offset); self.pending += stream.read(); self.offset = stream.tell()
            while b'\n' in self.pending:
                line, self.pending = self.pending.split(b'\n', 1)
                try: self._apply(json.loads(line))
                except (ValueError, UnicodeDecodeError) as error:
                    self._event('command_rejected', reason=f'malformed JSONL: {error}')
        control = self.path / 'control'
        if (control / 'KILL_NOW').exists():
            self.status = 'killed'; self.terminal = True; self._event('killed'); self._state(); return False
        if (control / 'STOP').exists():
            self.status = 'stopping'; self._state()
            try: self._checkpoint()
            except Exception as error:
                self.status = 'failed'; self.terminal = True
                self._event('checkpoint_failed', reason=str(error)); self._state(); return False
            self.status = 'stopped'; self.terminal = True; self._event('stopped'); self._state(); return False
        self.status = 'paused' if self.paused_by_command or (control / 'PAUSE').exists() else 'running'
        self._state(); return self.status == 'running'

    def wait_if_paused(self, interval=0.05):
        while not self.poll() and not self.terminal: time.sleep(interval)
        return not self.terminal

    def finish(self):
        if not self.terminal:
            self.status = 'stopped'; self.terminal = True; self._event('completed'); self._state()

    def fail(self, error):
        self.status = 'failed'; self.terminal = True; self._event('failed', reason=str(error)); self._state()
