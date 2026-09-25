# Headless training control

The Stage 5D REINFORCE trainer accepts `run_dir=...` in `train(...)`. With that option, it exposes a file-backed control surface. The trainer contains no supervisory agent. Its existing campaign calls, which omit `run_dir`, retain their prior behavior.

## Files

A run directory contains immutable `config.yaml`, atomically replaced `state.json`, append-only `metrics.jsonl`, `events.jsonl`, and `commands.jsonl`, policy files in `checkpoints/`, and sentinel files in `control/` when requested. `state.json` contains `status`, `global_step` (completed epochs), `episode`, elapsed `wall_time_s`, latest checkpoint, last evaluation step, `config_version`, current `runtime_config`, and `updated_at`. Metrics include episode return, policy loss, and learning rate; evaluations add their results to `metrics.jsonl`.

Commands are one JSON object per line: `{"id":"unique-id","command":"evaluate","args":{"episodes":50}}`. A command must end with a newline. Write each complete line in one append operation. The trainer reads only complete lines, validates each command, and records `command_applied` or `command_rejected` in `events.jsonl`. Repeated IDs are ignored and logged. Processed IDs are reconstructed from events after a restart. The run directory is intended for one trainer and cooperating clients on a local filesystem.

Supported commands: `set_learning_rate` with finite `value` in `(0,1]`; `evaluate` with integer `episodes` in `1..1000`; `checkpoint`; `pause`; `resume`; `stop`. Only the learning rate is runtime mutable, and each change increments `config_version`. Reward weights, curriculum stage, scenario distribution, and rollback are rejected by this trainer because those controls or full optimizer/RNG restore are absent. Checkpoints contain policy values and metadata; they are **not** full training continuation snapshots.

`control/PAUSE` pauses at the next completed epoch. Removing it resumes; a `resume` command also removes it. The trainer keeps polling commands while paused. `control/STOP` saves a final policy checkpoint, records `stopped`, and exits after the current epoch. `control/KILL_NOW` takes priority at the next poll, records `killed`, and exits without a checkpoint or evaluation. These are cooperative boundaries, not OS signals; a long epoch cannot be interrupted mid-rollout. A normal completed run records `stopped`. Commands and events are flushed on write.

```text
initializing → running ↔ paused
                 ↓  ↘ evaluating/checkpointing → running
              stopping → stopped
                 ↘ killed
```

## Python client

```python
from bcod_sim.rl.training_control import TrainingRun

run = TrainingRun('runs/example')
while True:
    state = run.state()
    metrics = run.recent_metrics()
    # External reasoning happens here.
    if metrics and metrics[-1].get('episode_return', 0) < 1:
        run.set_learning_rate(0.1)
        break
run.evaluate(50)
run.checkpoint()
run.pause()
run.resume()
run.stop()
# Emergency: run.kill()
```

## CLI

```bash
bcod train status runs/example
bcod train set runs/example learning_rate 0.1
bcod train evaluate runs/example --episodes 50
bcod train checkpoint runs/example
bcod train pause runs/example
bcod train resume runs/example
bcod train stop runs/example
bcod train kill runs/example
```

CLI and Python commands return a command ID. Inspect `events.jsonl` to learn whether the trainer applied or rejected it. The original experiment configuration remains in `config.yaml`; applied changes and their old/new values are in the event log.
