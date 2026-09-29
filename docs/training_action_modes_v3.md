# V3 policy action modes

Set `policy_action.mode` to `low_level` or `high_level` in a V3 YAML config, or pass `--action-mode low-level|high-level` to `python -m bcod_sim.benchmark_v3 train`. Python callers can set `TaskConfig(action_mode="high_level")`. The modes share the same 78-value observation and `Box([-1,-1], [1,1])` tensor but have distinct schemas.

| Mode | Schema | Policy action | Intended use |
| --- | --- | --- | --- |
| `low_level` | `desired-speed-yawrate-v2` | desired speed and desired yaw rate | Control research, controller learning, direct maneuvering, RL/control comparisons |
| `high_level` | `desired-speed-heading-v1` | desired speed and relative heading | Navigation, obstacle avoidance, fleet coordination, formation behavior, MARL |

Both modes map the first normalized component `a0` to `(a0+1)/2 * 2.0` m/s. Low level maps `a1` to `a1 * 0.25` rad/s. High level maps `a1` to a relative heading offset `a1 * max_heading_offset_rad` (default π), added to the current **measured** heading and wrapped. In common East/North coordinates positive heading and yaw changes are counterclockwise: `+1` means left and `-1` means right. The shared `heading-p-v1` controller computes `error = wrap(desired - measured)` and `yaw = clip(0.5 * error, -0.25, +0.25)` rad/s. Its output and saturation are logged in `heading_control`. The controller uses the same gain on every backend and issues a feasible physical command; vessel physics remain active underneath.

Backends receive only `(desired_speed_mps, desired_yaw_rate_radps)`. The action decoder in `benchmark_v3.control.actions` can be called by a future PettingZoo wrapper, including its `action_space(mode)` helper. Task, reward, scenario, and observation equations do not branch on action mode.

Example: `python -m bcod_sim.benchmark_v3 train --config configs/benchmark_v3/s0_high_level_ppo.yaml --run-dir runs/my-high-level-run --action-mode high-level`. The low level counterpart uses `s0_low_level_ppo.yaml` and `--action-mode low-level`. The action mode and schema are recorded in the run schema, checkpoint sidecar, and evaluation report. Policy loading checks schema metadata before loading weights; older checkpoints require explicit migration.
