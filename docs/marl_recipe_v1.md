# M0 high-level MAPPO curriculum v1

This is the frozen paper-aligned navigation recipe in `configs/benchmark_v3/marl/m0_highlevel_mappo_curriculum_v1.yaml`. It is a new experiment version; the older M0-test bank is not reused. No simulator physics, reward, high-level controller, or MAPPO loss code was changed.

Frozen recipe SHA-256: `7613978a7295d438ee81d04808c1ebe8c6fd44d9092904d0208f431bba3d21f7`. The M0 curriculum scenario generator source SHA-256 is `3dceff56921507fcfbfc961f1a43996b8b9c55fdeb8cc209c9cb0c7c82238fc0`.

## Information and control boundary

Two homogeneous vessels share one decentralized actor. Each actor invocation reads its own 78-field `ego-relative-v2` observation and emits two bounded actions. The first maps to desired speed `((a0+1)/2)×2 m/s`; the second maps to relative heading `a1×π`. The existing heading controller converts that request to yaw rate using gain `0.5` and clips at `±0.25 rad/s`. The actor does not emit yaw rate or actuator commands. The centralized critic receives only the separate 89-field `fleet-global-state-v1` during training. Its state is not part of the deployed actor input. One shared actor and one shared critic are created by BenchMARL.

An agent marked reached stays in the physical environment. Subsequent policy actions for it are ignored and `(desired speed, desired yaw rate)=(0,0)` is sent to the backend. The fleet episode continues until all goals, collision, or deadline. A physical collision still terminates the fleet and gives every agent the existing collision penalty; actual colliders remain separately logged. Reward and discount are unchanged.

## Installed library and resolved hyperparameters

The optimizer is BenchMARL 1.5.2 MAPPO on TorchRL 0.11.1. TorchRL `PettingZooWrapper` consumes the V3 ParallelEnv. BenchMARL owns rollout collection, GAE, clipped PPO loss, optimizer, and minibatching. `ClipPPOLoss.normalize_advantage=True` is enabled through TorchRL's published configuration attribute; there is no custom loss math. BenchMARL 1.5.2's default MAPPO path does not expose value normalization, so this recipe records it as **not enabled** rather than claiming it is present.

| Setting | Value |
|---|---:|
| Shared actor / shared centralized critic | true / true |
| Actor and critic MLP hidden widths | 256, 256 |
| Activation | Tanh |
| Actor/critic optimizer learning rate | 5e-5 (BenchMARL shared experiment setting) |
| Optimizer | Adam (BenchMARL default), epsilon 1e-6 |
| Gamma | `0.99**0.2 = 0.9979919516614258` |
| GAE lambda | 0.9 |
| PPO clip epsilon | 0.2 |
| Entropy coefficient | 0 |
| Value-loss coefficient | 1.0 |
| Gradient norm clip | 5 |
| Advantage normalization | enabled via TorchRL loss |
| Value normalization | not exposed in installed default path |
| Parallel environments | 10 |
| Rollout | 6,000 transitions, 600 per environment |
| Minibatch size / iterations per rollout | 400 / 45 |
| Device | CPU |

The complete resolved library and experiment dataclasses are saved in every run `manifest.json`. The 10 environments are separate environment instances with independent reset RNG state and episodes: a seeded TorchRL `SerialEnv(10, ...)` reset produced ten distinct local initial observations. The collector uses BenchMARL's stock on-policy path. No SB3 PPO parameters were copied into the recipe.

## Stages and gates

All stages sample randomized geometry, with no static obstacles. The same actor, critic, optimizer, and normalization state persist when a stage changes. The stage is published atomically and read by each environment at reset.

| Stage | Scenario distribution | Promotion | Minimum | Maximum |
|---|---|---|---:|---:|
| A | parallel travel, separate goals, offset paths, mild merges | ≥90% dev fleet success in two consecutive evaluations | 12,000 | 48,000 |
| B | offset and perpendicular crossings, merging, asymmetric arrival | ≥85% dev fleet success and ≤10% collisions in two consecutive evaluations | 12,000 | 48,000 |
| C | full M0 head-on/crossing/merging/asymmetric distribution | final qualification | 12,000 | 48,000 |

Development evaluations occur every 6,000 transitions on 50 fixed cases for the current stage. A/B promotion at the declared maximum fails that seed. Stage C stops at 48,000 stage transitions or after three consecutive nonimproving development evaluations once its minimum is met. Best-dev selection uses fleet success, then fewer collisions, fewer deadlines, lower completion time, and higher path efficiency. The same live BenchMARL experiment carries its actor, critic, and optimizer across stage transitions. The installed BenchMARL checkpoint serialization omits optimizer state, so saved checkpoints are evidence of model/collector state and are not claimed as optimizer-continuous resume points. Only the best Stage-C actor is eligible for the once-only M0-C test evaluation.

The final-stage test gate, declared before training, is mean success ≥85%, every seed ≥75%, mean collisions ≤10%, all seeds materially better than untrained, and no deadline/deadlock-dominated seed. The three fixed seeds are 11, 22, and 33. If M0-A fails, training stops for that seed; no manual promotion is allowed.

## Bank identities and feasibility reference

All 450 fixed geometries are unique and disjoint, and sampled training geometries exclude them. The geometric controller uses the same high-level action/controller path. It solved A-dev 50/50, A-test 100/100, B-dev 40/50, B-test 88/100, C-dev 46/50, and C-test 93/100. These are feasibility references, not imitation data.

| Bank | Cases | SHA-256 |
|---|---:|---|
| A-dev | 50 | `fc17c1056335d138ca60ddf5ba7c9f2d3e929a1458d540636421dc1c79906a6c` |
| A-test | 100 | `e0155c3391ebe51a2a76e4977409dc643322fcee7fb4dbed4c2ede5e8fb8449d` |
| B-dev | 50 | `3cadd357ccbcf581ca0c7dac73928403c3abb663569d3c1001544300988e0edb` |
| B-test | 100 | `b4b9f2a6d3928aba26092b7c9a2643b0491191637b843de36582819e9967d373` |
| C-dev | 50 | `12eaa11f59f1825e35490704e5d9f2c928a19587430c7489f9af21a1054de905` |
| C-test | 100 | `e94e660a9abff6c65d62c34187e4b1b8faf6c8eda8ebbd18d93216d9293b6de3` |

The bank archive is in `artifacts/marl-m0-curriculum-v1/banks/`. An integration smoke with reduced evaluation count exercised A→B→C promotion, continuous checkpointing, and finite BenchMARL losses. The smoke's relaxed thresholds are not the qualification recipe.
