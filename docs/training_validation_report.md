# V3 training validation: bounded initialization study

**Status: blocked before simulator comparison or paper freeze.** The frozen kinematic S1 training recipe did not generalize reliably across three seeds on its first untouched final bank. This is a scientific comparison blocker, not evidence of a task, reward, or physics defect. The complete phase ledger is [training_stack_completion_status.md](training_stack_completion_status.md).

## Protocol and artifacts

The task, reward, observation and action contracts, SB3 PPO Config D, and 51,200-transition maximum were unchanged. Checkpoints were selected only by S1-dev using success, collision, deadline, completion time, then path efficiency. PPO initializations copy compatible policy weights into a fresh model and optimizer; no optimizer state is inherited from S0 or behavior cloning. All S1 runs use fresh training seeds 77, 88, and 99. Step-zero, periodic, latest, best-dev, and final checkpoints, selection history, optimizer-containing SB3 bundles, evaluation reports, and manifests are under [s1-qualification](../runs/benchmark-v3/s1-qualification/).

The original inspected 50-case bank is S1-dev (hash `3666670b1d6bddfe6144e7cd24b7f7777cc61dff06b88aac4f820e37d683f561`). The previously evaluated 100-case bank (hash `0aef5f73abdb32ccb3417719d4166f75d7ed906bebb510ce9e687ec383e7d705`) was treated as development evidence for recipe selection. A new 100-case final bank, hash `9932b07c41f2701e394a86e3dfd9d7a03c45601149f1de5ac1daa55bb0a38e7f`, was frozen before initialization experiments. All 100 geometries were unique, valid, and disjoint from development and training geometries. The scripted controller solved 98/100; its two failures remained geometrically feasible and were retained. The final bank was not used for checkpoint selection or recipe selection.

The S0 source was [seed 11 best checkpoint](../runs/benchmark-v3/s0-kinematic-51200/seed-11/best-by-task-metric.zip), a qualified S0 policy. The generic demonstration collector recorded observation/action pairs from the scripted controller on S1 training scenarios. The 200-episode dataset held 32,784 transitions; its BC-only model reached 13/50 on S1-dev. The 1,000-episode dataset held 163,895 transitions; its BC-only model reached 22/50 on S1-dev. The policy branch was supervised; the critic was not trained by BC. The latter BC1000→PPO recipe was selected from development evidence before its first final-bank evaluation.

## Results

Successes are counts out of 100; collision and deadline counts are separate. The same final bank was used for read-only comparison after the BC1000 recipe was frozen. Random-start comparison checkpoints were archived before this study; their seed set differs from the initialization runs.

| Initialization and training | Seeds | Development-era bank success | Final-bank success | Final-bank collisions | Final-bank deadlines |
|---|---|---|---|---|---|
| Random Config D PPO | 44 / 55 / 66 | 88 / 96 / 81 | 93 / 95 / 84 | 7 / 4 / 14 | 0 / 1 / 2 |
| S0 checkpoint → PPO | 77 / 88 / 99 | 77 / 70 / 78 | 82 / 62 / 77 | 11 / 38 / 22 | 7 / 0 / 1 |
| BC200 checkpoint → PPO | 77 / 88 / 99 | 91 / 91 / 54 | Not evaluated for selection | — | — |
| BC1000 checkpoint only | — | 22/50 S1-dev | 49 | 51 | 0 |
| **BC1000 checkpoint → PPO (frozen recipe)** | **77 / 88 / 99** | **89 / 86 / 90** | **78 / 90 / 89** | **17 / 10 / 11** | **5 / 0 / 0** |
| Scripted/geometric controller | — | — | 98 | 2 | 0 |

The frozen BC1000→PPO recipe averaged 85.7% final-bank success and had a 78% worst seed. It failed the earlier S1 qualification threshold (at least 85% in every seed and 90% mean). From-scratch PPO on the same final bank averaged 90.7%, but seed 66 scored 84%; the seed sets are not paired and this is not a basis for claiming a method improvement. S0 warm start was worse. More BC data improved BC-only dev performance from 13/50 to 22/50 and made PPO development results less variable, but it did not provide stable final-bank generalization. The scripted controller's high success shows the task remains feasible, while its two failures demonstrate it is not a perfect oracle.

The scripted teacher retains a route and waypoint index between actions. A diagnostic that replanned from scratch at every action solved only 40/50 S1-dev cases. This helps explain why a feed-forward actor fitted on isolated observation/action pairs may not reproduce the teacher's closed-loop behavior. It does **not** prove the V3 observation is insufficient for a different learned policy.

## Decision and scope

No robust reusable S1 recipe is qualified. Training BCOD-reduced, BCOD-full, MAPPO fleets, or other simulators from one of these policies would conflate policy seed instability with backend effects. The final bank has now been used for the frozen recipe's evaluation; adapting initialization or training based on these results would require a new predeclared experimental protocol and a fresh untouched final bank. That would change the experiment definition and is the current stopping gate.

Implemented and unit-tested here: fresh-optimizer policy warm start, generic observation/action demonstration collection, offline actor behavior cloning, S1 bank isolation, and the existing dev-based checkpoint protocol. The V3 focused suite passes 24 tests. The reusable registries, general lifecycle/resume, PettingZoo API, established MAPPO integration, BCOD S1 qualification, Pyquaticus/HoloOcean native qualification, throughput campaign, and paper artifacts remain unqualified and are not represented as completed.
