# S1 single-agent qualification decision

**Final classification: C. KINEMATIC S1 TRAINING MACHINERY STILL NOT RELIABLE.**

The prescribed gate stopped the pipeline after the corrected Config D and the one permitted Config D2 run both failed the untouched S1-test threshold. S0 regression, BCOD-reduced S1, and BCOD-full S1 were therefore not launched.

## Architecture and frozen contracts

The implementation uses stock Stable-Baselines3 PPO with the Gymnasium V3 single-agent task. The task is `navigation-v3`, observation `ego-relative-v2`, and action `desired-speed-yawrate-v2`. The action maps to commanded speed 0–2 m/s and yaw rate ±0.25 rad/s. The S1 generator is `s1-obstructed-v1`; its 5 layouts and geometry ranges are frozen. The deadline is 600 steps at 0.2 s/step. The step discount is `0.99 ** 0.2`. Reward components remain potential shaping (`Φ=-distance`, zero terminal potential), goal +20, collision −50, and step −0.01. No task, reward, observation, action, network, or backend physics changes were made.

Config D uses 8 independent environments, 256 steps per environment, 2,048 transitions per rollout, batch size 64, 10 epochs, learning rate 3e-4, GAE λ=0.95, clip range 0.2, entropy coefficient 0, value coefficient 0.5, and maximum gradient norm 0.5. Config D2 differs only by a linear learning-rate schedule from 3e-4 toward zero across 51,200 transitions. The last optimizer update occurs at the start of the final rollout, where its rate is 1.2e-5.

The previously inspected 50-case bank is **S1-dev**, SHA-256 `3666670b1d6bddfe6144e7cd24b7f7777cc61dff06b88aac4f820e37d683f561`. The new 100-case **S1-test** bank has SHA-256 `0aef5f73abdb32ccb3417719d4166f75d7ed906bebb510ce9e687ec383e7d705`. Test seed 4027 uses the same generator and a balanced 20 cases per layout. All geometry hashes are unique and disjoint from dev; training samples reject reserved geometries. The [bank manifest](../runs/benchmark-v3/s1-qualification/banks/manifest.json) records generation details. The test bank was frozen before PPO training. Its only pretraining evaluation was the prescribed non-learning validity gate: the scripted controller succeeded in **100/100**, with zero collisions and deadlines; every case passed the geometric validator.

Training saves step zero, each 10,240-step checkpoint, latest, final, and best-dev. Best-dev selection uses S1-dev in this strict order: success count, collision count, deadline count, mean successful completion time, and mean path efficiency; exact ties retain the earlier checkpoint. It does not use shaped reward. Training stops after three scheduled dev evaluations without improvement once 20,480 transitions have elapsed, or at 51,200 transitions. After training ends, best-dev is loaded and evaluated exactly once on S1-test. Test outcomes do not alter training or checkpoint selection. Selection history, reason for each replacement, and all checkpoints are saved with each run.

## Kinematic qualification results

Completion times below are seconds among successful episodes. Clearance is the mean minimum obstacle clearance in metres. Path efficiency is the mean across all 100 episodes, with failures contributing zero. Every run reached the 51,200-transition maximum (25 PPO rollouts, 250 epochs). The selector chose a prior checkpoint where indicated.

| Config | Seed | Best-dev step | Final step | Dev success / 50 | Test success / 100 | Collisions | Deadlines | Mean / median / p90 completion (s) | Mean clearance (m) | Path efficiency | Mean final goal distance (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 44 | 51,200 | 51,200 | 46 | 88 | 12 | 0 | 46.05 / 42.80 / 65.84 | 2.21 | 0.794 | 4.12 |
| D | 55 | 40,960 | 51,200 | 47 | 96 | 4 | 0 | 42.84 / 40.70 / 51.50 | 2.91 | 0.838 | 2.68 |
| D | 66 | 51,200 | 51,200 | 41 | 81 | 18 | 1 | 58.25 / 53.00 / 85.20 | 2.26 | 0.700 | 4.22 |
| D2 | 44 | 51,200 | 51,200 | 47 | 92 | 5 | 3 | 49.05 / 47.00 / 70.72 | 3.33 | 0.789 | 3.63 |
| D2 | 55 | 20,480 | 51,200 | 43 | 88 | 12 | 0 | 38.57 / 38.40 / 43.06 | 2.70 | 0.756 | 5.16 |
| D2 | 66 | 51,200 | 51,200 | 38 | 72 | 28 | 0 | 43.09 / 42.10 / 50.88 | 1.75 | 0.665 | 7.50 |

Config D averaged **88.3%** test success; seed 66 was **81%**, below the required 85% per seed, and the mean was below the required 90%. Config D2 averaged **84.0%**; seed 66 fell to **72%**. The dominant failure was collision (D: 34 collisions across 300 test episodes; D2: 45), with few deadlines. Selected dev success versus test success was 92%→88%, 94%→96%, and 82%→81% for D, and 94%→92%, 86%→88%, and 76%→72% for D2. These are not catastrophic dev-to-test collapses, but the absolute test gate still fails.

## Phase 3B task and reward audit

The 35 failed Config D test episodes were replayed independently. Every scenario passed the geometric validator and the scripted controller succeeded on that same scenario. Termination reason, minimum goal distance, and minimum obstacle clearance were checked per replay. Per-step reward was independently summed from its components with the implemented discount; every scripted success had higher discounted return than its failed learned route. Potential shaping satisfied its discounted telescoping identity (`Σ γ^t F_t = −Φ(initial)`, because terminal potential is zero) in every replay. Success, collision, and deadline bookkeeping agreed with the V3 task. The detailed [audit artifact](../runs/benchmark-v3/s1-qualification/phase3b-audit.json) contains all per-case measurements and return components. No concrete task/reward contradiction was found.

## Backend qualification status and decision

| Backend | Seeds 44 / 55 / 66 | Status |
|---|---|---|
| Kinematic S1 | D and D2 results above | Failed untouched-test gate |
| Kinematic S0 regression | — | Not run; Phase 3B gate says stop after D2 failure |
| BCOD-reduced S1 | — | Not run; blocked by kinematic gate |
| BCOD-full S1 | — | Not run; blocked by kinematic gate |

The smallest isolated blocker is seed-dependent PPO S1 reliability on feasible, correctly scored kinematic scenarios. Learning-rate annealing did not fix it. The task contract and reward are not implicated by the prescribed checks. No further sweep or backend training was started.

## Reproducibility

Run artifacts are under [s1-qualification](../runs/benchmark-v3/s1-qualification/). For each run the exact command was:

```bash
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 qualify-s1 --config configs/benchmark_v3/s1_ppo_stability_d.yaml --backend kinematic --seed SEED --run-dir runs/benchmark-v3/s1-qualification/D-kinematic-seedSEED
```

For D2, add `--anneal-lr` and use `D2-kinematic-seedSEED`. `SEED` took 44, 55, and 66 in fresh directories. The benchmark V3 test suite passed: **22 tests**. The run manifests record dependency versions, schema, bank hashes, PPO settings, and code revision.
