# Navigation V3 training architecture

V3 is the final-task **implementation foundation** for simulator comparisons. The simulator is selected behind one physical-command backend interface; task scoring, observations, evaluation, and stock Stable-Baselines3 PPO remain unchanged. The old `benchmark/runner.py`, `benchmark/ppo.py`, `benchmark/audit_learning.py`, V2 runs, and their reports are retained as **legacy/diagnostic** material and are not final training results. No custom PPO or MAPPO optimizer exists in the V3 path.

Only phases 1–3 are implemented here. The 2,048-step kinematic run is an integration smoke, not the kinematic S0 learnability gate. BCOD reduced and full pass interface/control checks, but no BCOD learning has been launched. S1, PettingZoo, MAPPO, fleet stages, warm starts, Pyquaticus, and HoloOcean remain sequential later phases.

## Frozen contracts

| Contract | Version | Details |
|---|---|---|
| Task | `navigation-v3` | Scenario, scoring, collisions, goals, 600-step deadline, reward, and episode metrics live in the simulator-independent task layer. |
| Observation | `ego-relative-v2` | Fixed 78-element `float32` vector in `[-1,1]`; schema hash in manifest and each policy sidecar. |
| Action | `desired-speed-yawrate-v2` | `Box(-1,1,(2,))`; physical speed = `(a[0]+1)/2 * 2 m/s`; yaw rate = `a[1] * 0.25 rad/s`. No backend rescales it. |

The environment rejects invalid actions. The backend receives physical speed and yaw-rate commands. A policy bundle with a missing or mismatched schema sidecar cannot be loaded by the V3 CLI. Legacy 47/48-element checkpoints therefore require an explicit migration, which is not provided.

### Exact observation order

Fields 0–5 are own measured x and y divided by the arena half-width (50 m), measured surge divided by 2 m/s, measured yaw rate divided by 0.25 rad/s, then sine and cosine of measured heading. Fields 6–8 are goal range divided by the arena diagonal (`2√2 × 50 m`) and sine/cosine of goal bearing **relative to heading**. The goal itself is scenario information; own pose comes from backend measurements.

Fields 9–26 are three nearest-vessel slots, six values each: range divided by visibility (30 m), sine/cosine of relative bearing, relative forward velocity divided by 4 m/s, relative lateral velocity divided by 4 m/s, and a presence mask. Relative world velocity is **neighbor minus ego**, derived from successive measured positions over `dt=0.2 s`; it is rotated into ego axes (forward positive ahead, lateral positive left). At reset, when no measured-position history exists, both relative velocity fields are zero. BCOD vessel positions are GPS-derived measured broadcasts, not scoring truth. Absent slots are all zero.

Fields 27–76 are ten nearest-obstacle slots, five values each: range divided by visibility, sine/cosine of relative bearing, radius divided by visibility, and a presence mask. BCOD centers come from entity detections; radii come from the known scenario obstacle definitions. Absent slots are all zero. Field 77 is `max(0, 600-steps)/600`, the remaining-time fraction. Values are checked for finiteness/shape and clipped to the declared bounds. The schema field list and hash are in `observations.py`; normalization values are in each policy bundle's sidecar.

Scoring uses simulator truth while policy state uses backend readings. `success`, `collision`, and `deadline` each return `terminated=True, truncated=False`. Trainer rollout boundaries do not end an episode. Collision takes precedence over simultaneous arrival. If one vessel collides, all fleet members receive the fixed −50 collision component; `physical_colliders` still identifies who collided.

Reward remains `Phi(s)=-distance_to_goal`, `F=gamma*Phi(s')-Phi(s)`, `gamma=0.99**0.2`, `Phi(terminal)=0`, goal +20, fleet collision −50, step −0.01. Reward is diagnostic for evaluation; checkpoint selection is lexicographic: **success rate**, then **lower collision rate**, then **shorter successful completion time**, then **higher path efficiency**.

## Backends and S0

The kinematic reference backend is a training control, not a paper comparison simulator. Its deterministic integration uses the commanded speed/yaw directly, with no lag:

```text
heading_next = wrap(heading + desired_yaw_rate * 0.2)
x_next = x + desired_speed * cos(heading_next) * 0.2
y_next = y + desired_speed * sin(heading_next) * 0.2
```

BCOD reduced/full use the existing validated Otter plant, actuators, sensors, frame handling, and yaw-rate controller through the same backend calls. One-agent scenarios instantiate one physical vessel. The same V3 task and Gymnasium wrapper run with `--backend kinematic`, `--backend bcod-reduced`, or `--backend bcod-full`; the learner code does not switch algorithms.

`S0-train` randomizes start, goal, path length (18–38 m), and left/right heading error (0.45–2.45 rad), with no obstacles. The fixed `S0-validation` bank has 50 distinct geometries; geometry hashes exclude seed metadata. Every checkpoint is evaluated on this bank. Straight-line Case A is retained only as a scripted integration/control test. The stock PPO policy has no manually set actor bias or distribution.

## Setup, checks, smoke, and future experiment

From the repository root:

```bash
bash scripts/setup_training_v3.sh
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 check --config configs/benchmark_v3/s0_ppo.yaml --backend kinematic --seed 11
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 check --config configs/benchmark_v3/s0_ppo.yaml --backend bcod-reduced --seed 11
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 check --config configs/benchmark_v3/s0_ppo.yaml --backend bcod-full --seed 11
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 scripted-s0 --backend kinematic
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 train --config configs/benchmark_v3/s0_ppo.yaml --backend kinematic --seed 11 --total-timesteps 2048 --run-dir artifacts/training-v3/my-s0-smoke
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 summarize --run-dir artifacts/training-v3/my-s0-smoke
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 evaluate --run-dir artifacts/training-v3/my-s0-smoke --checkpoint artifacts/training-v3/my-s0-smoke/latest.zip
```

The first **real multi-seed kinematic S0 learning experiment** is prepared below. It was not run during this implementation task:

```bash
.venv-training-v3/bin/python -m bcod_sim.benchmark_v3 train \
  --config configs/benchmark_v3/s0_ppo.yaml --backend kinematic \
  --seeds 11 22 33 --total-timesteps 51200 \
  --run-root runs/benchmark-v3/s0-kinematic-51200
```

Summarize each seed separately:

```bash
for seed in 11 22 33; do
  .venv-training-v3/bin/python -m bcod_sim.benchmark_v3 summarize \
    --run-dir "runs/benchmark-v3/s0-kinematic-51200/seed-$seed"
done
```

The V3 dependency lock is `configs/benchmark_v3/requirements.lock`. The setup creates `.venv-training-v3` with Python 3.13 and pinned SB3 2.9.0, Gymnasium 1.3.0, PyTorch 2.14.0, and their dependencies. PPO uses `n_steps=2048`, `batch_size=64`, `n_epochs=10`, learning rate `3e-4`, `gae_lambda=0.95`, clip `0.2`, zero entropy coefficient, value coefficient `0.5`, and global gradient norm `0.5`. The run manifest records the full task/config/dependency versions, code revision, bank hash, and schema. Step-zero, periodic, final, and best-by-task-metric policies are physically saved with schema sidecars.

The observation-only scripted controller solves **50/50** kinematic S0 validation cases with no collisions or deadlines, at a mean 129.6 steps and mean path efficiency 0.989. The stock untrained PPO policy solves **0/50** on the same bank. This establishes that S0 is nontrivial for the default policy yet directly navigable; it does not establish that PPO learns it within a particular step budget.

The next gate is to establish S0 learning across multiple seeds **on kinematic first**. If it does not learn there, stop before BCOD learning. If it learns, transfer identical task, schema, PPO settings, bank structure, budget, and seed set to BCOD reduced, then BCOD full. S1 and multi-agent work follow only after those gates.
