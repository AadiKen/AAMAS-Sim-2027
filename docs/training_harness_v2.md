# Training Harness V2: Navigation Case A

V2 is an isolated Stable-Baselines3 PPO harness. The legacy custom actor–critic remains available under `bcod_sim.benchmark`; V2 does not call it. The first task is a **single physical BCOD-reduced vessel** with no obstacles and a straight-ahead goal. This is the only V2 backend admitted for this milestone. Fleet training, Pyquaticus, and HoloOcean are future adapters.

## Reproducible setup and bounded smoke

From the repository root:

```bash
bash scripts/setup_training_v2.sh
.venv-training-v2/bin/python -m bcod_sim.benchmark_v2 check --config configs/benchmark_v2/case_a_sb3.yaml --seed 11
.venv-training-v2/bin/python -m bcod_sim.benchmark_v2 train \
  --config configs/benchmark_v2/case_a_sb3.yaml \
  --run-dir artifacts/training-v2/my-smoke-2048-seed11 \
  --total-timesteps 2048 --seed 11
.venv-training-v2/bin/python -m bcod_sim.benchmark_v2 summarize \
  --run-dir artifacts/training-v2/my-smoke-2048-seed11
```

The setup script uses Python 3.13 and `configs/benchmark_v2/requirements.lock`, then installs this repository editable into `.venv-training-v2`. V2 was developed against Stable-Baselines3 2.9.0, Gymnasium 1.3.0, and PyTorch 2.14.0. The config contains a **prepared** 51,200-step setting; that campaign is outside the milestone. The smoke command explicitly caps work at 2,048 environment steps: one rollout, 10 PPO optimization epochs.

The later learning experiment is prepared with the following command. It was **not** run as part of the integration milestone:

```bash
.venv-training-v2/bin/python -m bcod_sim.benchmark_v2 train \
  --config configs/benchmark_v2/case_a_sb3.yaml \
  --total-timesteps 51200 --seed 11 \
  --run-dir runs/benchmark-v2/case-a-sb3-seed11
```

`check` runs both Gymnasium and SB3 environment checkers and the deterministic 190-step straight-line controller baseline. Run it before each training host's first experiment.

## Task and action contract

The versioned task is `navigation-v2`. Each episode has one learning and physical vessel. A successful arrival, collision, or 600-step task deadline returns `terminated=True`; the deadline is a task failure with finite-horizon reward, **not** a Gymnasium time-limit truncation. The last observation is a `float32` vector of 48 finite bounded fields: the qualified 47 navigation/sensor fields plus remaining-time fraction. Scoring reads simulator truth, while the policy observation reads the sensor frame. The reward contains potential shaping with terminal potential zero, the fixed goal bonus, collision penalty, and step penalty; see the run manifest for the exact values.

Observation order and normalization are defined by `OBSERVATION_FIELDS` in `observations.py`: position east/north divided by arena half-width (2), sine/cosine of measured heading (2), measured surge and yaw rate divided by their maxima (2), goal offset divided by twice the half-width (2), three nearby-vessel slots of normalized range and sine/cosine bearing (9), ten obstacle slots of the same three values (30), then remaining-time fraction (1). Empty slots are zero. GPS, IMU, and entity-detection readings provide policy state; simulator truth is used only for collision and reward scoring. Values are clipped to the declared `[-1,1]` observation bounds after finite/shape validation. This schema is `navigation-v2-48`; 47-input checkpoints cannot be loaded into it.

PPO samples a symmetric two-component `Box(-1,1)` action. The environment maps it exactly once:

```
speed_mps = (action[0] + 1) / 2 * 2.0
yaw_rate_radps = action[1] * 0.25
```

Thus `-1` is stopped, `0` is 1 m/s, and `+1` is 2 m/s; yaw is ±0.25 rad/s. Invalid/nonfinite actions raise an error. PPO owns its normal distribution and action clipping at the Gymnasium action-space boundary. There is no legacy latent-tanh density or hidden adapter throttle clamp in V2.

The reference scenario is a 40 m straight-line Case A. The validation bank has ten **distinct** starts, distances, and headings. Its geometry hashes are recorded in the manifest; the bank is not ten copies of one geometry with different seeds. Deterministic evaluation uses a separate environment and restores Python, NumPy, and PyTorch RNG state. It runs at step zero and each configured evaluation point, writes per-episode trajectories, and never updates the model.

## PPO configuration and run records

`configs/benchmark_v2/case_a_sb3.yaml` is the resolved configuration source: `MlpPolicy`, one environment, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `learning_rate=3e-4`, `gamma=0.99**0.2`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0`, `vf_coef=0.5`, `max_grad_norm=0.5`, seed 11. The implementation uses the actual SB3 PPO update and optimizer; see [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html).

Each new run directory must not already exist. It contains `manifest.json` (code revision, task/action/observation versions, reward, gamma, config, seed, bank hashes, library versions), `config.resolved.yaml`, `dependency-versions.json`, `status.json`, `episodes.jsonl`, `logs/progress.csv`, `evaluations/`, `trajectories/`, `checkpoints/`, `latest.zip`, and `best_by_evaluation_reward.zip`. `status.json` records requested and actual **environment steps**, vector steps, agent transitions, rollout count, and PPO optimization epochs. For one environment and one agent, the first three step counts are equal. `best_by_evaluation_reward.zip` is chosen by validation-bank mean reward, not training return.

Inspect a completed run with:

```bash
.venv-training-v2/bin/python -m bcod_sim.benchmark_v2 summarize --run-dir RUN_DIR
.venv-training-v2/bin/python -m bcod_sim.benchmark_v2 evaluate --run-dir RUN_DIR
```

`evaluate` uses the saved latest checkpoint by default and can accept `--checkpoint PATH`. It writes a fresh evaluation record. A checkpoint can be continued with:

```bash
.venv-training-v2/bin/python -m bcod_sim.benchmark_v2 resume \
  --run-dir RUN_DIR --additional-timesteps 2048
```

Resume restores the model and optimizer from `latest.zip`, then resets the physical environment at the continuation boundary. `continuations.jsonl` explicitly records that boundary. Additional steps must be a multiple of `n_steps`. Interrupted runs retain `latest.zip` and `status.json` with the actual work completed; status never claims requested steps were completed.

V2 provides a narrow, testable environment and reporting boundary. A future MAPPO adapter would implement the same versioned task/backend contract and its own explicit multi-agent action/observation spaces. It is deferred until the single-agent PPO milestone is measured.
