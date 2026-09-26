# Four-vessel navigation benchmark

## Current acceptance gate — 2026-09-26

The shared desired-yaw-rate ceiling is **0.25 rad/s**. See [current qualification and short training checks](benchmark_yaw025.md). HoloOcean is explicitly excluded from the immediate BCOD/Pyquaticus smoke runs pending native Linux qualification.

| Simulator | Current status |
|---|---|
| BCOD full/reduced | Corrected rate controller and orientation estimator; requalified at the 0.25 rad/s ceiling (see current results). |
| Pyquaticus | New yaw sweep, coordinates, timing and six routes pass; full-throttle coupled speed narrowly fails the aggregate gate (1.797 vs 2.0 m/s). |
| HoloOcean | Closed-loop adapter prepared; Linux native control, sensing, and collision tests pending. |

Use `python -m bcod_sim.benchmark.qualify --sim ... --output ...` before any new comparative campaign. No long training was started in the audit.

## Common contract

`src/bcod_sim/benchmark/core.py` defines portable seeded JSON scenarios, 47-value observations, normalized surge/yaw actions, reward, collision/goal/timeout rules, and metrics. `runner.py` runs the same shared 47→128→128 actor-critic and Adam optimizer for every backend. The observation has own pose, speed, yaw rate, relative goal, three agent slots, and ten obstacle slots. Actions are normalized desired forward speed and desired yaw rate: `desired_yaw_rate = action[1] * 0.25 rad/s`; positive is counterclockwise in East/North coordinates. Actions −1, −0.5, 0, +0.5, +1 request −0.25, −0.125, 0, +0.125, +0.25 rad/s. Different vessels may have different transient responses. The ceiling was selected to lie within the sustainable control envelope of the benchmark vessels, rather than at the Otter's absolute actuator limit; HoloOcean still requires native verification. The operating square has half-width 50 m, four crossing routes, a 30 m visibility radius, and 6–8 nominal or 8–10 stress circles. One joint environment transition advances four vessels by 0.2 s. Reward is distance progress +20 on goal arrival −50 per colliding vessel −0.01 per step. Fleet success means all four reach within 2 m by 600 steps without collision.

BCOD uses the pinned, SHA-verified 55 kg MSS/Otter artifact at `artifacts/mss-6dof-validation/latest/plant/bcod_otter_parameters.json`: full 6-DOF mass/added mass, damping, hydrostatics, strip-theory crossflow, dual thrusters, and 0.1 s actuator response. The source validation report is `artifacts/mss-6dof-validation/latest/report.json` (execution PASS, model agreement NEAR PARITY). Policy pose/speed comes from native GPS; yaw rate and integrated heading from native IMU; static obstacle detections from native range-limited AbstractEntitySensor. Other vessels share measured GPS positions. Truth is separate for scoring.

Pyquaticus uses native Heron dynamics and circle obstacles, with capture-the-flag events disabled. The isolated worker maps native state to the nominal 30 m zero-noise observation. Desired yaw rate is converted using measured-rate feedback and the pinned native rudder/PID model. The old empirical ×20 heading mapping was incorrect at steady state and has been removed; see the audit for measurements and deviations.

HoloOcean uses the Ocean/OpenWater world, four SurfaceVessels, GPS/rotation/IMU/collision sensors, and static `spawn_prop` spheres at scenario coordinates. Policy position/heading derives from sensor packets; surge/yaw derives from consecutive sensor readings. Other vessels' GPS positions and static-map obstacle range are supplied to the common 30 m model. Force commands use native dual thrusters. The adapter uses documented `act`/`tick` multi-agent API. Its control gains and native sensor/prop behavior must be checked by the Linux smoke and scripted navigation tests before calling **HoloOcean training ready**. The engine can have platform-dependent physics and reset behavior; seeded geometry is deterministic, native dynamics reproducibility is not yet established.

The stress bank currently provides unseen geometry and 8–10 obstacles only. Stronger current/waves, sensor noise/dropout, and ±10–20% plant perturbation are explicitly marked *not comparable* in evaluation output; they are not silently simulated.

## Historical pipeline smoke checks (not the control acceptance gate)

```sh
bash scripts/install_pyquaticus_benchmark.sh
PYQUATICUS_BENCHMARK_PYTHON=.venv-pyquaticus/bin/python .venv/bin/python -m pytest tests/benchmark -q
.venv/bin/python -m bcod_sim.benchmark.runner train --sim bcod --seed 11 --total-steps 50 --num-envs 1 --checkpoint-interval 10 --checkpoint-dir /tmp/benchmark-bcod-smoke --headless
.venv/bin/python -m bcod_sim.benchmark.runner train --sim pyquaticus --pyquaticus-python .venv-pyquaticus/bin/python --seed 11 --total-steps 50 --num-envs 1 --checkpoint-interval 10 --checkpoint-dir /tmp/benchmark-py-smoke --headless
```

Evaluation uses `python -m bcod_sim.benchmark.runner evaluate --sim ... --checkpoint ... --scenario-bank configs/benchmark_scenarios/nominal --episodes 20 --mode nominal --deterministic --output ...`. Training writes full run configuration, JSONL training/episode metrics, and numbered checkpoints. Checkpoint at 50 steps was loaded for both backends and independently evaluated on one fixed nominal episode.

Add `--eta-bar` to any `train` or `evaluate` command for an optional stderr progress bar with estimated time remaining. Training tracks joint environment transitions; evaluation tracks completed scenarios. The bar is off by default and does not change checkpoints or metric files.

| Backend | Raw environment transitions/s | Trainer transitions/s | Short evaluation |
|---|---:|---:|---|
| BCOD Otter full-6 | 34.9 | 21.3 | 600 steps; no collision; 0 fleet success |
| Pyquaticus Heron | 368.9 | 27.4 | 600 steps; no collision; 0 fleet success |

These are historical pre-audit 50-step diagnostic runs on this macOS ARM host. Trainer throughput includes startup, optimization, logging, and checkpoint writing. The old contract-mock throughput is not a simulator result.

## Linux HoloOcean installation and launch

On an Ubuntu 24.04 x86_64 host with OpenGL 3+ and several GB free, first link the host's GitHub account to Epic Games to access the [licensed HoloOcean source](https://byu-holoocean.github.io/holoocean-docs/v2.3.0/usage/installation.html). From this repository root, the single setup command is:

```sh
bash scripts/setup_holoocean_linux.sh
```

The script installs system libraries, creates `.benchmark-deps/holoocean-linux/venv`, checks out HoloOcean `v2.3.0`, records its resolved commit, installs the client and BCOD package, installs the matching Linux Ocean world, and launches four vessels for one tick. It fails clearly if GitHub/Epic access, the matching package, GPU/OpenGL, or Python 3.12+ is unavailable. Set `PYTHON_BIN` and `HOLOOCEAN_REF` only when using a different supported Python or HoloOcean release. Run subsequent HoloOcean commands with `.benchmark-deps/holoocean-linux/venv/bin/python`. Do not use the obsolete PyPI `holoocean==0.5.8` placeholder; it has no usable `make()` implementation.

## Historical campaign command templates — DO NOT LAUNCH until audit gate passes

These templates are retained for reference, not a recommendation to launch. All backends must use the same optimizer-update interval independently of their checkpoint schedule. The audit acceptance blockers must be resolved first.

```sh
.venv/bin/python -m bcod_sim.benchmark.runner train --sim bcod --seed 11 --total-steps 1000000 --num-envs 1 --update-interval 250 --checkpoint-interval 100000 --checkpoint-dir runs/benchmark-bcod-full --headless --eta-bar
.venv/bin/python -m bcod_sim.benchmark.runner train --sim bcod-reduced --seed 11 --total-steps 1000000 --num-envs 1 --update-interval 250 --checkpoint-interval 100000 --checkpoint-dir runs/benchmark-bcod-reduced --headless --eta-bar
.venv/bin/python -m bcod_sim.benchmark.runner train --sim pyquaticus --pyquaticus-python .venv-pyquaticus/bin/python --seed 11 --total-steps 1000000 --num-envs 1 --update-interval 250 --checkpoint-interval 100000 --checkpoint-dir runs/benchmark-pyquaticus --headless --eta-bar
.benchmark-deps/holoocean-linux/venv/bin/python -m bcod_sim.benchmark.runner train --sim holoocean --seed 11 --total-steps 1000000 --num-envs 1 --update-interval 250 --checkpoint-interval 100000 --checkpoint-dir runs/benchmark-holoocean --headless --eta-bar
```

API references: [HoloOcean package installation](https://byu-holoocean.github.io/holoocean-docs/v2.3.0/usage/installation.html), [multi-agent act/tick and props](https://byu-holoocean.github.io/holoocean-docs/develop/usage/environments.html), [SurfaceVessel thruster control](https://byu-holoocean.github.io/holoocean-docs/develop/agents/agents/surface-vessel-agent.html), [sensor payloads](https://byu-holoocean.github.io/holoocean-docs/develop/holoocean/sensors.html).
