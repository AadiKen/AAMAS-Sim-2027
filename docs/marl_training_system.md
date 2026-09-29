# V3 local MARL training system

The M0 integration uses `NavigationParallelEnv` (PettingZoo ParallelEnv), TorchRL 0.11.1 `PettingZooWrapper`, and BenchMARL 1.5.2 `MappoConfig`/`Experiment`. BenchMARL owns rollout collection, GAE, clipped PPO loss, minibatching, and optimization. The V3 task owns scenario sampling, action conversion, reward, and termination.

## Frozen M0 interfaces

- Two kinematic vessels, no obstacles; one shared actor for homogeneous agents.
- Actor input: each agent's 78-field local observation (`ego-relative-v2`). The actor module input key is `('agents', 'observation')`; its action shape is `[2 agents, 2 controls]`. One agent cannot read the other's actor observation or the global state through its input.
- High-level action: `desired-speed-heading-v1`; first control maps to desired speed, second to desired relative heading. The existing heading controller generates the physical yaw-rate request.
- Critic input: `state`, a separate 89-field global state. It comprises remaining-time fraction, 12 fields for each of four padded agent slots, and 4 fields for each of ten padded obstacle slots. Agent fields are position x/y, heading sine/cosine, surge, yaw rate, estimated velocity x/y, goal x/y, reached, and presence. Obstacle fields are x/y, radius, and presence. Critic input key is `state`; the two agents share its value network.
- Agent group: `{'agents': ['vessel_0', 'vessel_1']}`. The state schema version and SHA-256, actor schema SHA-256, scenario version, and bank hashes are in each run manifest.

`tests/benchmark_v3/test_parallel_env.py` and `test_marl_m0.py` cover PettingZoo API/seed behavior, TorchRL specs, two-agent physics, distinct action alignment, collision reward/termination, and bank isolation. The actual BenchMARL experiment's actor/critic keys were also inspected and saved in `artifacts/marl-m0/wiring-audit.json`.

## Commands

Create the local environment using the commands in [marl_dependency_stack.md](marl_dependency_stack.md). A fresh M0 run is:

```bash
.venv-marl/bin/python -m bcod_sim.benchmark_v3.marl.train train \
  --run-dir artifacts/marl-m0/new-seed11 --seed 11 \
  --frames 48000 --rollout-frames 6000 --envs 10 \
  --minibatch-size 400 --minibatch-iters 45
```

Contract and saved-result commands:

```bash
.venv-marl/bin/python -m bcod_sim.benchmark_v3.marl.train check
.venv-marl/bin/python -m bcod_sim.benchmark_v3.marl.train summarize \
  --run-dir artifacts/marl-m0/qual48-seed22
```

After all seeds and the configuration are frozen, evaluate one selected policy once:

```bash
.venv-marl/bin/python -m bcod_sim.benchmark_v3.marl.train evaluate-selected \
  --run-dir artifacts/marl-m0/new-seed11
```

The test command refuses to overwrite an existing test result. Runs save manifests, resolved BenchMARL configuration, step-zero, periodic BenchMARL checkpoints, best-dev, latest actor, final actor, dev evaluations, test evaluation, and short per-agent traces. The installed BenchMARL `Experiment.state_dict()` omits optimizer state; these checkpoints are **not** validated optimizer-continuous resume points. The current CLI does not expose a resume command. This stage-specific CLI supports only kinematic M0 and the high-level action mode; broader backend and agent-count selection is deferred because the M0 qualification gate failed.

M0 has **not** qualified for downstream M1. See [marl_m0_validation.md](marl_m0_validation.md) for the frozen experiment and stop decision. BCOD, Pyquaticus, and HoloOcean MARL were not run.
