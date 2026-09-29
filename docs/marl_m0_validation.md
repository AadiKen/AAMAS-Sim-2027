# M0 kinematic MAPPO validation

**Result: PARTIAL, not training-qualified.** Local PettingZoo/TorchRL/BenchMARL integration works and MAPPO learns in two seeds, but the frozen configuration does not reliably generalize across three seeds. M1 was not started.

## Protocol and data

M0 uses two kinematic vessels, no static obstacles, and a cooperative fleet objective. Five scenario layouts cover head-on, perpendicular, oblique, merging, and different arrival-time encounters, with rotations and reflections. All 150 fixed scenarios pass the geometry validator, are unique by geometry hash, and are disjoint from the training sampler. The 50-case development bank was used for checkpoint selection; the 100-case test bank was used once per selected policy only after the 48k configuration was frozen.

The M0 training distribution is version `m0-two-vessel-crossing-v1`; the frozen `scenarios.py` source SHA-256 is `94863ebbd1ebb22de197a16d604aa081451c2416a5f7affb8057d855f006f173`. As an on-demand distribution it has no finite bank hash; each sampled training geometry is checked against the reserved fixed-bank hashes.

| Bank | Cases | SHA-256 |
|---|---:|---|
| M0-dev | 50 | `d73df5d79047a5ad369462597343b58567ea6ceffb5ac4b07d8a459a0004af31` |
| M0-test | 100 | `9d96a60bf9985c569e5414efbf7b3b254a8d6b02ba8701cfcbc61cd090e5819e` |

The straight-to-goal controller achieved 34/50 development fleet successes (16 collisions). The geometric controller achieved 47/50 (3 collisions, 0 deadlines). Untrained MAPPO actors achieved 0/50 in each seed, with 21–50 collisions. These baselines establish that M0 is generally solvable and that trained policies improve materially over initialization.

## Frozen BenchMARL configuration

BenchMARL 1.5.2 MAPPO with TorchRL 0.11.1, shared MLP actor, shared centralized critic, continuous two-control high-level actions, CPU device. The resolved configuration is in each `manifest.json`: learning rate `5e-5`, PPO clip `0.2`, critic coefficient `1.0`, entropy coefficient `0`, gradient clipping enabled at `5`, 10 parallel environments, 6,000 collected transitions per rollout, minibatch size 400, and 45 minibatch iterations per rollout. Gamma is `0.99 ** 0.2 = 0.9979919516614258`, matching V3 task shaping. Other network, GAE, and optimizer settings are the installed BenchMARL defaults in the manifest. The bounded initial study used 24,000 transitions; one longer, otherwise identical 48,000-transition study was then run. There was no reward, physics, control, or policy-schema change between studies.

Smoke: a 512-transition CLI run saved and reloaded checkpoints, produced finite diagnostics, ran deterministic evaluation, and generated distinct actions for distinct agent observations. `check_env_specs`, PettingZoo parallel API and seed checks, and six MARL/PettingZoo tests passed.

## Development learning curves

Fleet successes out of 50 cases. The selected checkpoint maximizes development success, then minimizes collisions and deadlines, then completion time, then maximizes path efficiency.

| Transitions | Seed 11 | Seed 22 | Seed 33 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 6,000 | 1 | 0 | 0 |
| 12,000 | 4 | 22 | 1 |
| 18,000 | 21 | 34 | 6 |
| 24,000 | 3 | 35 | 12 |
| 30,000 | 25 | 38 | 16 |
| 36,000 | 11 | 43 | 24 |
| 42,000 | 6 | 47 | 31 |
| 48,000 | 7 | 46 | 40 |

Seed 11 has substantial policy regression after its 30k development peak. Seed 22 learns the task well. Seed 33 improves steadily but remains collision prone.

## Untouched test evaluation of best-dev actors

| Seed | Selected transition | Test fleet success | Collisions | Deadlines | Per-agent goals (0/1) | Mean episode steps | Mean path efficiency | Mean minimum separation (m) | Heading controller saturation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 30,000 | 48/100 | 14 | 38 | 67/69 | 385.39 | 0.316 | 10.13 | 28.4% |
| 22 | 42,000 | 92/100 | 7 | 1 | 96/95 | 161.77 | 0.800 | 8.06 | 12.9% |
| 33 | 48,000 | 77/100 | 20 | 3 | 80/82 | 181.64 | 0.649 | 6.10 | 19.4% |

Mean test fleet success is 72.3%, with a 48% worst seed and 13.7% mean collision rate. This is not a low-collision, reliable multi-seed result. The selected seed-11 policy shows a particularly large deadline rate. M0 is **PARTIAL**: real learning and some generalization, but not qualified for M1.

## Failure diagnosis and gate

The actual BenchMARL experiment exposes actor input `('agents', 'observation')`, critic input `state`, one shared actor, and one centralized critic. TorchRL's actor spec is `[2,78]`; state spec is `[89]`. Tests confirm simultaneous action alignment and a fleet-consistent collision penalty/termination for both agents. The fixed banks are unique and geometrically valid. The geometric baseline succeeds on 47/50 development cases. No direct wiring or task-feasibility defect was found. The remaining evidence points to seed-sensitive training/generalization, including saturation and late policy regression; it does not identify a unique algorithmic cause.

**Stop:** the required multi-seed M0 gate did not pass. Do not run M1, M2, BCOD, Pyquaticus, or HoloOcean MARL on this recipe. A further experiment needs a separately declared protocol and fresh untouched test bank; the present M0-test bank is now used final evidence.

Run records are in `artifacts/marl-m0/qual48-seed{11,22,33}/`; the wiring audit is `artifacts/marl-m0/wiring-audit.json`. Per-episode test metrics and sample per-agent trajectories are saved alongside each run.
