# V3 training stack completion status

Updated: 2026-09-28. Status terms: implemented, unit-tested, API-qualified, training-qualified, paper-qualified, deferred, blocked.

| Phase | Status | Evidence / artifact | Blocker | Next action |
|---|---|---|---|---|
| 0 Contracts | In progress | V3 task, observation, action versions and schema sidecars exist | Scenario, metrics, and manifest versions not yet canonical | Freeze and test all contract identifiers |
| 1 Reusable package | In progress | V3 runner has train/evaluate/check; S1 qualification runner exists | Stage-specific paths and missing registry/API commands | Refactor around registries and public CLI |
| 2 Reproducibility | In progress | Model/optimizer checkpoints, manifests, status and logs exist | Resume and RNG lifecycle incomplete | Add continuation provenance and tests |
| 3 Train/dev/test | Implemented for S1 | `runs/benchmark-v3/s1-qualification/banks/`; final bank hash `9932b07c41f2701e394a86e3dfd9d7a03c45601149f1de5ac1daa55bb0a38e7f` | The final bank was evaluated once after BC1000 recipe selection; further recipe changes cannot use it as untouched test data | Preserve as final evidence; preregister a new bank only under a new protocol |
| 4 Selection/early stop | Implemented for S1 | `qualification.py`; six Config D/D2 runs | Generalize beyond S1 | Extract reusable selector |
| 5 Curriculum/initialization | Implemented and tested, training gate failed | S0→S1 warm start and generic offline BC modules; `runs/benchmark-v3/s1-qualification/` | Frozen BC1000→PPO recipe scored 78/90/89 on untouched final bank; no robust 3-seed recipe | Requires a new predeclared training/validation protocol before further tuning |
| 6 Single-agent ladder | Blocked | Kinematic S1 final test failed; see `docs/training_validation_report.md` | No qualified single-agent recipe | Do not make BCOD simulator comparisons from this policy |
| 7–11 MARL/communication | M0 API and curriculum smoke qualified; new curriculum training gate FAILED | Dedicated `.venv-marl`, BenchMARL 1.5.2/TorchRL 0.11.1, `docs/marl_recipe_v1.md`, `docs/marl_m0_curriculum_results.md`; seed 11/33 failed A, seed 22 failed B | No seed reached M0-C; no learned C-test result | Keep M1 and physical MARL paused; predeclare a new recipe before another training experiment |
| 12–13 Other backends | Deferred | Earlier adapter work outside V3 | Native qualification, Linux HoloOcean access | Prepare adapters and native acceptance |
| 14–18 Paper protocol/evaluation | Deferred | — | Frozen recipe and simulator qualification | Freeze comparison protocol before final tests |
| 19–24 Throughput/API/acceptance/paper | Deferred | — | Earlier gates | Build reports from provenance-bearing artifacts |

Current scientific conclusion: kinematic S1 remains seed-sensitive. The frozen BC1000→PPO recipe failed its first untouched final-bank evaluation. Changing the recipe now would require a new benchmark protocol and untouched bank; continuing to the backend comparison with the failed recipe would confound the simulator comparison. The prescribed stopping condition for an invalid scientific comparison is met. This is not a demonstrated task/reward/physics correctness error.
