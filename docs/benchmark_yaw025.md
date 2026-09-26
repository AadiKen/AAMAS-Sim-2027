# Shared yaw ceiling: 0.25 rad/s

## Contract and scope

`BenchmarkConfig.max_yaw_rps = 0.25`. Every adapter computes desired yaw rate as normalized action 1 times this value. Actions −1, −0.5, 0, +0.5, +1 request −0.25, −0.125, 0, +0.125, +0.25 rad/s, respectively. Positive yaw is counterclockwise in East/North coordinates. Observation field 5 is also normalized by the new ceiling. Existing checkpoints used a different action/observation scale; use fresh runs.

The ceiling was selected to lie within the sustainable yaw control envelope of the benchmark vessels, rather than at the Otter's absolute actuator limit. Transient response and coupled speed capability can differ. HoloOcean remains unqualified and is explicitly excluded from the immediate BCOD/Pyquaticus campaign.

BCOD retains measured-rate PI control, native allocation, validated Otter coefficients, actuator limits and lag. Pyquaticus's native feedback/PID inversion and HoloOcean's feedback controller already read the shared ceiling. No physics, reward weights, RL architecture or task geometry changed. Scripted navigation now permits full normalized yaw, preserving its previous physical 0.25 rad/s turning limit.

## Native qualification

Each backend ran 30 s responses over surge {0.5, 0.8, 1.0} × yaw {−1, −0.5, 0, +0.5, +1}, plus the independent surge sweep and a low-speed coupled case. Final window: 5 s. Tolerances remain 0.05 rad/s yaw, 0.20 m/s surge and 0.03 rad/s symmetry. Monotonicity compares final yaw rate across increasing yaw commands, not transient overshoot. Normal-case persistent saturation is rejected if any of the last 25 samples saturates at surge 0.8, yaw ±1.

| Backend | Full yaw at surge 0.5 | Full yaw at surge 0.8 | Full yaw at surge 1.0 | Full qualification |
|---|---:|---:|---:|---|
| BCOD full | ±0.2503 | ±0.2511 | ±0.2518 | PASS |
| BCOD reduced | ±0.2503 | ±0.2511 | ±0.2518 | PASS |
| Pyquaticus | ±0.2503 | ±0.2503 | ±0.2500 | FAIL: coupled speed at surge 1.0 |
| HoloOcean | Pending Linux | Pending Linux | Pending Linux | NEEDS_NATIVE_HOST |

All three runnable backends pass yaw tracking, correct sign, positive/negative symmetry, monotonic yaw response, coordinates, timing and all six scripted navigation scenarios. No final-window saturation occurs at surge 0.8, yaw ±1. Maximum final-window yaw peak-to-peak variation across trials is 0.00136 rad/s for BCOD and 0.000153 rad/s for Pyquaticus; measured trajectories remain finite and settled.

At full throttle and full yaw, Pyquaticus tracks yaw but saturates and reaches **1.796919 m/s** against 2.0 m/s requested: **0.203081 m/s error**, exceeding the unchanged 0.20 m/s speed tolerance. This is reported as a failed metric; neither physics nor tolerances were changed to hide it. BCOD reaches approximately 2.00049 m/s in that case without final-window saturation.

## Training gate

BCOD full/reduced are ready for a fresh short training smoke test. Pyquaticus meets the revised yaw gate but fails the strict aggregate control gate at the maximum-throttle corner. The matched BCOD/Pyquaticus training check is pending a decision to accept this coupled speed limitation as a physical difference. No new training was started, and no long training is recommended on the basis of these results. HoloOcean is explicitly excluded until native Linux qualification.

When that decision is made, use the same seed 11, total joint environment steps 1000, update interval 250, checkpoint interval 500 and one environment for each backend, in fresh unique directories. Expected optimizer updates: 4. Verify manifest/run IDs, monotonically increasing step counts, checkpoint optimizer counts, and single-run JSONL data before interpreting learning. Such a short run checks execution and initial behavior; it cannot establish that collision-heavy learning is eliminated.

## Reproduction and tests

```sh
PYQUATICUS_BENCHMARK_PYTHON=.venv-pyquaticus/bin/python .venv/bin/python -m pytest tests/benchmark -q
.venv/bin/python -m bcod_sim.benchmark.qualify --sim bcod --output artifacts/benchmark-yaw025/bcod.json
.venv/bin/python -m bcod_sim.benchmark.qualify --sim bcod-reduced --output artifacts/benchmark-yaw025/bcod-reduced.json
.venv/bin/python -m bcod_sim.benchmark.qualify --sim pyquaticus --pyquaticus-python .venv-pyquaticus/bin/python --output artifacts/benchmark-yaw025/pyquaticus.json
# On the Linux HoloOcean host:
.benchmark-deps/holoocean-linux/venv/bin/python -m bcod_sim.benchmark.qualify --sim holoocean --output artifacts/benchmark-yaw025/holoocean.json
```

Benchmark tests: **51 passed in 26.73 s**, including native Pyquaticus tests. `git diff --check` passed. Full repository suite was not repeated for this configuration change; the preceding audit records its full-suite results separately. New JSON artifacts preserve the complete traces; historical 0.5 rad/s traces remain in `artifacts/benchmark-audit` and are not overwritten or relabeled.
