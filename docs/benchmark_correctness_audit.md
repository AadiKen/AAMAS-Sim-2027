# Benchmark correctness audit — 2026-09-26

## Historical decision (before the 0.25 rad/s ceiling update)

**Historical audit at the former 0.5 rad/s ceiling.** Current contract and qualification results are in [yaw ceiling qualification](benchmark_yaw025.md). Old measurements below are preserved as evidence, not the current action contract.
The earlier “TRAINING READY” declarations covered pipeline execution and were too strong. Previous learning curves are not valid controlled comparisons: action semantics and optimizer schedules differed. No plant coefficients, actuator bounds, reward weights, or training task distribution were changed in this audit. A quaternion numerical discontinuity was corrected in the planar integrator.

Native measurements on this macOS ARM host cover BCOD full, BCOD reduced, and the isolated official Pyquaticus Heron worker. HoloOcean changes are code-level only; the engine requires a Linux host. Unit fakes are never counted as HoloOcean qualification.

## Reproduction and artifacts

From the repository root:

```sh
PYQUATICUS_BENCHMARK_PYTHON=.venv-pyquaticus/bin/python .venv/bin/python -m pytest tests/benchmark -q
.venv/bin/python -m bcod_sim.benchmark.qualify --sim bcod --output artifacts/benchmark-audit/bcod.json
.venv/bin/python -m bcod_sim.benchmark.qualify --sim bcod-reduced --output artifacts/benchmark-audit/bcod-reduced.json
.venv/bin/python -m bcod_sim.benchmark.qualify --sim pyquaticus --pyquaticus-python .venv-pyquaticus/bin/python --output artifacts/benchmark-audit/pyquaticus.json
.venv/bin/python -m bcod_sim.benchmark.qualify_scoring
```

On Linux, use the installed HoloOcean interpreter with the same `qualify` command and `--sim holoocean`. No training is involved. Exit 1 denotes a failed/incomplete quantitative gate; exit 2 denotes a missing native runtime. JSON contains individual samples and failures, not just a pass label. `holoocean.json` records the local missing-runtime result. `pyquaticus-legacy-multiplier.json` preserves the pre-fix multiplier measurements.

Control trials use a clear 1000 m square with widely separated vessels to avoid boundary/task interference during a 30 s step response. Normal benchmark geometry and reward are unchanged. The final averaging window is 5 s. Explicit tolerances: speed error ≤0.20 m/s, yaw error ≤0.05 rad/s, left/right mean-rate symmetry error ≤0.03 rad/s; coordinate/47-field reset errors ≤1e-5. These are engineering tolerances, not a claim of identical physics. Samples include measured and truth-derived yaw rates, heading-estimator error, and saturation flags. Summaries include peak, rise to 90%, overshoot, monotonicity, heading change, sign and symmetry. A null rise time means the target was not reached.

## A. CONFIRMED BUGS

| Issue / severity | Location / former behavior | Expected behavior, evidence and fix | Previous-result impact / regression |
|---|---|---|---|
| Yaw intent attenuated — critical | `bcod_adapter.py:step`, one-tick heading offset passed to HeadingSpeedAutopilot | Desired yaw **rate** must feed measured-rate control. Replaced with speed/yaw PI, nominal damping feedforward and existing native fixed-thruster allocation. The old full command created 0.1 rad heading error and about 6 N·m. | Invalidates old steering/learning comparisons. `test_bcod_controller_keeps_native_limits_and_reports_saturation`; native response suite. |
| Body gyro r treated as Euler yaw rate — high | `bcod_adapter.py:_read` | Integrate all three gyro axes with midpoint quaternion exponential using native frame utilities, initialize from reset orientation, derive heading and Euler yaw rate. Obstacle vectors now rotate through the full attitude too. | Old full-6 pose/bearing error grew with tilt. `test_full_imu_orientation_matches_native_kinematics_with_tilt`; native trace heading-error field. |
| Planar quaternion hemisphere jump — critical | `dynamics/plant6.py:_constrain` reconstructed a quaternion with a sign jump at yaw ±π inside RK4 | Keep projected quaternion on the input hemisphere. No coefficient/force changes. Before fix, native heading could stall at the wrap while body r continued; estimator/truth error reached ~3 rad. | Invalidates reduced-fidelity trajectories crossing the wrap. `test_planar_yaw_crosses_pi_without_stalling`. Other pre-existing edits to this file were left intact. |
| Optimizer/checkpoint coupling — critical | `runner.py:train`, `_update` only on checkpoint | Added `--update-interval` (250 default), independent of checkpoint interval. Flush final short rollout once. Log optimizer count on every training/episode row. | Invalidates comparisons with differing checkpoint intervals; 50k/250 vs 50k/5000 meant 200 vs 10 updates. Test compares exact learned tensors with checkpoint schedules 2 vs 3 and common update interval 2. |
| New runs merged with old logs — critical | `runner.py:train` appended logs and overwrote checkpoints | Refuse nonempty directory by default. `--overwrite` archives the entire old directory under a UUID name, then starts a **new** UUID run. Every JSONL row has run ID. | Old appended curves cannot be interpreted as one run. `test_checkpoint_schedule_does_not_change_learning_and_runs_do_not_merge`. |
| Timeout treated as absorbing terminal — high | `runner.py:_update` used `done or truncated` | Explicit continuing-navigation value convention: collision/fleet success bootstrap zero; time limit bootstraps the final observation; returns do not flow into the next reset episode. | Changes value targets; old policy results belong to algorithm v1. `test_terminal_truncated_and_rollout_bootstrap`. |
| Exact visibility boundary affected by floating point — medium | native entity-range rotation + `core.py:observation` | 30.0 m inclusive; allow 1e-8 m numerical tolerance in acquisition/encoding. Diagonal resets previously omitted the 30 m obstacle. | Affects boundary observations, not overall task radius. Tests cover 29.9/30/30.1 m, cardinal/diagonal poses and all 47 fields. |
| Goal crossing could be missed — medium | `core.py:step` inspected goal endpoint only | Goal entry uses the same straight transition segment convention as collision scoring; reached remains latched. | Only pass-through cases differ; reward equation unchanged. Swept-goal and nonsimultaneous completion tests. |
| Reset failure could leak previously created environments — medium | `runner.py:train` initialized before cleanup try | Register each environment with ExitStack before reset. | Operational failure handling; does not change successful rollouts. |

### Controller derivation and physical envelope

BCOD PI feedback uses the validated total mass diagonal: Kp=2M, Ki=M, corresponding to a nominal critically damped pole pair at −1 s⁻¹ after damping feedforward. It uses native speed and native full-attitude Euler yaw rate internally; the policy still receives sensor estimates. Native least-norm allocation is retained, commands clamp to the original individual thruster force bounds, and integrators freeze under saturation. The existing 0.1 s actuator lag and two 0.1 s physics substeps remain.

The pinned Otter force bounds are −66.708 to +119.682 N per thruster, separated by 0.79 m. Maximum opposing-thruster yaw moment is **73.624 N·m**. Nominal yaw drag alone at 0.5 rad/s is **117.975 N·m**, before other coupling/crossflow terms. Thus the specified full yaw command is outside the demonstrated sustained envelope. The audit does **not** relax the ±0.05 tracking tolerance or relabel saturation as a pass.

The script checks constant commands; command reversals and disturbed operation still merit native qualification before extending the operating envelope. Quaternion integration uses 5 Hz sampled gyro endpoints, so it is an estimator, not exact ground truth: maximum observed heading error was about 0.004 rad in these 30 s trials.

## B. CROSS-SIMULATOR CONTRACT VIOLATIONS

| Issue / severity | Evidence | Fix / remaining decision | Regression / impact |
|---|---|---|---|
| Pyquaticus fixed ×20 heading mapping — critical | Exact historical 4 s window reproduces +0.27953 rad/s for target +0.25; by the final 5 s of a 30 s run it reaches +1.080 rad/s. Target +0.5 reaches +2.0944. | Replaced multiplier with native Heron steady rudder/thrust inversion and measured-rate feedback. Invert the pinned heading PID including its derivative/integral state without advancing it; the native dynamics/PID still execute normally. | Extended 30 s ±0.25/±0.5 regression tests reject the old mapping. Old results are not equivalent yaw-rate runs. |
| HoloOcean yaw was open-loop force — critical | `holoocean_adapter.py:step` formerly used `100*yaw_action` without rate feedback. | Closed-loop desired rate minus measured rate now produces differential thrust; speed PI likewise closes speed loop. Existing ±500 N wrapper bounds retained. Gains are **provisional**, awaiting Linux step-response qualification; not presented as calibrated. | Wiring test remains; native response suite is mandatory. Old HoloOcean action semantics differed from both other backends. |
| BCOD full yaw target not attainable — high | Both full/reduced reach ~0.364 at target 0.5 while saturating all 150 steps; coupled speed falls to ~0.834 at target 1.6. | No plant change. Requires an explicit benchmark decision: retain a saturating desired-command contract and classify it as physical capability, or authorize a lower common feasible envelope. This audit does neither silently. | Quantitative gate deliberately fails. Half-yaw command passes; symmetric saturation is visible. |
| Sensor claim overstated — high | Pyquaticus uses state; HoloOcean RotationSensor is perfect orientation and obstacle map is directly range-filtered; BCOD uses ideal GPS/IMU/entity packets with integration error. | Explicit scope: **ideal local kinematics + ideal 360° proximity detections**, no claim of equivalent real sensors, occlusion or noisy sensing. The 47-field semantics match at frozen reset; native source/latency and estimator differences remain declared. A noisy physical-sensing comparison is not qualified. | Frozen observation and boundary tests cover local geometry, not sensor-realism equivalence. Old results cannot support a realistic sensing comparison. |

### Measured control comparison (30 s, final 5 s mean)

| Request (speed, yaw) | BCOD full achieved | BCOD reduced achieved | Pyquaticus achieved | HoloOcean |
|---|---|---|---|---|
| 1.6 m/s, ±0.25 rad/s | 1.600, ±0.251 | 1.600, ±0.251 | 1.474, ±0.250 | Pending Linux |
| 1.6 m/s, ±0.50 rad/s | 0.834, ±0.364, saturated | 0.834, ±0.364, saturated | 1.457, ±0.501 | Pending Linux |
| 0.8 m/s, +0.50 rad/s | 0.649, +0.376, saturated | 0.649, +0.376, saturated | 0.731, +0.501 | Pending Linux |
| Straight speed 0 / .5 / 1 / 1.5 / 2 m/s | 0 / .5 / 1 / 1.5 / 2 | 0 / .5 / 1 / 1.5 / 2 | 0 / .486 / .971 / 1.400 / 1.847 | Pending Linux |

Left/right mean yaw symmetry errors are below 1e-12 rad/s in tested conditions. Full/reduced agree closely after the quaternion fix. A slower surge transient is an allowed physical difference; a fixed multiplier with uncontrolled steady yaw is not.

## C. ACCEPTABLE PHYSICAL DIFFERENCES (within declared scope)

Different acceleration, hydrodynamic drag, turn-induced speed loss, actuator saturation, sample latency and native contact response may remain different. They must be reported, not hidden behind a common normalized action label. BCOD max-yaw saturation remains an **acceptance blocker under the requested numeric gate**, even though its cause is physical rather than a remaining controller sign bug.

### All 47 observation fields

All fields are float32 and clipped to [-1,1]. Common world x=East, y=North; heading 0=East, +π/2=North; positive yaw is counterclockwise. The frozen cardinal/diagonal audit independently predicts all 47 values, including zero slots.

| Indices | Quantity / normalization | Frame |
|---|---|---|
| 0–1 | own x,y / arena half-width | East/North |
| 2–3 | sin/cos heading | CCW from East |
| 4 | forward planar speed / 2 m/s | heading projection |
| 5 | heading rate / max_yaw_rps (currently .25 rad/s) | CCW world yaw |
| 6–7 | goal minus own position / full arena width | East/North |
| 8–16 | 3 nearest agents, each: range/30, sin bearing, cos bearing | horizontal center range; bearing relative to own heading |
| 17–46 | 10 nearest obstacles, same triples | horizontal center range; bearing relative to own heading |

Agent slots are ordered by center distance; exact ties retain source order (scenario/name order). Obstacle ties retain scenario order. Missing slots are [0,0,0]. Radius affects collisions but **is not represented in observations**. FOV is 360°, no occlusion, no line-of-sight requirement. Nominal noise and dropout are zero. Boundary uses center range, never hull/surface range. Tests verify 29.9 and 30 m visible, 30.1 invisible. BCOD acquisition is a native 3-D entity range test; after nonzero heave, its boundary can differ slightly from the horizontal benchmark boundary. This remains a declared sensor-model difference, not proven exact equality for tilted/heaving states. Reset geometry (zero heave) matches.

### Source/latency matrix

| Property | BCOD full | BCOD reduced | Pyquaticus | HoloOcean | Equivalent? |
|---|---|---|---|---|---|
| Action 0 semantics | target speed, PI+drag feedforward | same | native target speed PID | target speed PI | Meaning yes; Holo native unverified |
| Action 1 semantics | target Euler yaw rate, PI | same | rate feedback → native PID heading input | rate feedback → force difference | Meaning yes; Holo gains pending |
| Surge response | ~target by final window | same | slower native Heron acceleration | unknown | Different physics |
| Yaw response | tracks .25; saturates .5 | same | tracks .25 and .5 | unknown | Full envelope no |
| Position source | native GPS packet | same | native state range model | GPSSensor | Ideal values comparable |
| Heading source | initialized quaternion + 3-axis IMU integration | same, planar reduction | native state | RotationSensor | Estimator vs perfect angle |
| Surge observation | GPS horizontal velocity projected on heading | same | native speed | GPS difference over .2 s | Instantaneous vs interval mean |
| Yaw observation | full attitude transformed gyro | same | heading difference/.2 | heading difference/.2 | Instantaneous vs interval mean |
| Obstacle source | native AbstractEntitySensor then full attitude transform | same | scenario map center | scenario map center | Ideal detections; BCOD 3-D range caveat |
| Agent source | broadcast GPS positions | same | native state positions | broadcast GPS positions | Ideal/no communication latency |
| Collision physics | native 1 m sphere + static spheres | same | circles; agent contacts counted; obstacle response next step | native hull mesh + spawned spheres | Holo mismatch unresolved |
| Goal logic | common swept 2 m disk + latch | same | same | same | Yes at harness layer |
| Step duration | .2 s, 2×.1 | same | tau=.2, speedup=1 | 10×.02 engine ticks | BCOD/Py measured; Holo intended |
| Sensor latency | end-of-step packets; midpoint gyro integration | same | end state; yaw interval mean | end packets; velocity interval mean | Not identical phase |
| Noise | zero | zero | zero | defaults zero | Nominal only |
| Determinism | seeded geometry/replay | same | seeded native reset | seeded geometry only; native engine untested | Holo pending |

At reset, speeds/rates are zero, BCOD orientation comes from the known reset orientation, other headings come from native state/sensors, and relative goals/proximity derive from the same geometry. No observation is a real-world sensor guarantee. HoloOcean currently also scores position from GPS and heading from RotationSensor; this is equivalent to noiseless truth only under the zero-noise configuration. A future noisy configuration must separate scoring truth before use.

### Timing, collision and goal audit

BCOD native frame times and Pyquaticus native `current_time` match 0.2/2/20/120 s after 1/10/100/600 transitions. HoloOcean's diagnostic clock is wrapper-derived, not independent evidence of engine time; Linux must verify actual ticks. All controllers are updated once per .2 s; BCOD/Holo physics substeps do not provide extra policy actions.

Controlled straight contact: BCOD first native and common sampled obstacle contact both occur at 6.0 s, x≈−2 m, speed≈0 after native contact resolution. Vessel contact is also 6.0 s; moving hull x≈−1.977 m after pair resolution, speed≈0.510 m/s. Pyquaticus common scoring and post-step overlap occur at 12.4 s, x≈−1.844 m, speed≈0.795 m/s. Instrumenting the native obstacle-check path places its actual rollback/rotation response at **12.6 s**, x≈−2.003 m, speed=0: one step after the harness has already terminated. Native agent contact counting occurs at 12.4 s. The qualification deliberately advances the raw adapter once after common termination to observe that late native response; normal benchmark runs do not. Native agent collision counting does not provide the same physical response as BCOD. No full-step early-native-contact discrepancy was observed in these tests, but substep contact timing/velocity immediately before contact is not exposed by this audit. HoloOcean's mesh and sphere z/diameter/collision behavior require native measurement; equal 1 m hull-circle scoring is not yet verified.

Goal tests cover exact 2 m entry, 2.001 m non-entry, swept crossing, leaving after reaching, nonsimultaneous completion, and collision after one vessel reached. Reached is cumulative; vessels remain active and may later collide. Fleet success requires all reached and zero collision. Native BCOD task uses a remote sentinel goal, Pyquaticus CTF flags/tags/termination are disabled, and HoloOcean native reward is ignored. The common harness owns the returned reward/done flags.

### Shared trainer audit

Algorithm v2 is on-policy shared actor-critic with independent per-environment returns, four-agent mean loss, detached advantages for actor loss, .5 squared-error critic loss, Adam 3e-4, gamma .99, and norm-1 gradient clipping. Gaussian samples use `.sample()` (detached score-function samples), tanh transforms actions, and log density now includes the stable tanh Jacobian. For this score-function estimator that Jacobian is constant with respect to parameters at the sampled raw action; omitting it did not change the old actor gradient, but the reported density was incomplete. No entropy term or PPO is implied.

Update cadence is global joint transitions, independent of `num_envs` cycling and checkpoint cadence. Rollout tails bootstrap from current value; true episode boundaries reset the return; truncations bootstrap final pre-reset observations. Final partial rollout updates once. Checkpoints include model, optimizer, environment-step count, optimizer-update count and CPU RNG states. They are explicitly marked **not resumable**: native environment snapshots, partial autograd rollout and device RNG state are not restored, so no `--resume` is offered. A fresh run is never described as continuation. Deterministic evaluation uses tanh(mean); stochastic evaluation now has explicit `--seed` (default 0).

### Reward audit (unchanged equation)

Prescribed trajectories run through the actual common scorer; these are scoring tests, not simulator performance. Return below is vessel_0, gamma=.99 where discounted.

| Trajectory | Steps | Undiscounted return | Discounted return |
|---|---:|---:|---:|
| Stationary | 600 | −6.000 | −0.998 |
| Straight and succeed (20 m goal) | 91 | 37.290 | 19.482 |
| Progress then collide | 66 | −37.460 | −16.805 |
| Partial progress then timeout | 600 | 4.000 | 6.902 |
| Safe detour and succeed | 113 | 37.067 | 16.335 |
| Reckless faster collision | 33 | −37.130 | −25.241 |
| Long progress then collision (72 m goal) | 158 | **11.620** | **20.710** |

For the 20 m examples the desired ordering succeeds: safe success > partial timeout > stationary > collision. Faster collision is slightly favored in undiscounted return through smaller time cost, while discounting penalizes the earlier collision more. The 72 m case confirms that a fixed −50 penalty does not guarantee negative collision return when prior distance progress exceeds 50 m. It is a different start/goal distance, so this is not evidence that collision beats successful navigation on the same route. Reaching followed by collision can retain the +20 goal bonus by design. Reward redesign, if wanted, must be a separate benchmark decision.

### Scripted qualification

One common observation-based waypoint controller, no native state inputs, solves straight/left/right/S-turn/obstacle-detour scenarios in the three runnable backends. Crossing uses four intersecting offset lanes and a fixed staggered departure schedule, avoiding parked vessels occupying one another's goals. This demonstrates solvability/control authority, not optimal learned collision avoidance. Final crossing times are about 106.2 s (BCOD full/reduced) and 114.4 s (Pyquaticus), within the training 120 s horizon. The qualification configuration permits 130 s but all six successful routes finish within 120 s. Obstacle/turn routes use moderate .8 m/s desired speed and .25 rad/s maximum desired yaw to stay inside the demonstrated Otter envelope.

## D. OPEN QUESTIONS / NEEDS NATIVE HOST TEST

1. **Linux HoloOcean:** run the unchanged full qualification suite. Verify actual yaw sign/gain, speed response, engine timing, GPS/rotation payload freshness, reproducibility, and teardown/process lifetime. Provisional PI gains are not qualified by mocks.
2. **HoloOcean geometry:** determine native hull footprint and physical sphere placement relative to the waterline; compare first native contact with common swept-circle scoring. No claimed resolution without engine evidence.
3. **Full command envelope:** the fixed max_yaw=.5 plus unchanged Otter saturation fails the requested tracking gate. User decision is required before reducing limits or changing the acceptance contract. No comparative long run is recommended.
4. **Sensing scope:** approve the declared ideal/noiseless proximity scope or require physically comparable perception. BCOD nonzero-heave visibility and integrated-heading error remain differences; no occlusion/noisy-sensor equivalence has been demonstrated.
5. **Native substep collision timing:** current recordings have benchmark-step resolution. Precise within-step contact timestamps and pre-contact velocities need finer telemetry before claiming identical physical contact behavior.

Previous policies/checkpoints are preserved. They can be analyzed as historical runs with known violations, but should not be pooled with algorithm-v2/controller-corrected results.

## Validation results

- Benchmark suite with the official Pyquaticus interpreter: **50 passed in 24.53 s**.
- Full repository suite: **492 passed, 2 failed in 916.59 s**. Complete output is saved in `artifacts/benchmark-audit/full-suite.log`. The failures were Capytaine's attempt to create a cache outside the writable sandbox (`test_surface_piercing_lid_and_strip_comparison`) and a missing `CoefficientSurface.moment_reference_frd_m` attribute (`test_episode_randomization_is_deterministic_and_preserves_stability`). Both are outside the benchmark changes.
- Isolated retry of those two tests with `CAPYTAINE_CACHE_DIR=/private/tmp/benchmark-audit-capytaine`: **2 passed in 12.74 s**. No vessel-generation source was changed for this retry. The attribute failure did not reproduce in the fresh process; this does not establish a clean full-suite run or a diagnosed root cause for that failure.
- `git diff --check` and benchmark compilation passed.
- Native qualification: Pyquaticus passed; BCOD full and reduced deliberately failed the full yaw-envelope gate; HoloOcean returned `NEEDS_NATIVE_HOST`. All three runnable backends passed the coordinate, timing and six scripted-navigation checks.
- No long training was launched. Short trainer regression runs used six joint transitions each.
