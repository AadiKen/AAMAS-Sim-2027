# BCOD PHYSICS FOUNDATION UPGRADE RESULT

## Architecture

- physics component interface: `PhysicsComponent` with resolve-time configuration, resettable runtime context, provider-independent environment input, and `WrenchResult` in plant-origin body FRD
- authoritative dynamics path: unchanged single `Plant6` RK4 path for engine, validation, web, and RL
- force accounting: explicit rigid Coriolis, added-mass Coriolis, linear damping, nonlinear damping, hydrostatics, crossflow, current, wind, waves, propulsion, wake, contact, and manual wrench ledger terms; ledger balance is asserted
- deterministic replay: exact component repeatability and stable seeded wave/uncertainty substreams tested

## Reference-point transforms

- supported transforms: points/CG/CB, spatial matrices, rigid inertia, added mass, hydrostatic stiffness, and force/moment translation; actuator and aerodynamic moments use the same canonical lever-arm convention
- tests: round-trip point, parallel-axis inertia, wrench translation, matrix symmetry, hydrostatic transform, and actuator signs pass

## Damping

- linear: diagonal compatibility model and full 6x6 matrix
- nonlinear: diagonal quadratic coefficients
- coupled: explicit configurable polynomial/cross terms by output axis and velocity factors
- relative-flow semantics: translational damping and crossflow consume vessel-minus-water velocity after one NED-to-FRD conversion

## Hydrostatics

- coupled matrix: full symmetric semidefinite 6x6 model with explicit equilibrium, eigenvalues, conditioning inputs, validity envelope, reference point, and provenance
- mesh buoyancy: exact triangle/plane clipping, closed-polyhedron volume/centroid, hashed watertight mesh, one gravity/buoyancy authority
- equilibrium solver: bracketed displaced-mass solve with residual artifact
- local linearization: central-difference 6x6 G derivation with perturbations, residuals, geometry hash, and equilibrium hash
- lookup backend: not implemented

## Cross-flow

- strip model: generic midpoint-station Hoerner model, constant-section resolver, horizontal and optional vertical strips, water-relative velocity
- MSS component parity: PASS; maximum hydrostatic error 1.13687e-13 and crossflow error 1.42109e-14

## Added mass

- full matrix: symmetric full 6x6 support retained
- conditioning: total effective-mass positive definiteness, eigenvalue range, and condition-number checks/reports
- reference transform: canonical spatial transform from declared source point; Coriolis is always derived from the resolved runtime matrix

## Propulsion

- thrust maps: linear, piecewise-linear/lookup, and polynomial with explicit extrapolation policy
- lag: first-order thrust and RPM response
- deadband: thrust and RPM deadbands
- rate limits: thrust, RPM, azimuth, and rudder mechanisms supported by actuator types
- local inflow interface: RPM propulsor accepts local FRD water velocity without changing engine stepping semantics

## Environment

- canonical field API: provider-independent `WorldSample` exposes water/air velocity, density, surface elevation, orbital velocity/acceleration, depth, and weather observables
- current: uniform, spatial-linear, and sinusoidal time-varying fields; acts through water-relative damping/crossflow
- wind: apparent-wind coefficient-table loads with configurable areas/reference height
- regular waves: deep-water linear/Airy elevation, velocity, and acceleration
- irregular waves: deterministic seeded JONSWAP and Pierson-Moskowitz component synthesis
- wave loading: explicitly simplified `kinematic_wave_drag`, not diffraction/radiation seakeeping

## Payloads

- mass/CG/inertia: immutable installation updates total mass, CG, inertia by parallel-axis theorem, and fingerprint
- hydrostatic recomputation: updated mass is accepted by the mesh equilibrium solver; compiled vessels must resolve a new equilibrium/hydrostatic artifact rather than mutate the base vessel

## Uncertainty

- supported distributions: normal, uniform, triangular
- deterministic resolution: stable parameter-path substreams record definition, master seed, substream seed, and sampled value

## CFD

- supported fitted terms: surge/sway/heave/roll/pitch/yaw linear and quadratic terms plus sway/yaw cross terms where identifiable
- campaign design: deterministic excitation selection for requested targets
- identifiability checks: design-matrix rank and condition number; rank-deficient requests fail closed

## MSS parity-v2

- approximate material terms: 0
- unrepresentable material terms: 0
- Track A: NEAR PARITY
- Track B: NEAR PARITY
- worst discrepancies: position 1.63131e-9 m; orientation 2.95756e-6 deg; linear velocity 2.33388e-9 m/s; angular rate 8.11088e-9 rad/s

## Environmental validation

- current: PASS; current-only response generated entirely through relative-flow physics
- wind: PASS; wind-only apparent-flow response and wrench accounting produced
- waves: PASS; regular and irregular responses, heave/roll/pitch RMS, and wave wrench recorded
- combined conditions: PASS; current+wind, wind+waves, and current+wind+waves cases generated separately

## Performance

- baseline: 1,805.03 microseconds/environment-step, scalar deterministic CPU
- upgraded coefficient model: 3,058.11 microseconds/environment-step
- exact mesh hydrostatics: 709.71 microseconds/evaluation for the 12-triangle analytic box
- batched throughput: 331.85 env-steps/s for 100 environments and 338.54 env-steps/s for 1,000 environments using the current scalar loop

## Regression

- Python: PASS, 143 tests
- Phase 13: PASS, 163 checks, 15 accepted skips, 0 unexpected failures
- legacy isolation: PASS, 3 tests
- web build: PASS
- deterministic tests: PASS

## Files changed

- `src/bcod_sim/physics/`
- `src/bcod_sim/dynamics/{reference_points,damping,restoring,crossflow,mesh_buoyancy,environmental,validity,matrices,diagnostics,plant6}.py`
- `src/bcod_sim/actuators/{base,propulsion}.py`
- `src/bcod_sim/world/{fields,waves,world}.py`
- `src/bcod_sim/core/engine.py`
- `src/bcod_sim/config/{models,vessel_compiler}.py`
- `src/bcod_sim/scenario/uncertainty.py`
- `src/bcod_sim/vessel_generation/{payload,campaign_design}.py`
- `src/bcod_sim/logging/physics_report.py`
- `src/bcod_sim/web/runtime_factory.py`
- `src/bcod_sim/data_sources/world_bundle.py`
- `configs/mss-otter-parity-v2.json`
- `tools/{mss_6dof_validation,environment_validation,benchmark_physics}.py`
- `scripts/{run_mss_6dof_validation,run_environment_validation}.sh`
- physics, hydrodynamics, MSS, environment, and performance tests/artifacts

## Known limitations

- Mesh clipping currently supports one convex waterplane contour; general non-convex/multiple contours remain future work.
- Hydrostatic lookup tables are not implemented.
- Current batched benchmark uses scalar loops; GPU/vectorized component execution is not yet implemented.
- The wave-load model is intentionally simplified and does not model diffraction, radiation memory, RAOs, slamming, or planing.
- Detailed propeller-hull interaction and inflow-dependent thrust correction remain deferred.
- Automatic multidimensional heave/roll/pitch equilibrium for asymmetric payloads remains architectural rather than implemented.

## Exact reproduction commands

```bash
./scripts/run_mss_6dof_validation.sh
./scripts/run_environment_validation.sh
PYTHONPATH=src .venv/bin/python tools/benchmark_physics.py
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m audit.run
(cd web_client && npm run build)
```
