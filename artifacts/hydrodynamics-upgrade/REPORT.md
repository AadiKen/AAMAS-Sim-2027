# BCOD HYDRODYNAMICS UPGRADE RESULT

## 1. Coupled hydrostatics

- model/interface: pure `HydrostaticsModel.evaluate(...) -> WrenchResult`; legacy constant buoyancy and new `LinearHydrostatics` share the single Plant6 restoring authority
- full 6x6 supported: yes, including semidefinite free-floating matrices and off-diagonal coupling
- reference-point transform: one-time Fossen/MSS `H(r)' G H(r)` transform, with source and resolved matrices retained
- validation: finite 6x6 shape, symmetry, eigenvalue reporting, explicit equilibrium, attitude validity envelope, materially negative eigenvalue rejection unless explicitly allowed
- tests: zero/equilibrium wrench, diagonal heave, coupled heave-pitch, restoring sign, reference transform, unstable matrix rejection

## 2. Geometry-based buoyancy

- clipping algorithm: deterministic triangle/plane half-space clipping in body coordinates plus waterplane cap triangulation
- watertight validation: closed two-manifold edge incidence, consistent edge orientation, finite/nondegenerate faces, nonzero volume; no runtime repair
- submerged-volume computation: signed tetrahedral integration over the closed clipped polyhedron
- CB computation: first volume moment of the same tetrahedral decomposition
- equilibrium solver: bracketed bisection on displaced mass
- box analytic error: <= 1e-12 m3/coordinate for dry, half-immersed, and fully submerged box cases
- symmetry tests: roll and pitch mirror tests pass; level lateral moments are zero within 1e-9 Nm
- runtime cost: 709.71 microseconds/call for a 12-triangle box on the validation host
- lookup backend implemented: no

## 3. Strip-theory crossflow

- geometry representation: ordered midpoint stations; constant-section geometry resolves deterministically into stations
- Cd model: Hoerner table with linear interpolation and bounded high-ratio behavior matching pinned MSS
- integration method: midpoint strips; parity vessel uses 20 strips
- relative-flow convention: body-FRD `v_i = v_water-relative + x_i*r`; production engine supplies current + wake + wave-orbital water velocity in NED and transforms per state
- MSS component max error: hydrostatics 1.13687e-13; crossflow 1.42109e-14 across 100 seeded samples
- invariant tests: zero, sway sign reversal, yaw opposition, symmetry cancellation, energy dissipation, water-relative cancellation, and N/2N/4N convergence pass

## 4. MSS parity-v2

- approximate terms remaining: 0
- unrepresentable terms remaining: 0

### Track A

- previous classification: SMALL SYSTEMATIC DIFFERENCE
- new classification: NEAR PARITY
- worst position discrepancy: 1.63131e-9 m (T08)
- worst orientation discrepancy: 2.95756e-6 deg (T07)
- worst linear-velocity discrepancy: 2.33388e-9 m/s (T08)
- worst angular-rate discrepancy: 8.11088e-9 rad/s (T08)

### Track B

- previous classification: MATERIAL DIVERGENCE
- new classification: NEAR PARITY
- worst discrepancies: position 1.14403e-11 m, orientation 2.95756e-6 deg, linear velocity 6.94123e-11 m/s, angular rate 7.65427e-10 rad/s

## 5. Regression

- Python tests: PASS, 136 passed
- Phase 13 audit: PASS, 163 checks passed, 15 accepted skips, 0 unexpected failures
- legacy isolation: PASS, 3 passed

## 6. Performance

- linear hydrostatics cost: 62.76 microseconds/call
- exact mesh hydrostatics cost: 709.71 microseconds/call for the 12-triangle analytic box
- lookup hydrostatics cost if implemented: not applicable; backend not implemented
- strip-theory cost: 121.69 microseconds/call for 20 strips

## 7. Files changed

- `src/bcod_sim/dynamics/restoring.py`
- `src/bcod_sim/dynamics/crossflow.py`
- `src/bcod_sim/dynamics/mesh_buoyancy.py`
- `src/bcod_sim/dynamics/plant6.py`
- `src/bcod_sim/dynamics/diagnostics.py`
- `src/bcod_sim/core/engine.py`
- `src/bcod_sim/web/runtime_factory.py`
- `configs/mss-otter-parity-v2.json`
- `tools/mss_6dof_validation.py`
- `tests/physics/test_hydrodynamics_upgrade.py`
- `tests/physics/test_plant6.py`
- hydrodynamics and parity artifacts under `artifacts/hydrodynamics-upgrade/` and `artifacts/mss-6dof-validation/latest/`

## 8. Known limitations

- Exact mesh clipping currently supports one convex waterplane contour; multi-contour/non-convex waterplanes require loop-aware cap triangulation.
- Exact mesh cost scales with triangle count and is not intended as the final high-throughput RL backend.
- The deterministic lookup-table hydrostatics backend remains future work.
- Linear-matrix hydrostatics is deliberately restricted to its configured roll/pitch validity envelope.

## 9. Exact reproduction commands

```bash
./scripts/run_mss_6dof_validation.sh
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m audit.run
```
