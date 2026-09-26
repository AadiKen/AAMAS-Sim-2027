# Spec A maneuvering pipeline

This implementation is opt-in. Existing mesh hydrostatics, equilibrium, BEM,
resistance, and the default V5 runtime remain the baseline. The new surface
supplies **physical fluid-on-hull FRD** Y/N and the maneuvering surge increment
ΔX. It has not passed vessel-level accuracy acceptance.

## Run the scripts

Use the repository virtual environment and activate OpenFOAM 11 for `mesh`, `select`
and `run`. Install `.[sysid]` for output-error optimization, and `.[validation]`
for plots. No command starts CFD implicitly.

```sh
.venv/bin/python tools/spec_a_pipeline.py empirical output/ vessel.yaml
.venv/bin/python tools/spec_a_pipeline.py make output/ vessel.yaml
.venv/bin/python tools/spec_a_pipeline.py select output/ vessel.yaml
# Review mesh_selection.json before running the remaining nine states.
.venv/bin/python tools/spec_a_pipeline.py run output/
.venv/bin/python tools/spec_a_pipeline.py extract output/
.venv/bin/python tools/spec_a_pipeline.py fit output/ vessel.yaml
```

The implementations are plain Python modules in
`src/bcod_sim/vessel_generation/spec_a/`: `empirical.py`, `make_case.py`,
`run_matrix.py`, `extract_forces.py`, `fit.py`, `check.py`, `sysid.py`, and
`randomize.py`. `run_matrix.write_slurm_array` writes an optional serial Slurm
array using the same checked runner, one retry, and campaign stop rule.

Example input:

```yaml
case:
  hull_path: underwater_hull_frd.stl
  length_m: 5.75
  beam_m: 0.805
  draft_m: 0.27
  block_coefficient: 0.6506
  density_kg_m3: 1025.0
  kinematic_viscosity_m2_s: 1.0e-6
  speed_mps: 1.953
  cg_frd_m: [0.0, 0.0, 0.0]
  waterline_frd_z_m: 0.0
  scale_to_m: 1.0
mass_kg: 833.38
hull_family: displacement_monohull
operating_envelope:
  max_beta_deg: 16.0
  max_abs_r_prime: 0.6
v5_runtime_payload: path/to/existing/runtime_payload.json
```

The input mesh must be closed, in FRD, and already clipped to the frozen
waterline with a waterplane closure. This step does not redo hydrostatics or
silently reinterpret source axes. CG and waterline ordinates are SI; only the
mesh coordinates use `scale_to_m`. OBJ/STL are accepted by the mesh loader.
The force integration uses the actual underwater hull without a factor of two.
The force origin is CG projected to the waterline, and extraction shifts the
full wrench to the configured CG once before applying the frame rotation.
The surface records that reference point. Plant6 translates water-relative
velocity to it and shifts the physical wrench back to the body-frame origin
using `r × F` when the two origins differ.

## Load contract

Copy the generated YAML payload under the runtime vessel's
`maneuvering_surface` key, and set `added_mass_coriolis_enabled: false`.
The payload must declare `bcod-maneuvering-spec-a-v1` and
`physical_fluid_on_hull_FRD`; resisting-wrench payloads are rejected.

At forward operating speed the surface replaces horizontal V5 loads; C_A is
zero and M_A remains in the acceleration matrix. Surge is the existing
straight-line R(u) plus ΔX. The low-speed/reverse and Fr>0.45 routes use the
historical V5/C_A path. Other hydrostatics/resistance providers are untouched.
The runtime exposes `maneuvering_surface_route` in the wrench ledger.

Specify `min_forward_speed_mps` under `maneuvering_surface` for the intended
low-speed transition. The runtime factory currently defaults this to 20% of
the reference speed; this is a numerical applicability choice, not calibrated
physics. The switch is discrete and should be reviewed on trajectories.

For Fr 0.30–0.45, generate coefficients explicitly at Fr=0.30 and evaluate the
nondimensional surface at runtime speed with `extrapolated_fr` visible in the
ledger. The case generator refuses a higher design CFD speed instead of
silently changing it. Above Fr 0.45 it refuses CFD generation.

## Matrix, meshing, and runner

The default matrix has ten cases. A smaller drift/rate envelope scales each
group's design coordinates so cases remain distinct. Only geometry and
template parameters enter generation; experimental force data never do.
One checked mesh is copied across cases. The 8° drift runs first to supply
the near-zero straight-case force scale. A failed case gets one continuation
with every relaxation factor halved; after three failed cases the runner
stops. Restarted force outputs are stitched by iteration.

Legacy `mesh_level='coarse'|'medium'|'fine'` calls map to FAST/STANDARD/REFERENCE.
The historical three-mesh report helper remains available for explicit debug
work. Automatic production selection uses only the two +8° sentinels below.
Five wall layers retain the existing flat-plate cell-center y+ estimate; actual
layer coverage and y+ must still be measured.

## OpenFOAM 11 turning formulation

Foundation 11's installed rotor example uses all-domain `MRFProperties`,
`MRFFreestreamVelocity`, and `MRFnoSlip`. The turn is through still water:
the absolute far-field velocity is zero. The rotating-frame origin accounts
for both sway and surge: relative to CG, its foam coordinates are
`(-v/r, -u/r, 0)` and its angular velocity is `(0,0,-r)`.

Sources:

- [MRF no-slip implementation](https://cpp.openfoam.org/v11/MRFnoSlipFvPatchVectorField_8C_source.html)
- [MRF freestream definition](https://cpp.openfoam.org/v11/MRFFreestreamVelocityFvPatchVectorField_8H_source.html)

The small execution fixture checks hull boundary velocities, mesh validity,
and solver startup. It is not a validated turning-force solution.

## Fit and empirical checks

Y/N fits use odd cubic and absolute-value bases; ΔX uses the even three-term
basis after subtracting the matched-speed straight CFD X. Synthetic mirrors
stay in the same leave-one-out fold as their original cases.

The ten-case design is full rank, but removing either mixed case makes that
fold rank deficient for both Y/N bases. The fitter reports these folds,
uses minimum-norm least squares consistently for the requested comparison,
and requires review. A smaller leave-one-out score alone does not validate
the coupling terms. Failed cases are excluded; unidentifiable fits stop.

The report includes both candidate scores, linear sign/factor checks,
measured straight-case bias, force-range LOO checks, stability index, and
measured serial core-hours when available. Polynomial coefficients are not
automatically accepted because a report was written.

The quoted Clarke equations match [Fossen's published reference
implementation](https://raw.githubusercontent.com/cybergalactic/PythonVehicleSimulator/master/src/python_vehicle_simulator/lib/models.py).
The original 1983 paper has **not been directly verified**, and there is no
claimed published-KCS regression validation. `compare_v5` therefore always
marks `requires_review`. `linear_derivatives` accepts a physical full-fluid
V5 callback, excluding rigid-body Coriolis, and performs dimensional-to-prime
conversion separately for sway and yaw-rate derivatives.
The CLI computes these derivatives directly from `v5_runtime_payload` without
regenerating BEM; alternatively it accepts explicit `v5_linear_derivatives`.

## System ID and episode randomization

`sysid.fit_output_error` fits a parameter vector with a ridge penalty toward
the prior and one explicitly held-out maneuver. `plant6_simulator` connects
a fixed-mass Plant6 factory to timestamped observations and physical actuator
wrenches from an independently fixed actuator model. It rejects logs without
that model identity. Field logs and actuator characterization are not supplied
with this repository task, so no field fit is claimed.

`randomize.sample_episode` returns a new surface and seed/draw/multiplier log.
Linear/nonlinear coefficient groups use the specified intervals. The returned
added-mass multiplier is applied by the episode creator; it does not mutate a
shared mass object. Samples changing the nominal stability-index sign are
rejected.

## Verification and remaining acceptance

- Synthetic full-matrix fit, symmetry, physical/resisting sign boundary,
  Munk-once, C_A-off, current-relative loading, and surge preservation tests.
- Capped designs, deterministic randomization, system-ID holdout, failed-tail
  rejection, exactly one retry, and campaign stop tests.
- Tiny OpenFOAM drift and rotating cases: mesh pass and solver startup; the
  corrected rotating-wall speed error is below 5e-7 m/s.
- Ten KCS cases generated under `stage3_results/spec_a/kcs/cases_current/`.
  Earlier `cases/` and `cases_v2/` is retained as pre-audit output and must not be run.

Not yet demonstrated: production hull-family mesh qualification, a full KCS
CFD matrix, one ASV matrix, published MMG actuator integration with SIMMAN
turning/zigzag comparison, original Clarke-paper verification, or measured
compute per production vessel. These are explicit remaining acceptance work,
not results inferred from unit tests or the tiny solver fixture.

## Coarse-first mesh selection (default CFD quality: auto)

`make ROOT vessel.yaml` now prepares only the +8°, r′=0 FAST and STANDARD
sentinels. Run `select ROOT vessel.yaml` to mesh/qualify both, save
`mesh_selection.json`, and materialize the original ten production states using
one selected mesh. This command stops before solving the remaining states.
After reviewing selection, `run ROOT` reuses the selected sentinel and executes
the other nine states. The other sentinel remains validation evidence under
`sentinels/`; extraction only reads numbered production directories.

Explicit `make ... --quality fast|standard|reference` bypasses automatic selection.
REFERENCE is for benchmark/debug use. `SPEC_A_DOCKER=1` uses the installed
OpenFOAM Foundation 11 container; otherwise commands use local OpenFOAM.
The runner is serial: core count is one and core-hours are solver wall-hours.
Mesh wall time is stored separately in each sentinel's `mesh_runtime.json`.

| Profile | Background budget | Target final cells |
|---|---:|---:|
| fast | 150,000 | 300,000–700,000 |
| standard | 800,000 | 1,000,000–1,500,000 |
| reference | 1,800,000 | 2,000,000–3,000,000+ |

Background spacing is `(domain_volume / budget) ** (1/3)`, with integer cell
counts rounded down in each direction. Domain extents remain 6L × 6L × 1.5L.
Hull levels 4–5, stern/wake level 2, and five physical-thickness wall layers
are unchanged. Final-cell ranges are targets, not guaranteed across geometries;
actual counts come from checkMesh. No experimental loads enter mesh selection.

Selection requires both sentinels to pass the existing force-tail qualification.
For each channel, error is `abs(fast-standard)/max(abs(standard), scale_floor)`.
The force floor is `1e-4 * 0.5*rho*L²*U²`; moment floor is force floor times L.
This dimensionless floor is a numerical comparison safeguard, not a fitted
hydrodynamic constant. Both errors at most 10% selects FAST; otherwise STANDARD.
Failed/unqualified runs never silently select a profile. Profile metadata is
retained in CSV/YAML; mixed-profile fitting is rejected. Per-case results retain
cell count, solver wall seconds, core count, core-hours and total iterations
(including any retry).
