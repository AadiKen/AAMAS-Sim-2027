# Phase 12 CFD implementation discovery

This inventory records the actual implementation after the minimum production fixes required by the real-solver smoke.

| Stage | Production code path |
|---|---|
| Geometry input | `tools/cfd_smoke/generate_smoke_hull.py`, `generate(...)` creates the deterministic source STL. |
| Geometry normalization and validation | `src/bcod_sim/vessel_generation/geometry.py`, `import_ascii_stl(...)` validates finite nondegenerate triangles, closed two-manifold edges, meters, and the declared source frame. |
| OpenFOAM case generation | `src/bcod_sim/vessel_generation/cfd.py`, `OpenFOAMAdapter.generate_operating_case(...)` writes the complete OpenFOAM v11 case. |
| Meshing | `OpenFOAMAdapter.mesh_case(...)` invokes real `blockMesh`, `snappyHexMesh -overwrite`, and `checkMesh -allGeometry -allTopology`. |
| Solver invocation | `OpenFOAMAdapter.solve_case(...)` invokes `foamRun -solver incompressibleFluid` in the pinned OpenFOAM Foundation container. |
| Solver status and convergence | `OpenFOAMAdapter.solve_case(...)` requires a zero exit, no fatal/nonfinite indicators, `SIMPLE solution converged`, and the OpenFOAM `End` marker. |
| Force and moment parsing | `OpenFOAMAdapter.parse_case(...)` reads the native `postProcessing/forces/*/forces.dat`, requires the pressure and viscous force/moment channel declarations, and returns all six axes. |
| Standard CFD result | `src/bcod_sim/vessel_generation/cfd.py`, `OpenFOAMResult`. |
| Coefficient fitting | `src/bcod_sim/vessel_generation/fitting.py`, `CoefficientFitter.fit_damping_axis(...)`. |
| Identifiability | `CoefficientFitter.fit_damping_axis(...)` explicitly checks design-matrix rank and condition number before fitting. |
| Canonical vessel generation | `src/bcod_sim/vessel_generation/models.py`, `CanonicalVessel`; `src/bcod_sim/vessel_generation/generation.py`, `VesselFactory`. |
| Vessel validation | `CanonicalVessel` physical/schema validators followed by `MassProperties`, `Damping`, `Hydrostatics`, and `OperatingEnvelope` validation during ordinary construction. |
| Production vessel loading | `src/bcod_sim/config/resolver.py`, `resolve(...)`, then `src/bcod_sim/web/runtime_factory.py`, `build_engine(...)`. |
| Normal simulator rollout | `src/bcod_sim/vessel_generation/cfd_smoke.py`, `_rollout(...)`, through the ordinary `EpisodeEngine.reset/step` lifecycle. |
| Provenance | `ParameterLineage`, canonical definition hashes, `cfd_manifest.json`, and `write_hash_manifest/verify_hash_manifest`. |
| Orchestration | `src/bcod_sim/vessel_generation/cfd_smoke.py`, `run(...)`, exposed by `scripts/run_cfd_smoke.sh`. |

## What existed before this smoke

The original Phase 12 `OpenFOAMAdapter` wrote only a `case.json`, invoked a configured executable once, and expected a repository-specific `forces.json`. It did not generate an OpenFOAM mesh/case, invoke the mesh tools, inspect convergence, or parse OpenFOAM native force output.

The existing Phase 12 tests use `SyntheticCFDAdapter` and analytically generated force values. They do not mock subprocess calls, install fake solver binaries, or use prerecorded OpenFOAM case output. Their synthetic results remain identification unit tests and are not evidence for the real CFD smoke.

The real smoke uses no mocked CFD output, fixture force values, fake executable, or prerecorded solver output.
