# Phase 12 CFD implementation discovery

| Stage | Actual production path |
|---|---|
| Geometry source | `tools/cfd_smoke/generate_smoke_hull.py::generate` |
| Geometry import/normalization | `vessel_generation/geometry.py::import_ascii_stl` |
| Case generation | `vessel_generation/cfd.py::OpenFOAMAdapter.generate_operating_case` |
| Meshing and mesh validation | `OpenFOAMAdapter.mesh_case` |
| Solver and convergence detection | `OpenFOAMAdapter.solve_case` |
| Native force/moment parsing | `OpenFOAMAdapter.parse_case` |
| Standard result schema | `vessel_generation/cfd.py::OpenFOAMResult` |
| Coefficient fit and identifiability | `vessel_generation/fitting.py::CoefficientFitter.fit_damping_axis` |
| Canonical vessel | `vessel_generation/models.py::CanonicalVessel` and `generation.py::VesselFactory` |
| Physical/schema validation | `CanonicalVessel`, followed by ordinary runtime `MassProperties`, `Damping`, `Hydrostatics`, and `OperatingEnvelope` checks |
| Normal loading | `config/resolver.py::resolve` then `web/runtime_factory.py::build_engine` |
| Normal rollout | `vessel_generation/cfd_smoke.py::_rollout` through `EpisodeEngine.reset/step` |
| Provenance/integrity | `ParameterLineage`, canonical definition hashes, `cfd_manifest.json`, and `write_hash_manifest/verify_hash_manifest` |

Before the real smoke work, `OpenFOAMAdapter` only wrote `case.json`, invoked one executable, and expected a nonnative `forces.json`. It had no OpenFOAM case dictionaries, meshing, convergence checks, or native force parser.

Existing Phase 12 tests use `SyntheticCFDAdapter` and analytically generated force values. They do not mock subprocesses, install fake solver binaries, or use prerecorded OpenFOAM output. Those tests validate identification code only. The real smoke uses no mocked output, fixture forces, fake solver, or prerecorded case output.
