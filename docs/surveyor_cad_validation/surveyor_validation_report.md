# Surveyor CAD validation — geometry gate report

The later [raw AP214 provenance audit](pocket_face_attribution/provenance/provenance_report.md) maps the two large starboard conical faces to the same original open shell as broad pontoon skin. Their **component attribution is hull/recess skin at moderate confidence**; the water-accessible opening and closed displaced solid remain unresolved. This provenance task did not generate new H1/H2/H3 meshes or run M1. Geometry force uncertainty remains unquantified, the primary package is **NONE**, and trajectory validation remains **NOT RUN**. The earlier three M1 packages below compare bridge shapes under one shared mirrored interpretation; they are not the requested physical pocket ensemble.

The follow-up [CAD geometry recovery audit](geometry_recovery/geometry_recovery_report.md) confirmed large ambiguous lower-hull openings on both sides and stopped without reconstructing underwater surfaces.

The current [constrained-envelope report](geometry_recovery/section_reconstruction/constrained_envelope_report.md) supersedes the initial low-shell experiment. It found the principal buoyant side faces in a mixed STEP group and produced three watertight, hydrostatically plausible diagnostic meshes. Frozen M1 generated diagnostic coefficient packages for all three; none is accepted for trajectory validation because the mirrored pocket region conflicts with surviving opposite-side source faces. The nominal loading assumption is **52.3 kg**; 49 kg remains a sensitivity reference.

## Decision

**Trajectory validation remains stopped at source fidelity.** The uploaded STEP is an open surface-shell assembly. Section lofts pass mesh watertightness and 52.3 kg hydrostatic checks, but the mirrored pocket region does not preserve every surviving opposite-side face. The three M1 packages are diagnostic; no package was frozen for primary trajectory validation. No actuator calibration, Plant6 replay, or simulated-error metrics were produced.

## Frozen baseline and provenance

The [M1 freeze manifest](m1_freeze_manifest.json) records git HEAD `4bb33c17c8c4ad11057c0f57f44461e97945a59c`, per-file hashes, the `manta-hydrodynamics-v1` schema, generation defaults, and a 62-test passing baseline. The worktree was already dirty at freeze; [source snapshot](m1_source_snapshot.tar.gz) preserves the selected M1 files. No Surveyor trajectory data modified the generator or any CAD-derived hydrodynamic coefficient.

The STEP SHA-256 is `1490d6a7769c1bd79e1a14cf2fac330ce0b08818105997a9f28ec4331601282a`; the experiment ZIP SHA-256 is `96e8f384df058380de8f77ae27f2a2e6d376db18de5f8a0b18fe717fcef7b0b5`.

## Geometry result

[Initial geometry report](geometry/geometry_report.md), [source manifest](geometry/source_manifest.json), and [extraction manifest](geometry/extraction_manifest.json) document the first open-shell gate failure. The [new chain-envelope audit](geometry_recovery/section_reconstruction/constrained_envelope_report.md) identifies the larger mixed-group hull faces, three diagnostic meshes, hydrostatics and conditional M1 coefficient comparison. No hull is accepted for primary validation; pocket classification, final FRD origin and physical thruster positions remain unresolved. The [coefficient status](coefficients/status.json) separates diagnostic packages from accepted packages.

## Recorded data and command diagnostics

The ZIP's two state CSVs are preserved byte-for-byte under [processed_logs/raw](processed_logs/raw). The [importer](../../src/bcod_sim/surveyor_validation/logs.py) writes canonical per-vessel CSVs and segments without interpolation. The [quality report](processed_logs/quality_report.json) records 112/91 raw rows, 110/85 normalized rows, 2/6 duplicate timestamps, 1/0 invalid GPS fixes, and 8/5 invalid headings for USV1/USV2. Median sampling is 1 s; the largest gap is 7 s for USV2. The absolute timestamp timezone is unverified. Local coordinates are ENU. Heading is magnetic and has unknown declination. IMU acceleration retains gravity/specific force, so it is not a clean translational acceleration estimate.

Commands are normalized with `port = clip((T+D)/100)` and `starboard = clip((T-D)/100)`. At a 1 s lag, the recorded differential command and yaw rate correlate 0.901 (31 pairs) for USV1 and 0.901 (14 pairs) for USV2, with matching sign in 30/31 and 14/14 pairs respectively. These are convention checks from [sensor diagnostics](processed_logs/sensor_diagnostics.json), not fitted actuator dynamics or proof of a thrust curve. The absolute heading convention remains uncertain.

The boats' median matched separations are 5.07 and 5.22 m, with minimum 3.39 m. Distance plus relative-heading screening identifies only 11/110 and 7/85 rows as low-interaction candidates, scattered across 69 and 52 short maneuver segments. This record cannot support a robust isolated-vessel nuisance calibration subset. The [calibration split](calibration_split.yaml) is explicitly unfrozen and empty. Close-following and near-vessel rows remain in the processed data. The [plots](plots) show recorded command, yaw, heading, GPS speed, separation, and GPS tracks only; none shows a simulation.

## Quantity ownership

| Quantity | Status |
| --- | --- |
| CAD-derived hydrostatics and hydrodynamic coefficients | Three diagnostic M1 packages; none accepted because source fidelity fails in the pocket region |
| Estimated physical metadata | 52.3 kg published nominal; 49 kg historical sensitivity; lateral CG zero by assumed symmetry; planform yaw-inertia estimate and scales in [assumptions](assumptions.yaml) |
| Longitudinal CG and thruster positions | Unresolved from this open assembly |
| Actuator model and thrust scale | Candidate equations in [actuator config](actuator/candidate_models.yaml); uncalibrated |
| Shared environmental current | Unestimated |
| Measured validation quantities | Imported GPS, magnetic heading, yaw rate, IMU and commands; no simulated comparisons |

The [metrics file](metrics.csv) contains explicit `NOT_RUN_SOURCE_FIDELITY_GATE_FAILED` entries, with empty metric cells. It must not be interpreted as zero error. No uncertainty case was chosen or fit to these trajectories.

## Reproduction and next gate

Run `PYTHONPATH=src .venv/bin/python -c 'from bcod_sim.surveyor_validation.logs import import_archive; import_archive("/path/to/task1-20260928T024322Z-1-001.zip", "docs/surveyor_cad_validation/processed_logs")'` and then `PYTHONPATH=src .venv/bin/python tools/analyze_surveyor_logs.py docs/surveyor_cad_validation` from the repository root. The log adapter tests are in `tests/vessel_generation/test_surveyor_log_adapter.py`.

The next geometry gate requires a trustworthy classification of the mixed-group starboard pocket faces and confirmation of the hull frame and thrust axes. Diagnostic frozen M1 generation has run; Plant6 trajectory replay and validation metrics remain deferred until an envelope passes the source-fidelity gate.
