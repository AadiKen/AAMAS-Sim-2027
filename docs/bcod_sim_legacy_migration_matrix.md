# BCOD-Sim Legacy Migration Matrix

| Legacy subsystem / evidence | Disposition | Required action before new core use |
|---|---|---|
| Floating vehicle preset system | DISCARD | Build canonical vessel registry from scratch |
| `vehicle-a-otter` production dispatch | DISCARD | Central fail-closed resolver with content hashes |
| Vehicle B | DISCARD | Retain only failure evidence/regression fixtures |
| Vehicle C legacy coefficients/object | REFERENCE ONLY | Rebuild dual-azipod vessel from authoritative geometry + CFD/calibration |
| SeaRobotics Surveyor legacy presets | REBUILD | Create one canonical base + explicit physical-instance calibrations if needed |
| MSS Otter dedicated parameters | REFERENCE/REBUILD CLEANLY | Pin primary source, create reference vessel, preserve analytic tests |
| Dedicated planar kernel | REFERENCE/ORACLE | Freeze; do not import into production; use for verification only |
| `legacy-production-engine` | DISCARD | One new engine/lifecycle path |
| Old internal simulator harness | DISCARD | Direct import/public API becomes normal harness |
| Old Gazebo/VRX/etc. comparison harnesses | REFERENCE ONLY | Redesign generalized external comparison contract later |
| Old comparison metrics/figures without replayable provenance | UNVERIFIED | Regenerate from new core before claims |
| Sensor SDK concept | ADAPT | New standard sensor contract and registry |
| Old sensor behavior | REBUILD | Must pass mount/rate/lever-arm/FOV/resolution tests |
| Parametric world generation | ADAPT/VERIFY | Map into canonical World/config system |
| Real-world import plumbing | ADAPT/VERIFY | Central frames/units/coverage/provenance; no hot-loop fetches |
| GEBCO import | REBUILD/VERIFY | Explicit vertical datum/sign + checksum/provenance |
| ENC/buoy import | ADAPT/VERIFY | Convert to collision-capable physical world entities |
| RTOFS | REBUILD/ADAPT | Use 3D depth-specific current; explicit coverage; no 2ds substitution |
| NDBC | REIMPLEMENT TESTED BEHAVIOR | Preserve fail-closed station radius, add full provenance contract |
| CO-OPS | REIMPLEMENT TESTED BEHAVIOR | Preserve station-radius behavior, explicit datum/time semantics |
| NWS | REBUILD | Explicit SI conversion, especially km/h to m/s regression |
| ERA5 | DEFER | Not part of coordinated V1 source stack |
| AIS | REBUILD | Fresh authoritative adapter; old extraction provenance not trusted |
| Collision dynamics | BUILD FRESH | Deterministic collision/contact subsystem |
| Wake/environment-mediated coupling | BUILD/VALIDATE | Double-buffered field; validate with two-Surveyor data later |
| Old task fallback behavior | DISCARD | Closed task registry, config-first tasks |
| Old silent command clamp | DISCARD | Error by default; explicit clamp event only if configured |
| Old validation labels | DISCARD | Computed validation/provenance only |
| Old PPO curves with unresolved producer | DISCARD/REGENERATE | New bundles record trainer/core/config/contract hashes |
| Reset/checkpoint/batch behavioral evidence | KEEP AS TESTS | Re-run through new audit adapter |
| Wave phase-sign corrected behavior | KEEP AS REGRESSION | Re-test against analytic convention |
| Added-mass-in-mass-matrix behavior | KEEP AS REGRESSION | Permanent negative test against delayed-feedback implementation |
| RTOFS/NDBC/CO-OPS pinned payload evidence | KEEP AS FIXTURES | Revalidate adapter parsing/coverage/provenance |
| Audit suite and evidence archive | KEEP | Make portable adapter point at new core and wire into CI |
| Web authoring/demo UX concepts | ADAPT | Backend owns all simulation; web exports canonical configs |
