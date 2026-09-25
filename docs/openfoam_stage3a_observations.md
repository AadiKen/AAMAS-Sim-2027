# Stage 3A observation contract

An OpenFOAM `forces.dat` row is the fluid-on-hull wrench in the solver's forward/port/up axes at the declared `CofR`. It is **not** a fitting observation. The qualifier subtracts a matched quiescent static wrench in those same axes, shifts the remaining moment to the BCOD reference point, rotates to forward/right/down, and negates the fluid load. Each output row is therefore a **hydrostatic-offset-corrected resisting wrench**, not a total wrench. The source force time is the common time for displacement, velocity, acceleration, force, and moment. Motion is evaluated analytically from the case's commanded law at that time; missing or sparse force samples fail.

| Experiment | Reference and fitted quantity | Window |
| --- | --- | --- |
| Steady translation | Matched static wrench; dynamic resisting damping. Fixed-hull inlet is the negative commanded body velocity. | Active-channel stationary tail. |
| Steady rotation | Matched static wrench; dynamic resisting rotational damping. | Two orientation-matched revolutions after a startup revolution; phase waveform repeatability required. |
| Forced horizontal translation | Matched static wrench; dynamic resisting force, sampled with commanded velocity and acceleration. | Three complete cycles after at least one startup cycle; harmonic repeatability required. |
| Forced yaw | Matched static wrench; dynamic resisting yaw moment, sampled with commanded angular velocity and acceleration. | Same cycle gate. |
| Hydrostatic restoring | Owned by the simulator hydrostatics model; not sent to the damping or added-mass fitter. | Separate static displacement/attitude study if later required. |

Heave, roll, and pitch forced-motion cases require a position-dependent hydrostatic reference and are rejected by the current qualifier. A constant baseline would leave restoring terms in the dynamic observation. Horizontal translation and yaw preserve the hull's vertical placement in still water, so a constant static buoyancy baseline is appropriate only when geometry, waterline, center of gravity, moment center, fluid, mesh settings, and domain match.

The `bcod-qualified-observation-v1` JSON artifact is generated with `bcod vessel qualify-case --case CASE --reference STATIC_CASE --output ARTIFACT.json`. It records case and reference identities, geometry and source-file hashes, fluid and mesh definitions, waterline, motion law, numerical and experiment quality metrics, the selected force-window hash, and every aligned transformed sample. `bcod vessel identify --observations ARTIFACT.json` recomputes the artifact from its solved source cases and rejects any mismatch. The direct fitter also requires verified observation records. Hand-authored observation arrays require the explicit `--unsafe-debug-observations` switch and are unsuitable for production fitting.

This is a scientific qualification gate, not a CFD accuracy or benchmark validation. A passing numerical history alone does not establish a usable fitting window. Restart equivalence likewise requires a separately executed split and uninterrupted pair with matching phase and force samples.
