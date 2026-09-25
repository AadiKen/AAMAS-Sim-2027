# OpenFOAM 11 prescribed-motion qualification

Production forced translation and rotation use a **moving hull in fluid**.
A fixed hull with varying inflow was not substituted: it is not established
as equivalent for the acceleration-dependent inertia needed by added-mass
identification.

The previous `dynamicMeshDict` used obsolete top-level `dynamicFvMesh` and
an unsupported `prescribedMotion` block. OpenFOAM Foundation 11's modular
`foamRun -solver incompressibleVoF` ignored it and selected a static mesh.
The corrected case uses `mover {type motionSolver; motionSolver
displacementLaplacian; ...}`, a hull `pointDisplacement` boundary, a
`movingWallVelocity` hull velocity condition, and `cellDisplacement` solvers.
The hull patch itself supplies motion; no moving cell or point zone is needed.
The surrounding fluid mesh deforms while the far field stays fixed.

Forced translation follows `x=A sin(omega*t)`, velocity `A*omega*cos(omega*t)`,
and acceleration `-A*omega^2*sin(omega*t)`. Forced rotation uses
`angularOscillatingDisplacement` with angle in radians. Steady rotation uses
`solidBodyMotionDisplacement` and `rotatingMotion` with angular speed in
rad/s. All use the established FRD-to-OpenFOAM axis and reference transform.

The 0.5 × 0.2 × 0.2 m box fixture had 728 cells. Each fluid run lasted
0.08 simulated seconds; four mesh states were saved. The geometric checker
in `tools/verify_openfoam_motion.py` reads actual hull patch points and
compares them with commanded motion. Evidence is in
`stage3_results/frame-contract-motion/`.

| Case | Time | Commanded | Observed | Max point error | Solver wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forced surge | 0.04 s | 0.02152068273 m | 0.02152068271 m | 2.70e-11 m | 32.09 s |
| Forced yaw | 0.04 s | 0.14347121818 rad | 0.14347121817 rad | 4.91e-11 m | 3.35 s |
| Steady yaw | 0.04 s | 0.02000000000 rad | 0.01999999999 rad | 5.56e-11 m | 6.56 s |

The trajectories establish displacement, direction and phase; velocity and
acceleration signs follow their analytic derivatives. Forces and moments
were written. These runs verify execution of motion, not coefficient accuracy.

OpenFOAM 11's [incompressibleVoF tutorial](https://github.com/OpenFOAM/OpenFOAM-11/blob/master/tutorials/incompressibleVoF/mixerVesselHorizontal2D/constant/dynamicMeshDict)
uses the `mover` form. Its [angular displacement source](https://cpp.openfoam.org/v11/angularOscillatingDisplacementPointPatchVectorField_8C_source.html)
uses `angle0 + amplitude*sin(omega*t)`; its [rotating-motion source](https://cpp.openfoam.org/v11/rotatingMotion_8C_source.html)
integrates angular speed over time.

Numerical/stationarity qualification, hydrostatic-offset treatment,
provenance-safe observations, and benchmark comparison remain outstanding.
No 72-case campaign or fidelity study was run.
