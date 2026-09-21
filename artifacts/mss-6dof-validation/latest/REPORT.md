# MSS 6DOF VALIDATION RESULT

## Reference

- MSS source: https://github.com/cybergalactic/MSS.git
- version/commit: `98970f71a21cfe81e7e29abdcc1bb6741789cddc`
- exact 6DOF model: `CRAFT/USV/models/otter.m`
- runtime: GNU Octave (aarch64-apple-darwin25.4.0) version 11.3.0

## Plant equivalence

- EXACT: mass, rigid_body_inertia, CG, added_mass, linear_damping, quadratic_yaw_damping, cross_flow_drag, full_coupled_linear_restoring, thrust_mapping, shaft_speed_saturation
- EQUIVALENT_REPARAMETERIZATION: Coriolis, actuator_dynamics
- APPROXIMATE: none
- UNREPRESENTABLE: none

## Frames

- MSS navigation frame: NED
- MSS body frame: FRD
- BCOD mapping: identity into canonical NED/FRD; MSS Euler ZYX is converted once to body-FRD → world-NED quaternion.

## Results

- T00: NEAR PARITY — max position 0 m, orientation 0 deg, linear velocity 0 m/s, angular rate 0 rad/s
- T01: NEAR PARITY — max position 2.77803e-17 m, orientation 1.70755e-06 deg, linear velocity 2.12193e-16 m/s, angular rate 6.19296e-16 rad/s
- T02: NEAR PARITY — max position 1.56616e-13 m, orientation 2.41484e-06 deg, linear velocity 1.40045e-12 m/s, angular rate 3.54484e-12 rad/s
- T03: NEAR PARITY — max position 3.89644e-18 m, orientation 1.70755e-06 deg, linear velocity 7.37596e-17 m/s, angular rate 2.61943e-16 rad/s
- T04: NEAR PARITY — max position 4.857e-14 m, orientation 2.41484e-06 deg, linear velocity 6.55893e-13 m/s, angular rate 2.08012e-12 rad/s
- T05: NEAR PARITY — max position 8.48926e-18 m, orientation 1.70755e-06 deg, linear velocity 1.24693e-16 m/s, angular rate 4.48426e-16 rad/s
- T06: NEAR PARITY — max position 3.94165e-13 m, orientation 2.41484e-06 deg, linear velocity 3.70676e-12 m/s, angular rate 9.39688e-12 rad/s
- T07: NEAR PARITY — max position 6.98411e-13 m, orientation 2.95756e-06 deg, linear velocity 6.34754e-12 m/s, angular rate 1.69692e-11 rad/s
- T08: NEAR PARITY — max position 1.63131e-09 m, orientation 2.41484e-06 deg, linear velocity 2.33388e-09 m/s, angular rate 8.11088e-09 rad/s
- T09: NEAR PARITY — max position 3.80881e-15 m, orientation 1.70755e-06 deg, linear velocity 1.31765e-14 m/s, angular rate 4.9339e-14 rad/s
- T10: NEAR PARITY — max position 1.14399e-11 m, orientation 2.95756e-06 deg, linear velocity 6.94123e-11 m/s, angular rate 7.65427e-10 rad/s
- T11: NEAR PARITY — max position 1.14399e-11 m, orientation 2.95756e-06 deg, linear velocity 6.94123e-11 m/s, angular rate 7.65427e-10 rad/s
- T12: NEAR PARITY — max position 5.39764e-12 m, orientation 2.95756e-06 deg, linear velocity 4.67909e-11 m/s, angular rate 1.20839e-10 rad/s

## Interpretation

- Track A: NEAR PARITY
- Track B: NEAR PARITY
- Worst position discrepancy: 1.63131e-09 m (T08)
- Worst orientation discrepancy: 2.95756e-06 deg (T07)
- Worst linear-velocity discrepancy: 2.33388e-09 m/s (T08)
- Worst angular-rate discrepancy: 8.11088e-09 rad/s (T08)
- Overall classification: **NEAR PARITY**

- Short-horizon behavior: see the per-window metrics in report.json.
- Long-horizon behavior: see the full-trajectory metrics above.
- Likely source: if residual error remains after component parity, inspect the dynamics formulation and actuator path according to the Track A/Track B decision tree.

The precondition gate passed for execution, frames, initial state, constants, timestamps, finite values, and input histories. The parity-v2 plant uses the exact resolved coupled MSS stiffness matrix and independently implemented MSS-equivalent strip theory.

## Timestep sensitivity

No case required dt=0.005 rerun.

## Regression tests

- Python suite: PASS (143 passed)
- Phase 13 audit: PASS (163 PASS, 15 accepted SKIP, 0 unexpected FAIL)

## Exact reproduction command

```bash
./scripts/run_mss_6dof_validation.sh
```
