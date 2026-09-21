# BCOD Stage 2 validation report

## Executive result

**STAGE 2: FAIL**

## Subsystem table

| Subsystem | Status | Cases | Passed | Failed | Blocked | Warnings |
|---|---:|---:|---:|---:|---:|---:|
| baseline | PASS | 1 | 1 | 0 | 0 | 0 |
| bathymetry | FAIL | 4 | 2 | 0 | 2 | 2 |
| collision | FAIL | 12 | 2 | 0 | 10 | 10 |
| combined | FAIL | 8 | 1 | 0 | 7 | 7 |
| current | FAIL | 6 | 2 | 0 | 4 | 4 |
| grounding | FAIL | 5 | 0 | 0 | 5 | 5 |
| irregular_waves | FAIL | 3 | 2 | 0 | 1 | 1 |
| regular_waves | FAIL | 6 | 2 | 0 | 4 | 4 |
| vessel_vessel | FAIL | 7 | 2 | 0 | 5 | 5 |
| wind | FAIL | 6 | 2 | 0 | 4 | 4 |

## Failure table

| Test ID | Category | Observed | Expected |
|---|---|---|---|
| BATH-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| BATH-004 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-001 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-005 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-006 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-007 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-008 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-009 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-010 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-011 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COLL-012 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COMB-002 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COMB-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COMB-004 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COMB-005 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COMB-006 | REFERENCE | Grounding prerequisite is not implemented | Production capability and independent validation fixture available |
| COMB-007 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| COMB-008 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| CUR-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| CUR-004 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| CUR-005 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| CUR-006 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| GROUND-001 | REFERENCE | No production seabed-contact/grounding solver exists | Production capability and independent validation fixture available |
| GROUND-002 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| GROUND-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| GROUND-004 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| GROUND-005 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| VVC-002 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| VVC-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| VVC-004 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| VVC-005 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| VVC-007 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WIND-002 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WIND-004 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WIND-005 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WIND-006 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WIRR-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WREG-003 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WREG-004 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WREG-005 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |
| WREG-006 | REFERENCE | Required Stage 2 case runner/reference is not implemented yet | Production capability and independent validation fixture available |

## Quantitative evidence

- **BATH-001 (PASS)**: max_depth_error=0.0
- **BATH-002 (PASS)**: max_plane_error=0.0
- **COLL-002 (PASS)**: contact_count=1, impulse_ns=67.16135462325637, penetration_m=0.19999999999999996, momentum_residual=0.0
- **COLL-004 (PASS)**: contact_count=1, impulse_ns=67.16135462325637, penetration_m=0.19999999999999996, momentum_residual=0.0
- **COMB-001 (PASS)**: composition_max_abs_error=1.4210854715202004e-14
- **CUR-001 (PASS)**: max_abs_field_error=0.0
- **CUR-002 (PASS)**: matched_relative_flow_wrench=0.0, stationary_surge_force=54.28810264385692, stationary_sway_force=-32.15869381203331
- **ENV-000 (PASS)**: max_environment_wrench=0.0
- **VVC-001 (PASS)**: contact_count=1, impulse_ns=67.16135462325637, penetration_m=0.19999999999999996, momentum_residual=0.0
- **VVC-006 (PASS)**: contact_count=1, impulse_ns=67.16135462325637, penetration_m=0.19999999999999996, momentum_residual=0.0
- **WIND-001 (PASS)**: max_abs_wrench=0.0
- **WIND-003 (PASS)**: max_abs_wrench_error=1.9895196601282805e-13
- **WIRR-001 (PASS)**: same_seed_max_error=0.0, different_seed_max_difference=0.6823122222434432
- **WIRR-002 (PASS)**: realized_Hs=0.8027574758918153, peak_period=4.102564102564103, Hs_relative_error=0.003446844864769122, Tp_relative_error=0.025641025641025772
- **WREG-001 (PASS)**: amplitude_relative_error=4.455184596394157e-05, max_value_error=2.7755575615628914e-17
- **WREG-002 (PASS)**: max_spatial_error=0.0

## Limitations

This execution validates only the registered cases above. Grounding is blocked because production seabed contact is absent. The campaign does not claim mesh import, per-obstacle materials, continuous collision detection, or full completion of every case in the Stage 2 specification. Missing coverage is not a pass.
