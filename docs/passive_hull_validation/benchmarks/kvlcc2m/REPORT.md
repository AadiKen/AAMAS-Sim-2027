# KVLCC2M passive-hull validation

## Final status

```text
KVLCC2M GEOMETRY                PASS (wetted area differs from published by 1.33%)
DESIGN-WATERLINE MODE           PASS
LOCAL HYDROSTATICS              PASS
M1 PACKAGE                      PASS
RUNTIME ROUNDTRIP               PASS (max residual 6.82e-13)

RESISTANCE CT ERROR             26.745% (38.21 x the stated 0.7% EFD uncertainty)
STATIC CX                       scored at 14 drift points
STATIC CY                       62.00% high at +6 deg; 14.53% high at +12 deg
STATIC CN                       19.59% low at +6 deg; 13.32% low at +12 deg

ADDED-MASS SENSITIVITY          CX/CY unchanged; CN changes materially

GENERATOR HYDRO MODEL MODIFIED  NO
HYDROSTATICS INPUT MODE ADDED   YES
```

## Geometry and package

The input is the official NMRI KVLCC2M underwater grid, reconstructed as one capped watertight full submerged hull. The cap is the official design waterline. No topsides were invented. The official raw archive SHA-256 is `b0d83589d85673d8f56ec4a06cbbbe9215ce34bb9d0947a951e8e0af6a8da024`; reconstructed OBJ SHA-256 is `b19c8d9763f1376edcc872f3a66bf22479a6086825a93ca5e285b9f6305fbf0a`.

| Property | Mesh | Published | Difference |
|---|---:|---:|---:|
| Lpp | 4.970 m between perpendiculars | 4.970 m | 0 |
| Mesh x extent (LOA with overhang) | 5.18034 m | — | retained, not used as Lpp |
| Waterline beam | 0.900898 m | 0.9008 m | +0.0108% |
| Draft | 0.323092 m | 0.323 m | +0.0286% |
| Displacement | 1.171283 m³ | 1.17142 m³ | −0.0117% |
| Wetted area, cap excluded | 6.676737 m² | 6.58919 m² | +1.3286% |

Watertightness, winding consistency, positive volume orientation, one connected component, and zero boundary edges pass. The wetted-area difference is retained and reported; the official hull was not edited to reduce it.

Local design-waterline hydrostatics report `rho*volume = 1200.565 kg` at 1025 kg/m³ against input loading 1200.7055 kg, a 0.0117% residual within the campaign geometry tolerance. Waterplane area is 4.04049 m²; centroid is (0.001212, ~0, 0) m; roll and pitch waterplane second moments are 0.244697 and 7.058818 m⁴. Heave, roll, and pitch restoring are explicitly local, with nonlinear vertical motion unsupported.

The coefficient package was frozen before physical comparison. Canonical package SHA-256: `968dcfd74cc4b8717fd9ef4713be9d582036f09dc7899b78ad351a4310df5415`; file SHA-256: `3ab64f6c1a90b4fed42aedfd41f5cba6c8ecabb51eab032cbc0f86ba71e9a4f1`. It uses strip added mass (low confidence); BEM was disabled. Generator git commit at freeze was `4bb33c17c8c4ad11057c0f57f44461e97945a59c`; complete source hashes and inputs are in `package/package_manifest.json` and `package/generation_input.yaml`.

The new mode only changes how a design-waterline-terminated closed geometry is admitted and how local hydrostatic stiffness is produced. Resistance, added-mass, cross-flow, sway/yaw coefficient, schema, and Plant6 ownership formulas remain unchanged. The generic interface also exposes reference Lpp and water kinematic viscosity so Reynolds-dependent terms can use the published condition without changing the resistance formula.

## Runtime roundtrip and tests

The frozen package passed 343 state comparisons between its independent evaluator and Plant6; maximum absolute residual was `6.82e-13`. Rectangular-barge and curved-ellipsoid fixtures compare full-hull and waterline-capped geometry for displacement, CB, waterplane area/centroid/second moments, wetted area, local restoring stiffness, added mass, and horizontal passive wrench. The M1 coefficient, pipeline, Plant6, and fixture suite passed: **55 tests**.

Convention checks confirm FRD axes, positive drift mapping (`beta > 0` gives `v=-U sin(beta)`), force-on-body signs, midship waterplane moment reference, and coefficient normalization. At +6 degrees MANTA predicts positive CY and CN, matching EFD signs; there is no global Y/N sign inversion.

## Physical scores

### Fixed towing resistance

At U=0.994 m/s, double-model flow, fixed even keel, Fn=0, Re=3.945e6, EFD is `CT=0.0042612`; MANTA predicts `CT=0.00540086` (RT=18.0203 N using package density and the published S0 reference area). Absolute coefficient error is `0.00113966`; relative error is `26.745%`. Against the published 0.7% experimental uncertainty this is 38.21 uncertainty widths. The EFD value is source quality B, reproduced from the published resistance table; the official NMRI condition page provides condition provenance.

### Static oblique towing

All 14 admitted NMRI points from −3 to +18 degrees were scored. The evaluated load includes complete steady Plant6 fluid terms: added-mass/Coriolis, linear damping, surge resistance, and cross-flow. The package remains sealed and unchanged.

| beta | EFD CX | MANTA CX | EFD CY | MANTA CY | EFD CN | MANTA CN |
|---:|---:|---:|---:|---:|---:|---:|
| 0° | −0.01756 | −0.022168 | −0.00004 | 0.000000 | −0.00006 | 0.000000 |
| 6° | −0.01771 | −0.021945 | 0.02560 | 0.041473 | 0.01392 | 0.011192 |
| 12° | −0.01750 | −0.021285 | 0.07082 | 0.081107 | 0.02539 | 0.022007 |

At 0 degrees percentage error is suppressed for near-zero CY and CN: absolute residuals are 0.00004 and 0.00006. At 6 degrees absolute errors are 0.004235 in CX, 0.015873 in CY, and 0.002728 in CN. At 12 degrees they are 0.003785, 0.010287, and 0.003383.

Source-reported CY/CN uncertainty entries are available at 0°, 9°, and 18° only: at 0° they are 0.00027 and 0.00010; at 9° they are 0.00145 and 0.00084; at 18° they are 0.00384 and 0.00129. Other staged drift rows have no uncertainty value, so their CSV fields remain blank.

The lateral model has the correct sign but grows too quickly at low-to-moderate drift, then approaches EFD at the largest angles (CY errors are about 2.8% at 15° and 2.1% at 18°). Yaw has the correct sign and moderate error over 6–12°. The +6° yaw prediction comprises about 42.75 N m of added-mass/Coriolis and 2.47 N m of cross-flow moment, making its uncertainty sensitive to the added-mass matrix.

### Derivative diagnostics

Centered finite differences of the frozen runtime evaluator give `Y_v'=-0.4015`, `N_v'=-0.1076`, `Y_r'=-0.00738`, and `N_r'=-0.03771` under the normalization in `results/derivatives.json`. Existing literature derivatives for original KVLCC2 are contextual only: the geometry differs and their convention is flagged ambiguous, so they are not scored.

### Added-mass sensitivity

Scaling the low-confidence strip added-mass matrix to 0.5x, 1.0x, and 1.5x after the primary score leaves CX and CY unchanged. CN spans 0.00590–0.01648 at +6 degrees and 0.01166–0.03236 at +12 degrees. This is material sensitivity, not a tuned result; the primary package and scores are unchanged.

## Diagnosis

- **Surge resistance: weak.** CT is 26.7% high; the baseline remains the unchanged ITTC-57 friction plus generic low-confidence form factor.
- **Linear sway: weak/not represented as a separate horizontal linear term.** Horizontal lateral force is supplied by cross-flow.
- **Nonlinear sway: moderate.** Correct sign, substantial overshoot at 3–6 degrees, with better agreement at large drift.
- **Yaw moment: moderate.** Correct sign and 13–20% low at the required 6° and 12° points, with material dependence on low-confidence added mass.

No benchmark-specific constants, multipliers, or geometry changes were introduced.
