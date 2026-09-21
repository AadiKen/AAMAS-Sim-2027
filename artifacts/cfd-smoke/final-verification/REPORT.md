# CFD smoke: PASS

This smoke test validates software plumbing only. It does not constitute real-vessel hydrodynamic validation.

Git commit: UNBORN (dirty=True)
OpenFOAM: OpenFOAM-11 via Docker; solver `foamRun`
Geometry: deterministic symmetric ellipsoidal smoke hull, 2.0 × 0.4 × 0.15 m, `0e19660845038bcf53f3a038f599eb853d375d16f6421a23b4cf19d69a427956`

## Mesh and cases
- surge_0p5: 12920 cells; mesh PASS; solver converged in 4.424s; [X,Y,Z,K,M,N] = [0.528564520054, 0.0158892088871316, -0.058271797347819, -0.0106526581547151, 0.0401890416126928, 0.0050337278734954]
- surge_1p0: 12920 cells; mesh PASS; solver converged in 4.400s; [X,Y,Z,K,M,N] = [2.019498675323, 0.0640830683149261, -0.234914700419384, -0.042825492710485996, 0.16204160742832, 0.0204774730018841]
- surge_1p5: 12920 cells; mesh PASS; solver converged in 4.505s; [X,Y,Z,K,M,N] = [4.4726299342599996, 0.14458531927675, -0.5302007944687549, -0.096514212838183, 0.36587894551454797, 0.046344669228506004]

## Coefficient fit
Rank 2; condition 8.7178; residual RMS 2.28524e-05 N.
Surge linear damping 0.094895614; surge quadratic damping 1.92457582.

## Vessel and runtime
Artifact: `/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/cfd-smoke/final-verification/vessel/vessel.json`
Fingerprint: `d097cb5942df21557f2cfe22be2654e53313cc77b39bfc6b8a344d4178d293a5`
Validation state: unvalidated
5 s final displacement: [0.995235072298653, 0.0, 0.0]
Maximum recorded state/diagnostic magnitude: 1.0
Repeatability maximum difference: 0.0

## Negative tests
- solver_failure: PASS
- missing_force_output: PASS
- nonfinite_input: PASS
- unidentifiable_fit: PASS
- provenance_tampering: PASS

## Code changes needed
Added production STL validation, OpenFOAM v11 case generation, real Docker execution, mesh/convergence checks, native force parsing, subset damping identification, integrity verification, and the smoke orchestrator.

## Remaining limitations
This coarse steady, laminar, single-phase sweep validates integration only. Mesh concavity findings from strict `checkMesh -allGeometry` are recorded; topology, volumes, non-orthogonality, and skewness pass the production smoke criteria. No real-vessel accuracy or full 6-DOF identification is claimed.

## Regression
- Python suite: 128 passed
- Phase 13 audit: 163 PASS, 15 accepted SKIP, 0 unexpected FAIL
- Legacy isolation: 111 production files passed
- Independent `checkMesh -allGeometry -allTopology`: exit 0
