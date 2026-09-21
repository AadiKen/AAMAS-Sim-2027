# CFD smoke: PASS

This smoke test validates software plumbing only. It does not constitute real-vessel hydrodynamic validation.

Git commit: UNBORN (dirty=True)
OpenFOAM: OpenFOAM-11 via Docker; solver `foamRun`
Geometry: deterministic symmetric ellipsoidal smoke hull, 2.0 × 0.4 × 0.15 m, `0e19660845038bcf53f3a038f599eb853d375d16f6421a23b4cf19d69a427956`

## Mesh and cases
- surge_0p5: 12920 cells; mesh PASS; solver converged in 4.506s; [X,Y,Z,K,M,N] = [np.float64(0.528564520054), np.float64(0.0158892088871316), np.float64(-0.058271797347819), np.float64(-0.0106526581547151), np.float64(0.0401890416126928), np.float64(0.0050337278734954)]
- surge_1p0: 12920 cells; mesh PASS; solver converged in 4.581s; [X,Y,Z,K,M,N] = [np.float64(2.019498675323), np.float64(0.0640830683149261), np.float64(-0.234914700419384), np.float64(-0.042825492710485996), np.float64(0.16204160742832), np.float64(0.0204774730018841)]
- surge_1p5: 12920 cells; mesh PASS; solver converged in 4.551s; [X,Y,Z,K,M,N] = [np.float64(4.4726299342599996), np.float64(0.14458531927675), np.float64(-0.5302007944687549), np.float64(-0.096514212838183), np.float64(0.36587894551454797), np.float64(0.046344669228506004)]

## Coefficient fit
Rank 2; condition 8.7178; residual RMS 2.28524e-05 N.
Surge linear damping 0.094895614; surge quadratic damping 1.92457582.

## Vessel and runtime
Artifact: `/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/artifacts/cfd-smoke/latest/vessel/vessel.json`
Fingerprint: `48e9dc19434e0725066ece0b32d8c7a47ea6c9ed2eedb222c85847197cb958ea`
Validation state: unvalidated
5 s final displacement: [0.995235072298653, 0.0, 0.0]
Maximum recorded state/diagnostic magnitude: 10.0
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
