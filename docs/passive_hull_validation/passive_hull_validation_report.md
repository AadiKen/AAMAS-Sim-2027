# MANTA Passive Hull Validation Report

## Campaign state

Baseline commit: `4bb33c17c8c4ad11057c0f57f44461e97945a59c`. Generator constants were not tuned. Frozen package schema: `manta-hydrodynamics-v1`.

## Geometry and hydrostatics

KVLCC2M official grid reconstruction is watertight. Displacement is -0.012% from the official offset value; wetted area is 1.329% high; beam and draft errors are 0.011% and 0.029%. Lpp is taken from the official perpendiculars; the mesh bounding length includes overhang and is not treated as Lpp.

## Surge resistance

Public numeric resistance references were recovered: KVLCC2M CT=0.0042612 with 0.7% experimental uncertainty at the fixed double-model case, and KCS CT=0.003557 at Fn=0.26 with 1% uncertainty. Both remain unscored because M1 produced no KVLCC2M package and the official KCS geometry has not been staged. Values and source mappings are versioned under each benchmark reference directory.

## Static sway and yaw

14 NMRI static-drift points from -3° through 18° were recovered; 0 were scored. NMRI defines CY positive to starboard; the adapter maps positive beta to negative FRD sway while preserving FRD force and yaw moment signs, then applies the published dynamic-pressure normalizations.

No static force point was scored: package generation reached the restoring LUT, but the source geometry ends at the design waterline and the frozen ±20% draft restoring domain includes fully emerged/submerged states. The missing topside geometry is a geometry/envelope limitation; no benchmark-specific geometry extension was made.

The symmetry self-test is defined as even CX and odd CY/CN across positive and negative drift. It could not be run against MANTA because no package was produced.

## Linear and nonlinear maneuvering

KVLCC2 literature derivatives are contextual only because they are not automatically equivalent to KVLCC2M. The original KVLCC2 geometry and all derivative scaling definitions were not recovered as a matched package. No hard derivative score is reported.

## Added mass and dynamics

DTMB 5512 has not been quantitatively scored. The source audit flags bilge keels and unresolved moment mapping. The load contract excludes rigid-body inertia and separates fluid added-mass reaction, added-mass Coriolis, and damping.

## Generalization

KCS is not scored because its official geometry has not been staged, even though the peer-reviewed resistance reference and test condition are now recovered. It remains a held-out benchmark once the production CAD input can be assembled.

## Failure diagnosis

- KVLCC2M resistance: package generation is blocked by geometry/restoring-envelope coverage.
- KVLCC2M static drift: official reference data and convention mapping are available, but the frozen package gate failed because the source geometry omits topside needed by the full restoring LUT; horizontal force accuracy remains unscored.
- KVLCC2 derivatives: geometry/convention uncertainty.
- DTMB 5512: appendage mismatch and moment-origin uncertainty.
- KCS holdout: reference-data availability.

## Final status

```text
PUBLIC SOURCE ACQUISITION         PARTIAL
BENCHMARK GEOMETRY                PASS for KVLCC2M design-waterline hull; restoring envelope limitation
CONVENTION ADAPTERS               PARTIAL (KVLCC2M reference mapping reviewed; MANTA self-test blocked)
KVLCC2M RESISTANCE                REFERENCE RECOVERED; UNSCORED
KVLCC2M STATIC DRIFT              UNSCORED; M1 package generation did not complete
KVLCC2 DERIVATIVES                PARTIAL / CONTEXT ONLY
DTMB5512 DYNAMIC PMM              PARTIAL / UNSCORED
KCS HELD-OUT RESISTANCE           REFERENCE RECOVERED; UNSCORED

PASSIVE CAD→COEFFICIENT PHYSICAL VALIDATION STATUS: PARTIAL; KVLCC2M geometry and numeric resistance reference recovered, but frozen package gate blocks force/resistance scoring; dynamic inertia and second-hull generalization remain unscored
```
