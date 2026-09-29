# Constrained section-envelope experiment

> **Superseded source selection.** This first experiment used only the smaller low-hull groups. The later [mixed-group chain reconstruction](constrained_envelope_report.md) found the principal buoyant side faces hidden in STEP group 13380. Its hydrostatic and coefficient results replace the incomplete-volume finding below. This page remains an audit of why the first source selection was rejected.

## Decision

**REJECT all three exploratory hull envelopes for coefficient validation.** Section lofting can make watertight, two-component meshes that pass the unchanged M1 topology gate. It cannot reconcile the surviving selected pontoon surfaces with the independently [published 52.3 kg weight and 0.17 m draft](https://okeanus.com/wp-content/uploads/2025/08/sr-surveyorm18-specsheet-new01.pdf). The largest hypothesis encloses 19.46 L, displacing at most 19.94 kg in 1,025 kg/m³ seawater even when fully submerged. The nominal mass requires 51.02 L. Approximately 31.6–31.9 L of buoyant volume is missing from these envelopes. Surface deviations at some surviving pocket features also exceed the required 5 mm maximum. No CAD-derived coefficient package was produced or compared with trajectories.

## Source sections and hypotheses

[Source-face selection](source/group_selection.json) tessellated 587 low STEP faces without altering M1. Cross-sections showed that several small end groups are circular propulsion hardware/inserts, while the outer pontoon candidates are the long side shells. The selected source tessellation and [section constraints](hypotheses/section_constraints.npz) are retained for inspection. The cleaner negative-source-X outer shell supplied sections every 5 mm with 512 angular samples. Missing lower angular sectors were interpolated along the longitudinal axis from neighbouring intact sections; a planar upper lid was added at the surviving top edge. Starboard was mirrored from port. `smooth`, `inset`, and `outset` differ by a tapered ±6 mm radial change only in unsupported lower angular sectors. The [section plot](hypotheses/section_hypotheses.png) shows original samples and reconstructed contours. The code is in `tools/reconstruct_surveyor_envelopes.py`.

These are **hypotheses, not accepted source-preserving hulls**. The loft uses a single radial contour per station; the pocket features contain overlapping and non-radial surviving surfaces that this representation cannot preserve exactly. The mirrored starboard shell also differs locally from intact starboard faces. The selected main-pontoon source faces span approximately 1.41 m longitudinally and 0.82 m across both sides; the 1.83 m and 0.91 m specifications refer to the overall vehicle and were not used to stretch them. The candidate FRD mapping remains an inference from the STEP geometry, with the source longitudinal origin set at Z=915 mm. Propulsion hardware farther along the assembly was retained as source context but not silently converted into solid displaced pontoon volume.

## Measurements

| Hypothesis | Volume (L) | Maximum buoyant mass (kg) | Volume centroid FRD z (m) | Source distance p95 / max (mm) | Reconstructed area upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth | 19.304 | 19.79 | 0.20127 | 0.93 / 15.34 | 43.7% |
| inset | 19.160 | 19.64 | 0.20106 | 0.92 / 14.37 | 43.6% |
| outset | 19.456 | 19.94 | 0.20150 | 0.95 / 15.34 | 43.8% |

The [quality measurements](hypotheses/qualification.json) use sampled vertices and triangle centroids of surviving main-shell STEP faces, compared with the loft surface. The 95th-percentile 2 mm source-distance target passes, but the 5 mm maximum target **fails**. The upper-bound reconstructed-area estimate counts facets incident to unsupported section angles and end/upper caps. At the published draft reference, all of this exploratory envelope is below the waterline; therefore this is also an approximate upper bound on reconstructed *underwater* area as a fraction of the model's wetted area. It is much too large to treat as a minimal pocket patch.

The mesh vertical extent is FRD z=0.167–0.248 m, only about 0.081 m. A 0.17 m draft measured upward from its keel puts the waterline at z≈0.078 m, above its entire modelled volume. Waterplane area there is **zero**. Equilibrium draft, center of buoyancy and waterplane at **52.3 kg** are **undefined** because the required displaced volume exceeds the fully submerged volume. The reported centroids are full *enclosed-volume centroids*, not nominal-load centers of buoyancy. The [draft plot](hypotheses/published_draft_sanity.png) shows the mismatch. This indicates that the identified STEP surfaces do not represent the vessel's complete displaced geometry; the external facts were never used to deform intact faces.

All three STL hypotheses are watertight, winding-consistent, positive-volume, and have two separated hull components. [Frozen M1 geometry results](hypotheses/m1_geometry_gate.json) show `PASS` for the mesh topology gate. At 52.3 kg, independent calls to the unchanged M1 production generator each stop at `Mass is outside displacement capacity`; [attempt status](m1_nominal_attempts/status.json) records the failure before added mass, resistance, sway or yaw coefficients are computed. The placeholder finite CG supplied to meet the function signature was never used downstream of that stopping check. No coefficient spread or `<10% / 10–25% / >25%` maneuvering classification can be calculated.

Coarse (10 mm / 256 angles), nominal (5 mm / 512 angles), and fine (2.5 mm / 512 angles) lofts remained watertight. The nominal-to-fine volume changes were 0.535%, 0.517%, and 0.553% for smooth, inset, and outset respectively, slightly above the initial 0.5% target. Enclosed-volume centroid shifts were 0.67–0.71 mm. [Refinement measurements](hypotheses/refinement_metrics.json) preserve the cases. This numerical issue is secondary to the 32 L displacement deficit and source maximum-distance failure.

## Consequence

The three meshes in [hypotheses](hypotheses) are diagnostic only. Their small volume spread does **not** establish small hydrodynamic uncertainty: all three omit a much larger buoyant region implied by the independent weight/draft data. Do not use them for CAD-derived coefficient claims or select one using trajectory fit. The original STEP assembly needs a new source classification or a complete buoyancy-envelope reference showing which structures carry the remaining displaced volume and how the propulsion pockets join it. The nominal Surveyor configuration is now **52.3 kg**, with 49 kg retained only as a sensitivity reference in [assumptions](/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/docs/surveyor_cad_validation/assumptions.yaml).

## Reproduce

From the repository root, use the supplied STEP and the checked-in surface-group inventory:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/export_surveyor_section_sources.py 'PATH/Mock_Surveyor_1A (1).STEP' docs/surveyor_cad_validation/geometry_recovery/source/occ_surface_groups.json docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/source
MPLCONFIGDIR=/private/tmp/manta-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/reconstruct_surveyor_envelopes.py docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/source/selected_low_surfaces_source_mm.stl docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/hypotheses
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/qualify_surveyor_envelopes.py docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/source/selected_low_surfaces_source_mm.stl docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/hypotheses
```
