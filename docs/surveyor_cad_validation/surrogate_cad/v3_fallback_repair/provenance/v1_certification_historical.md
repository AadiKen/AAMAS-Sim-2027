# Surveyor-constrained hydrodynamic surrogate — final certification pass

> **Historical v1 certification (superseded).** The subsequent v2 bridge and v3 localized aft repairs passed the geometry and frozen M1 gates. See the [current final validation report](v3_fallback_repair/final_validation_report.md) and its frozen Surveyor-derived CAD and coefficient package. The findings below describe the earlier, unmodified surrogate only.

**Decision: not accepted.** The existing nominal, fuller and finer CAD files were inspected without rebuilding or changing their geometry. The original STEP and frozen M1 remain unchanged. No trajectory data were used.

## Correct dimensional comparison

The published **1.83 m × 0.91 m describes the whole vehicle**. It is retained only as context and is **not a gate for the wetted pontoon solids**. The source comparison below uses the selected original STEP pontoon face group, tessellated directly from the STEP; the submerged extent uses original face points at source Y≤−75 mm, near the 52.3 kg waterline. The cleaner port pontoon is the symmetry authority, consistent with the original reconstruction decision. The raw starboard source has conflicting pocket geometry.

| Pontoon measurement | Source port | Source starboard | Existing surrogate | Gate |
|---|---:|---:|---:|---|
| Full selected-face aft endpoint, source Z (mm) | −0.931 | −0.931 | 50.000 | Context |
| Full selected-face forward endpoint, source Z (mm) | 1778.760 | 1778.662 | 1750.000 | Context |
| Wetted aft endpoint, source Z (mm) | 18.818 | 18.818 | 50.000 | Compare |
| Wetted forward endpoint, source Z (mm) | 1763.860 | 1763.876 | 1750.000 | Compare |
| Wetted length (mm) | 1745.042 | 1745.058 | 1700.000 | **FAIL:** 2.58% short versus <0.5% |
| Wetted individual maximum width (mm) | 147.037 | 154.080¹ | 147.078 | **PASS** versus trusted port: 0.028% |
| Combined wetted beam from trusted port mirror (mm) | 823.079 | — | 822.950 | **PASS:** 0.016% |
| Wetted pontoon centerline spacing from trusted port mirror (mm) | 676.042 | — | 675.872 | **PASS:** 0.170 mm |

¹ The source starboard width is affected by the known asymmetric source conflict; it was measured but not used to deform the deliberately symmetric surrogate. Direct measurements and both interpretations are in [pontoon_dimensions.json](validation/pontoon_dimensions.json). The chosen 50–1750 mm section range truncates surviving wetted source tips; the length failure is therefore a real geometry limitation, rather than a published whole-craft comparison error.

## Trusted-surface deviation

The [area-weighted surface-distance map](source_comparison/surface_distance_map.png) compares original port STEP face triangle centroids with the nominal surrogate. Values below apply to underwater selected source faces within the modeled longitudinal range. The ordinary-hull set excludes declared review intervals source Z 325–350 and 900–1200 mm. Full data are in [area_weighted_surface_distance.json](source_comparison/area_weighted_surface_distance.json).

| Source area | Median | P90 | P95 | P99 | Maximum | Area >2 mm | Area >5 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| All underwater | 0.0016 mm | 0.374 mm | 0.753 mm | 3.661 mm | 7.870 mm | 0.014120 m² | 0.002357 m² |
| Ordinary hull | 0.0008 mm | 0.235 mm | 0.420 mm | 0.973 mm | **5.900 mm** | 0.000698 m² | **0.0000603 m²** |
| Pocket/boundary review zones | 0.0077 mm | 1.390 mm | 3.147 mm | 5.238 mm | 7.870 mm | 0.013422 m² | 0.002296 m² |

The ordinary-hull >5 mm patch consists of nine sampled source triangles near source **Z 510.9–513.2 mm, X −374.9 to −364.3 mm, Y −225.2 to −220.1 mm**. A denser source vertex/centroid search finds a nearby 6.31 mm point. Those points are roughly **38–45 mm from the selected source mesh's open boundary**. Although the patch is tiny (about 0.012% of assessed ordinary source area), it is outside the known pocket and broken-boundary review zones. The exception requested for pocket transitions, broken boundaries, or tiny edge regions is **not established** here. Ordinary-hull P95 passes, but trusted-surface fidelity remains **FAIL** under the stated exception rule.

## STEP validity and STL consistency

Gmsh/OpenCascade re-imported each STEP as **two positive-volume solids**. Every volume boundary edge has incidence two; there are no free or nonmanifold edges. The independently loaded STL meshes are watertight, consistently oriented, have two connected components and no duplicate triangles. Detailed counts and centers are in [brep_certification.json](validation/brep_certification.json). This is one B-rep kernel check plus an independent **mesh** topology check. An independent B-rep checker capable of `BRepCheck_Analyzer` or FreeCAD `checkGeometry` was not available, and B-rep self-intersection certification remains open. Thus the requested two-method B-rep validity gate is **not certified**.

For nominal, analytical STEP volume is **0.0934550 m³** and STL volume is **0.0935618 m³**, a **0.114%** difference (<0.25%). Their volume centroids differ by **0.529 mm** (<1 mm). Nominal versus fine STEP-derived tessellation bounding boxes differ by at most **0.022 mm**, and equilibrium waterlines by **0.073 mm**. OpenCascade's untrimmed mirrored surface bounding box is loose, so the comparison uses trimmed fine tessellation bounds. See [step_stl_consistency.json](validation/step_stl_consistency.json).

## Hydrostatics and fairness

At 52.3 kg in fresh water (1000 kg/m³), the nominal model displaces **0.0523 m³** at **0.17202 m draft**. Fuller/finer drafts are **0.17187/0.17221 m**. CB, waterplane area and waterplane moments are in [hydrostatics/comparison.csv](hydrostatics/comparison.csv). The ~0.17 m published draft sanity check passes without retuning geometry. Nominal-to-fine triangulation changes volume by 0.048%, CB by 0.102 mm and waterplane area by 0.0086%, within the earlier convergence gates.

The [five-metric fairness plot](renders/fairness_five_metrics.png) shows area, true polygon centroid, pontoon half-beam, vertical depth and perimeter along the longitudinal stations. Area jumps >10% occur near source Z 320–330 mm (the known ordinary reconstruction zone), 1030–1090 mm (the pocket) and 1720–1740 mm (the narrowing forward tip). The first two are sharp and require geometric review; this pass does not assert fairness acceptance solely because they are located in known reconstruction regions. The [orthographic views](renders/orthographic_views.png) and [pocket comparison](renders/pocket_variant_comparison.png) support review.

## Certification and next action

| Gate | Result |
|---|---|
| SOURCE PONTOON DIMENSIONS | **FAIL** — 2.58% wetted length short; beam/spacing pass |
| SOURCE SURFACE FIDELITY | **FAIL** — localized 5.9 mm deviation on ordinary source hull |
| B-REP VALIDITY | **NOT CERTIFIED** — OCC closed solids pass; second B-rep checker unavailable |
| STEP/STL CONSISTENCY | **PASS** — volume, centroid, trimmed bounds and waterline |
| HYDROSTATICS | **PASS** — 0.17202 m nominal draft at 52.3 kg |
| FAIRNESS | **NOT CERTIFIED** — reconstructed-region area jumps need review |
| **SURROGATE ACCEPTED** | **NO** |
| **M1** | **NOT RUN** on this new CAD family |

The nominal geometry hash is recorded in [provenance_manifest.json](provenance_manifest.json), but it is **not frozen as an accepted primary model**. The specific geometry defects now identified are truncated source-supported pontoon tips and the localized ordinary-hull deviation. No geometry was altered during this certification pass. Since the required gates do not all pass, the frozen M1 generator was not run and no state-grid force spread or trajectory validation was produced.
