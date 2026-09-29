# Surveyor v2 surrogate — independent certification pass

**Later targeted repair:** The bridge was subsequently corrected without changing the rest of v2. See the [current targeted-patch report](v2_targeted_patch/repair_report.md). This page records the earlier handoff certification result.

**Decision: do not accept v2 or run M1.** The current handoff fixes the wetted pontoon length and STL topology, but the ordinary source-supported hull is now displaced by up to **28.10 mm**. No STEP or STL geometry was changed in this pass. No trajectory data were used.

## Inputs and provenance

The supplied `/Users/aadikenchammanaold/Downloads/surveyor_agent_handoff.zip` has SHA-256 `7fb92cb87fbad3a39bb7f1762c231754c279a9106b95e9aee427f4b3b573e450`. The production files are in its nested `surveyor_surrogate_v2_fixed_meshes.zip`. All three STEP/STL hashes and the nominal coarse/nominal/fine STL hashes match the handoff's `mesh_repair_summary.json`. Exact hashes are in [certification_manifest.json](v2_certification/certification_manifest.json). The original Surveyor STEP hash remains `1490d6a7769c1bd79e1a14cf2fac330ce0b08818105997a9f28ec4331601282a`. All 15 files in the frozen M1 manifest still match their recorded hashes.

The handoff's scripts and pre-fix reports were treated as descriptions of its construction, **not** as certification results. Every numerical result below was recomputed from the current v2 STEP/STL and the previously selected original STEP face group.

## Topology, CAD and mesh agreement

| Variant | Watertight STL | Winding | Components | Boundary/nonmanifold edges | STEP→STL volume difference | STEP/STL centroid difference |
|---|---|---|---:|---:|---:|---:|
| Nominal | PASS | PASS | 2 | 0 / 0 | +0.03051% | 0.514 mm |
| Fuller | PASS | PASS | 2 | 0 / 0 | −0.00819% | 0.563 mm |
| Finer | PASS | PASS | 2 | 0 / 0 | +0.00099% | 0.546 mm |

All have positive volume and no duplicate STL triangles. Gmsh/OpenCascade re-imports each STEP as two positive-volume solids with positive analytical mass. It finds one **pointlike 2.2×10⁻¹³ mm edge per solid** at the pole of the v2 ellipsoidal local cut (nominal FRD x=−403, y=±370, z=224 mm); other boundary curves have two-face incidence. The pointlike edge is not a macroscopic opening, and the corresponding STL has no boundary edge. The handoff reports passing CadQuery/OCP `BRepCheck_Analyzer`, but that claim was not independently rerun here because OCP/FreeCAD is unavailable in this environment. **OCC B-rep validity is therefore not fully certified; second B-rep check is UNAVAILABLE.** See [OCC import topology](v2_certification/occ_import_topology.json), [mesh topology and hydrostatics](v2_certification/hydro_topology.json), and [STEP/STL consistency](v2_certification/step_stl_consistency.json).

## Pontoon-specific source dimensions

The published 1.83 m × 0.91 m whole-vehicle dimensions are a sanity reference only. At each variant's solved waterline, the source comparison uses the original STEP selected-face port pontoon and its intended mirror for the symmetric surrogate. The original source starboard endpoint and width were also measured; see [pontoon dimensions](v2_certification/pontoon_dimensions.json).

| Variant | Source wetted port length | v2 wetted port length | Length error | Beam error | Centerline-spacing error |
|---|---:|---:|---:|---:|---:|
| Nominal | 1745.042 mm | 1743.876 mm | **0.067%** | **0.068%** | **0.249 mm** |
| Fuller | 1745.042 mm | 1743.334 mm | **0.098%** | **0.068%** | **0.186 mm** |
| Finer | 1745.042 mm | 1743.876 mm | **0.067%** | **0.068%** | **0.186 mm** |

All dimension gates pass (<0.5% length/beam, <2 mm spacing). For nominal, the original source wetted aft/forward endpoints are source Z **18.818 / 1763.860 mm**; the current surrogate's are **19.285 / 1763.161 mm**. No global scaling was applied in this pass.

## Trusted-source surface fidelity — blocking failure

The same original port STEP face selection and previously declared exclusions—source Z **325–350 mm** and **900–1200 mm**—were used. The [new distance map](v2_certification/source_distance_map.png) and [area-weighted statistics](v2_certification/source_fidelity.json) are computed from original underwater face triangle centroids and face areas. All three v2 variants give identical ordinary-hull results because their changed pocket surfaces do not affect this region:

| Region | Median | P90 | P95 | P99 | Maximum | Source area >2 mm | Source area >5 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary source-supported hull | 0.0146 mm | 0.259 mm | **2.015 mm** | **12.153 mm** | **28.102 mm** | 0.02535 m² | **0.01328 m²** |

The ordinary-hull **P95 exceeds the 2 mm limit** and the **maximum exceeds the 5 mm limit**. Approximately **2.62% of assessed ordinary source area** exceeds 5 mm. The maximum is at original source `(X,Y,Z)=(-334.126,-130.923,323.868)` mm, or FRD approximately `(x,y,z)=(-0.591,+0.334,+0.131)` m. The modified bridge spans FRD x −625 to −545 mm, corresponding to source Z 290–370 mm. Its largest discrepancy falls **outside** the already documented 325–350 mm exclusion. In source Z 250–325 mm alone, P95 is 18.07 mm and 0.00421 m² exceeds 5 mm; in Z 350–450 mm another 0.00244 m² exceeds 5 mm. The new end extensions also differ from intact source surfaces near the aft tip. [Region diagnostics](v2_certification/source_region_diagnostics.json) list each interval and worst point.

This is not a permitted pocket exception. Nearest-triangle searches with 16 versus 128 candidates agree on the ~29 mm peak in the adjacent excluded zone, ruling out the nearest-neighbor shortcut as the cause. The section bridge looks smoother but replaces a measurable area of trustworthy source hull. Per the handoff's explicit rule, **M1 stops here**.

## Fairness, hydrostatics and convergence

New [five-metric section curves](v2_certification/fairness.png) were calculated from the current watertight v2 meshes at 10 mm stations; the [nominal CSV profile](v2_certification/surveyor-v2-fairness-nominal.csv) and corresponding fuller/finer profiles are in the evidence directory. The repaired ordinary interval source Z 250–450 mm has a maximum adjacent area change of **5.61%**. Larger area changes occur at the modeled pocket transition and tapering ends, so the **sectional fairness gate passes with these declared features**. Fairness does not establish source fidelity.

| Variant | Draft at 52.3 kg, fresh water | Displacement | CB, FRD (m) | Waterplane area (m²) | Waterplane roll / pitch moments (m⁴) |
|---|---:|---:|---|---:|---:|
| Nominal | 0.17048 m | 0.052300 m³ | (−0.05036, 0.00000, 0.14632) | 0.421852 | 0.048787 / 0.086034 |
| Fuller | 0.17034 m | 0.052300 m³ | (−0.05006, 0.00000, 0.14635) | 0.421762 | 0.048776 / 0.086002 |
| Finer | 0.17067 m | 0.052300 m³ | (−0.05074, 0.00000, 0.14626) | 0.421969 | 0.048800 / 0.086075 |

The published ~0.17 m draft sanity check passes without retuning. Nominal→fine differences are **0.0000029%** in total volume, **0.0024 mm** in CB and **0.00022%** in waterplane area, all below the required limits. Coarse/nominal/fine data are in [STEP/STL consistency](v2_certification/step_stl_consistency.json).

## Certification decision

```text
MESH TOPOLOGY                    PASS
STEP/STL CONSISTENCY             PASS
SOURCE PONTOON EXTENTS           PASS
SOURCE FIDELITY                  FAIL (P95 2.015 mm; max 28.102 mm; 0.01328 m² >5 mm)
FAIRNESS                         PASS (sectional curves; source fit fails separately)
B-REP OCC                        FAIL TO CERTIFY (positive solids; direct BRepCheck unavailable)
B-REP SECOND CHECK               UNAVAILABLE
HYDROSTATICS                     PASS
MESH CONVERGENCE                 PASS

SURROGATE ACCEPTED               NO

FROZEN M1 NOMINAL                NOT RUN
FROZEN M1 FULLER                 NOT RUN
FROZEN M1 FINER                  NOT RUN
FORCE-SPREAD                     NOT COMPUTED
GEOMETRY SENSITIVITY             UNDETERMINED

PRIMARY GEOMETRY HASH            NONE (candidate nominal STEP: 594a09235911819bd084f15e50ab70338a3722bda96ded82599e9dbc9c5304e5)
PRIMARY COEFFICIENT PACKAGE HASH NONE
TRAJECTORY VALIDATION            NOT RUN
```

The specific remaining geometry defect is the **ordinary-hull bridge over FRD x −625 to −545 mm**, which needs a localized source-constrained correction. The source-fit failure also warrants checking the new tip extensions and the small local cut before any renewed certification. Another broad reconstruction was not started, and the nominal geometry was not frozen.
