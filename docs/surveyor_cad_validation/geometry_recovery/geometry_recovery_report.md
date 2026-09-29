# Surveyor geometry recovery — stopped at ambiguous underwater openings

> **Superseded by [constrained-envelope reconstruction](section_reconstruction/constrained_envelope_report.md).** This page records the earlier CAD-sewing attempt. The later section workflow found the principal buoyant surfaces in mixed STEP group 13380 and generated diagnostic M1 packages; source fidelity in the starboard pocket remains unresolved.

## Result

**No watertight hydrodynamic mesh was recovered.** The STEP imports, its inch units are resolved, and likely pontoon faces have been identified. CAD-level sewing fails on an invalid face wire at every tested tolerance. More importantly, both pontoon candidates have large openings in low hull and propulsion-pocket regions. Their missing exterior shape is not uniquely determined by the surviving STEP faces. The recovery brief directs a stop for ambiguous underwater geometry, so no cap, mirrored hull, generic mesh reconstruction, hydrostatics, or coefficient package was made.

## Source and frame

The [source manifest](source/source_manifest.yaml) identifies the AP214 STEP and its SHA-256. Its representation unit is **inch**; OpenCASCADE/Gmsh imports in millimetres. The candidate mapping to FRD is `x=source Z`, `y=-source X`, `z=-source Y`, scaled by 0.001 from the imported millimetres. Bow toward positive source Z follows taper and remains unconfirmed. Translation and final reference origin are unresolved. The full assembly spans roughly 1.8 m longitudinally and 0.9 m laterally; the selected pontoon exterior candidates individually span only about 1.43 m. Dimensions therefore cannot be qualified as a final hull. The [assembly plot](plots/source_assembly_bbox_diagnostic.png) is a component bounding-box diagnostic, while the [retained-surface plot](plots/retained_surfaces_candidate.png) displays sampled candidate surfaces.

## Before-repair topology

The STEP import has 13,970 CAD surfaces and **zero volumes**. The [CAD topology audit](diagnostics_before/cad_topology.json) counts 92 candidate port faces, 81 free CAD curves totaling **4.608 m**, and 0.265 m² of selected surface area. The starboard candidates have 83 faces in two separated regions, 87 free CAD curves totaling **4.663 m**, and 0.236 m² of selected area. These areas are incomplete source surfaces, not wetted areas. [Classification](extraction_manifest.yaml) records source group IDs and exclusions; [full surface-group inventory](source/occ_surface_groups.json) preserves entity tags and bounds.

The diagnostic STL is **not a recovered hull**. Its [mesh topology audit](diagnostics_before/mesh_topology.json) shows four open boundary loops on each pontoon side, plus open edges on excluded above-water rails. The two watertight 58 mm diameter, 432 mm long cylindrical components are propulsion-region pieces and do not close either pontoon. The [free-edge plot](plots/free_edges_before.png) shows the openings in top, side, and front projections. The long planar upper edge lies around source Y = −168 mm. Other loops extend to Y = −238 to −242 mm through the lower hull and pocket region. The starboard exterior is also interrupted longitudinally from approximately Z = 1016 to 1119 mm, spanning much of its vertical section.

## Repair assessment

The [repair manifest](repair_manifest.yaml) classifies upper edges as possible planar closures, pending a reliable waterline. The long lower pocket interfaces, the starboard midbody break, and the port lower midbody opening are **Category D** ambiguous underwater missing geometry. The nominal 45–55 kg waterline envelope cannot be solved from an open shell; those lower openings cannot be certified above it. The missing patch area and percentage of wetted surface cannot be measured reliably until the intended pocket/exterior connectivity is supplied. No underwater surface was invented.

Independent port and starboard `healShapes` trials at 0.01, 0.1, and 1.0 mm all stop with `Could not fix wire in surface 13981`; [sweep results](diagnostics_after/cad_sewing_sweep.json) preserve each attempt. Neither pontoon supplies a complete source exterior for symmetry reconstruction. Mirroring one would replicate the unresolved pocket geometry rather than recover it. Poisson, voxel, and shrink-wrap methods were not used because they would invent precisely the underwater shape that controls displacement and coefficients.

The [unchanged M1 geometry gate](/Users/aadikenchammanaold/Desktop/LEADCAT/Website/AAMAS27-Sim/src/bcod_sim/vessel_generation/simple_geometry.py) rejects the candidate STL with `Submerged geometry must be a closed orientable surface; no underwater reconstruction is attempted`. No coarse/nominal/fine production meshes or deviation comparison exist, and hydrostatics and coefficients remain unrun. The frozen M1 file hashes were verified unchanged after this work.

## Status

| Gate | Status |
| --- | --- |
| SOURCE STEP IMPORT | PASS |
| UNIT RESOLUTION | PASS |
| HULL CLASSIFICATION | FAIL — candidate faces only; exterior connectivity unresolved |
| PORT HULL RECOVERY | FAIL |
| STARBOARD HULL RECOVERY | FAIL |
| SURFACE DEVIATION GATE | NOT RUN — no repaired surface |
| WATERTIGHTNESS | FAIL |
| MESH REFINEMENT STABILITY | NOT RUN |
| 45 KG HYDROSTATICS | NOT RUN |
| 49 KG HYDROSTATICS | NOT RUN |
| 55 KG HYDROSTATICS | NOT RUN |
| FROZEN M1 GEOMETRY GATE | FAIL |
| COEFFICIENT GENERATION | NOT RUN |

The next admissible source is a hull-only CAD export that closes the two pontoon exteriors and explicitly identifies how each propulsion-pocket tube joins the displaced hull volume, with forward direction marked. This source STEP alone cannot support a traceable minimal repair below the likely waterline.

## Reproduction

From the repository root, run the new preprocessing diagnostics with the same STEP and the checked-in `source/occ_surface_groups.json` inventory:

```sh
PYTHONPATH=src .venv/bin/python tools/diagnose_surveyor_cad_recovery.py 'PATH/Mock_Surveyor_1A (1).STEP' docs/surveyor_cad_validation/geometry_recovery/source/occ_surface_groups.json docs/surveyor_cad_validation/geometry_recovery/diagnostics_before/cad_topology.json
PYTHONPATH=src .venv/bin/python tools/try_surveyor_cad_sewing.py 'PATH/Mock_Surveyor_1A (1).STEP' docs/surveyor_cad_validation/geometry_recovery/source/occ_surface_groups.json docs/surveyor_cad_validation/geometry_recovery/diagnostics_after/cad_sewing_sweep.json
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/manta-mpl .venv/bin/python tools/plot_surveyor_open_boundaries.py docs/surveyor_cad_validation/geometry/pontoon_candidate_OPEN_NOT_FOR_COEFFICIENTS.stl docs/surveyor_cad_validation/geometry_recovery/diagnostics_before
```
