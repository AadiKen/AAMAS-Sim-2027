# Surveyor propulsion-pocket face attribution

**Provenance update:** The later [raw AP214 provenance analysis](provenance/provenance_report.md) directly matched the two large conical faces to the same original open shell as broad pontoon skin. They are now attributed to the hull/recess component with **moderate** confidence. H1/H2/H3 meshes and force spread have still not been generated. The evidence plots and underwater-area figures below were corrected to the frozen hydrostatic waterline at source `Y≈−78.27 mm` (FRD `z=0.07827 m`), distinct from the 0.16905 m keel-to-waterline draft.

## Decision

**Raw STEP provenance attributes the two large starboard conical faces to the hull/recess shell, but does not define the final water-accessible boundary.** The [manufacturer's specification](https://okeanus.com/wp-content/uploads/2025/08/sr-surveyorm18-specsheet-new01.pdf) identifies two 1 kW BLDC *pocket thrusters*. The STEP is an open surface assembly, not a closed displaced hull solid. H1/H2/H3 have not been generated as source-preserving physical hypotheses, and no new force-spread band or primary coefficient package is claimed.

The previous smooth/inset/outset meshes and packages vary only a bridge in a mirrored port-section loft. Their narrow coefficient spread does **not** bound the starboard attribution uncertainty. They remain diagnostic reference artifacts. No trajectory data was used or replayed here.

## Source and face inventory

The original AP214 STEP SHA-256 is `1490d6a7769c1bd79e1a14cf2fac330ce0b08818105997a9f28ec4331601282a`. Faces were imported in millimetres from the STEP's inch units. [The inventory](face_inventory.csv) records 93 original OCC faces with tessellated material in source `X>0`, `Z=900–1200 mm`, `Y<=0`. IDs `SF001`–`SF093` are deterministic for this file and import. Each row includes original face area, local pocket-patch area, centroid, bounding box, analytic surface type, tessellated principal normal and angular-spread proxy, neighbors within the inventory, connected-group root, approximate distance from the mirrored port loft, nominal-waterline relation, and OCC-import color. The [machine-readable inventory](face_inventory.json) and [per-face triangles](face_triangles.npz) retain the evidence.

**Inventory limit:** Gmsh's OCC face tags are source import identifiers, not STEP `#` entity numbers. The later raw STEP parser recovered 68 unique face IDs directly and assigned all 93 to a shell, with 25 exact face IDs still unresolved; these mappings are in [face_to_component.csv](provenance/face_to_component.csv), while the original inventory remains a record of the kernel import. AP214 layer/style names are empty. All 93 OCC-import colors are the same `(0,0,255,0)` and do not separate hull from hardware. Mirror distances are one-sided triangle-centroid to nearest mesh-vertex estimates, so they indicate mismatch size rather than exact surface-to-surface Hausdorff distance. Principal normals are tessellation winding directions; the open source shell gives no trustworthy global solid orientation. The angular spread is a curvature proxy, not a CAD principal-curvature measurement.

## Geometric evidence

The face-ID [external](renders/face_ids_external.png), [internal](renders/face_ids_internal.png), [stern](renders/stern.png), [underside](renders/underside.png), [plan overlay](renders/mirrored_overlay.png), and [perspective](renders/perspective.png) views show original source faces against the mirrored port envelope and 52.3 kg reference waterline. [Transverse and longitudinal cuts](renders/sections/) show that the original starboard material has interrupted/branching section curves near the pocket; a single closed contour cannot be selected without a physical interpretation. The [analytical exploded view](renders/exploded_feature.png) offsets the two conical faces only for inspection; it does **not** depict separate CAD components.

`SF057` (OCC tag 13344) and `SF058` (tag 13345) are paired conical patches of 0.0303 and 0.0304 m². About 0.0420 m² of their combined local sampled area is below the diagnostic 52.3 kg waterline. Their source points are roughly 50–67 mm from the mirrored envelope in the local nearest-vertex comparison. These are large enough to explain why the earlier mirror failed source fidelity. The surrounding broad B-spline faces `SF037`, `SF046`, `SF051`, and `SF068` are only about 2.4–2.6 mm from that mirror at the local 95th percentile. The two conical patches therefore form a distinct feature, rather than ordinary loft tessellation noise.

The [restricted adjacency graph](topology_graph.json) has one connected group of all 93 reviewed faces. Each cone shares original CAD edges with the other cone and with multiple small transition faces; graph paths lead through those transitions to the broad pontoon faces. Removing both cones leaves the restricted graph connected but exposes 22 candidate shared seams. It does not reveal a separate, self-contained motor body. Raw STEP references then place the cones in the same open shell as broad pontoon skin. The shell's openness still prevents an enclosed-volume or water-access inference.

## Conservative classification

The updated [classification](classification.yaml) assigns 35 faces to `HULL_SKIN`, 13 to `INTERFACE`, and 45 to `UNKNOWN`; none of these original faces has enough evidence for a high-confidence `HARDWARE` label. The two large cone faces are `HULL_SKIN` at moderate *component-attribution* confidence, based on raw STEP shell membership. Their exact wetted role remains to be modeled. Face-level evidence and waterline relation are preserved in YAML. The classification uses no trajectory fit.

| Class | Reviewed source-patch area (m²) | Area below nominal waterline (m²) | Faces |
| --- | ---: | ---: | ---: |
| HULL_SKIN | 0.1412 | 0.1216 | 35 |
| INTERFACE | 0.0478 | 0.00345 | 13 |
| UNKNOWN | 0.0207 | 0.0162 | 45 |

These are original *review-window* face areas, not reconstructed-area or whole-hull wetted-area fractions. Because the hypothesis solids are unavailable, original area retained, area reconstructed, area removed as hardware, and altered wetted fraction for H1/H2/H3 are undefined. Treating zero generated area as zero physical uncertainty would be incorrect.

## H1/H2/H3 qualification

The three intended interpretations and their exact blockers are recorded under [geometry_hypotheses](geometry_hypotheses/). `POCKET_SKIN_MAX` needs a displaced-skin choice and closure for the connected cone/transition network. `POCKET_HARDWARE_MAX` needs a defensible insert boundary and fair hull continuation; the prior mirrored port loft is **not** a qualified H2 because it retains a pocket slot and fails original starboard source fidelity. `POCKET_OPEN_PHYSICAL` needs the water-accessible opening and solid recess boundary. None can presently preserve all trusted original material outside the pocket while satisfying watertight, positive-volume, no-self-intersection, and mesh-refinement gates. No surrogate mesh is labeled as one of these hypotheses.

Accordingly, [hydrostatic comparison](hydrostatics/comparison.csv) has no H1/H2/H3 result. The prior mirrored diagnostic reference solves 52.3 kg at 0.16905 m draft, close to the published 0.17 m, but that check does not classify the conical faces. [Coefficient](comparison/coefficient_spread.csv) and [common-state force](comparison/state_grid_force_spread.csv) tables contain headers and no invented measurements. The [decision manifest](comparison/decision_manifest.json) records why frozen M1 was not invoked on invalid hypotheses.

The current evidence therefore cannot answer quantitatively whether the pocket changes passive M1 hydrodynamics. Geometry force uncertainty is **not yet quantified**, and it has not been demonstrated to exceed 15% in a computed force spread. A closed hull/recess boundary is needed before physical hypotheses can be generated. The manufacturer draft and dimensions alone cannot supply that boundary.

## Frozen M1 and final status

All 15 frozen M1 file SHA-256 hashes match the prior freeze manifest before and after this analysis. The unchanged M1 test command passed **62 tests in 27.62 s**. No geometry gate, coefficient schema, hydrodynamic model, force ownership, or runtime interpretation was changed. This task did not run real trajectory validation or calibration.

| Gate | Status |
| --- | --- |
| FACE INVENTORY | **PARTIAL** — 93 stable OCC IDs; 68 unique STEP face IDs; all mapped to a shell; 25 exact face IDs unresolved |
| FACE CLASSIFICATION | **PARTIAL** — paired cones attributed to hull/recess at moderate confidence; 45 other faces remain unknown |
| H1 WATERTIGHT | **FAIL** — no defensible source-preserving solid |
| H2 WATERTIGHT | **FAIL** — no defensible hardware boundary/continuation |
| H3 WATERTIGHT | **FAIL** — water-accessible recess boundary unresolved |
| 52.3 KG HYDROSTATICS | **FAIL for H1/H2/H3**; prior mirror diagnostic draft 0.16905 m |
| FROZEN M1 H1 | **FAIL / NOT RUN** — no valid mesh |
| FROZEN M1 H2 | **FAIL / NOT RUN** — no valid mesh |
| FROZEN M1 H3 | **FAIL / NOT RUN** — no valid mesh |
| FORCE-SPREAD ANALYSIS | **FAIL / NOT RUN** — no three valid physical hypotheses |
| GEOMETRY UNCERTAINTY | **NOT YET QUANTIFIED**; no numeric >15% finding |
| PRIMARY PACKAGE | **NONE** |
| TRAJECTORY VALIDATION | **NOT RUN** |
