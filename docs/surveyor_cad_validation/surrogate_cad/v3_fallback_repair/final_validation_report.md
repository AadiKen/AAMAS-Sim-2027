# SeaRobotics Surveyor-derived CAD — geometry-repair fallback

**Decision: accepted as repaired, Surveyor-derived hydrodynamic CAD for the uncalibrated coefficient pipeline.** This is not manufacturer-native or manufacturer-certified CAD. No trajectory data or manually tuned coefficients were used. The canonical files are [STEP](frozen/SeaRobotics_Surveyor_derived_geometry_repair_fallback.STEP) and [STL](frozen/SeaRobotics_Surveyor_derived_geometry_repair_fallback.stl).

## Provenance and repair

The supplied `Mock_Surveyor_1A (1).STEP` has SHA-256 `1490d6a7769c1bd79e1a14cf2fac330ce0b08818105997a9f28ec4331601282a`. The provided handoff ZIP and every pre-fallback v2 and targeted-bridge candidate were copied and hashed in [input_snapshot.json](provenance/input_snapshot.json), with status `pre_fallback_targeted_repair`. The v2 bridge repair had already reduced its local trusted-source P95 to **0.219 mm**, maximum to **3.982 mm**, and area above 5 mm to zero; its geometry outside that bridge was unchanged. The frozen M1 files remained unchanged throughout v3 (all hashes match [the freeze manifest](../../m1_freeze_manifest.json)).

The remaining v2 aft mismatch was not one noisy vertex. On the conformally remeshed v2 STEP, each source pontoon has two connected >5 mm aft components between source Z **18.818 and about 45.3 mm**. Their total affected area is **0.003467 m² port** and **0.003436 m² starboard**, with maxima **28.008/27.990 mm**. The original selected-face shell has **99 open boundary edges per side** in the aft interval, and sectioning finds four disconnected chains at source Z 19–22 mm before the principal chains rejoin at Z 23 mm. There are no local nonmanifold edges or sliver triangles in the sampled source. This supports a malformed/open source termination compounded by the v2 scaled cap. [The component audit](aft_defect_diagnostics.json) records bounds, normals, dihedral statistics and areas; [the before/after source map](figures/source_distance_before_after.png) shows its location. The source tips on both pontoons agree closely enough to use a symmetric repair.

The v3 mask is **FRD X −910 to −820 mm**, equivalent to source Z **5–95 mm**, on both pontoons. It covers **0.018314 m²** of measured underwater source area, **1.307%** of final wetted area. Within this mask each prior pontoon contributes **42 B-rep faces/52 vertices**; the replacement has **48 faces/59 vertices**. The cap uses the surviving original cross-section chains at source Z 18.8, 30, 45, 60, 75 and 90 mm. Eight matched spline edges per section meet the *exact* existing B-rep section wires at both cut planes; the tiny full-length terminal tip is retained. The piecewise ruled loft avoids the overshoot of a global smooth spline. It has zero planar seam faces. [The rejected overshoot and polygon-seam trials](trials/) remain archived rather than overwritten. The rebuilt pocket and all surfaces outside this mask remain the v2 targeted-bridge geometry: authoritative B-rep Boolean differences outside the mask are **0 mm³ added and 0 mm³ removed** for every variant. [The sampled v2→v3 difference map](figures/cad_difference_map.png) provides an independent surface view.

The local keel becomes markedly fuller around source Z 23–30 mm as the original deeper chains emerge. That is a sharp *internal* aft feature, not a seam at the protected boundary. A smooth global interpolant was tested and rejected because it produced multiple overlapping section loops and an implausible tip overshoot. The accepted ruled cap keeps one closed section at every checked station and follows the measured source chains. [Five section metrics](figures/aft_section_fairness.png) and [the underlying data](figures/aft_section_fairness.json) document the residual local chine. At the forward repair boundary, wet adjacent triangle normal changes are **0.224° median, 3.986° P95, 11.19° maximum**; there is no planar closing face. The local patch's largest mesh dihedral is about **105.9°** at the source-derived keel transition near Z 30 mm, so this aft corner should not be described as curvature-continuous. This is a localized source/reconstruction uncertainty, not a hidden fairing operation.

The source-fit exemptions are explicit: v3 aft cap **source Z 5–95 mm**; prior v2 forward cap **Z 1750–1779 mm**; prior v2 tiny local correction **Z 500–525, X −385 to −360, Y −260 to −205 mm**; and the original broken bridge boundary **Z 325–350 mm**. The pre-existing propulsion-pocket review interval **Z 900–1200 mm** remains outside the ordinary-hull source gate. These bounds and their measured source areas are in [geometry_validation.json](geometry_validation.json). The protected ordinary source area is about **0.4943 m² per sampled port side**; it has zero area above the original 5 mm limit.

## Final geometry gates

| Gate | Result | Evidence |
|---|---|---|
| Two closed, positive-volume pontoon solids; B-rep validity | **PASS** | OCP `BRepCheck_Analyzer` for every final solid; Gmsh/OCC imported two solids and generated **403,834** tetrahedral elements for nominal STEP |
| STL watertight, manifold, oriented, two components | **PASS** | [Mesh manifest](mesh_manifest.json), independently checked by Trimesh |
| No duplicate or unintended disconnected shells | **PASS** | Two expected pontoon components; conformal STL topology and 3D CAD meshing |
| Trusted source surface fidelity outside declared masks | **PASS** | P95 **0.191 mm**, maximum **4.010 mm**, ordinary area >5 mm **0** for each variant; [full statistics](geometry_validation.json) |
| Source pontoon dimensions | **PASS** | Wetted length error **0.071–0.084%**, beam error **0.0069–0.0097%**, spacing error **0.112–0.113 mm**; limits 0.5%, 0.5%, 2 mm |
| Protected v2 geometry outside aft mask | **PASS** | Both exact B-rep set differences **0 mm³**; sampled outside difference P95 **0.065 mm** from independent remeshing |
| STEP↔STL consistency | **PASS** | Adaptive OCC volume differences **0.011–0.012%** and centroid differences **0.0049–0.0063 mm**; [consistency audit](step_stl_consistency.json) |
| 52.3 kg fresh-water hydrostatics | **PASS** | Draft **0.17104–0.17138 m**, 0.052300 m³ displaced, positive waterplane and restoring terms |
| Aft transition fairness | **PASS with documented source chine** | One closed section throughout, no spline overshoot or planar seam; [plot](figures/aft_section_fairness.png) |

The published **1.83 × 0.91 m** are whole-vehicle dimensions, not wetted-pontoon gates. At the solved nominal waterline, the source port wet length is **1745.042 mm**, symmetric source beam **823.079 mm**, and source centerline spacing **676.042 mm**. The final nominal values are **1743.677 mm**, **823.146 mm**, and **675.929 mm**. The full final solid bounding box remains FRD X **−0.915931 to 0.863760 m**, Y **±0.415100 m**. The published 0.17 m draft is an independent sanity check, not a shape-fitting target.

An OCC default *nonadaptive* volume call underestimates this spline B-rep by about 0.26%; it also shifts the apparent volume centroid by about 1.45 mm. The authoritative comparison uses `BRepGProp.VolumeProperties` with adaptive Gauss integration at `Eps=1e-6`. It converges at **0.095061672 m³** and agrees with a finer, independent STEP tessellation: nominal 4 mm versus 3 mm meshes differ in volume by **0.0053%** and centroid by **0.0021 mm**. No geometry was retuned to reconcile the integration setting. Exhaustive all-triangle collision testing is unavailable in the pipeline; OCP validity, conformal manifold surface meshing and successful two-solid volume meshing detected no self-intersection.

## Effect of the aft repair

Nominal values below compare conformal v2 and final v3 tessellations at identical 52.3 kg fresh-water loading; B-rep volumes and centers are independently recorded with adaptive integration in [the CAD manifest](cad_repair_manifest.json).

| Quantity | v2 before | v3 final | Change |
|---|---:|---:|---:|
| Total enclosed mesh volume | 0.09469446 m³ | 0.09505047 m³ | +0.00035601 m³ (**+0.376%**) |
| Draft | 0.171555 m | 0.171194 m | −0.361 mm |
| Waterplane area | 0.421906 m² | 0.423082 m² | +0.279% |
| CB longitudinal X | −0.049653 m | −0.052042 m | −2.389 mm |
| CB vertical Z | 0.146367 m | 0.146467 m | +0.100 mm |
| Final port/starboard volume asymmetry | — | — | **0.000053%** of total volume |

[Machine-readable patch effects](patch_effects.json) include wetted area and center shifts. The three final variants have nearly identical full extents and waterplanes. Nominal was frozen because it is the central pocket hypothesis and passes topology, source-fit, repair-minimality and hydrostatic checks; no downstream coefficient was used to choose it. Canonical STEP SHA-256 is `f44f0876d0465e3738511f0307a31f28c00c1077cb8c93de3abae45693ec81ee`; canonical STL SHA-256 is `4302564c6b22729e9cc64abd7b040e23426b14fbc75b2215189d83e69b03c5a6`. [Frozen metadata](frozen/geometry_metadata.json) and [all variant hashes](mesh_manifest.json) preserve the exact selection.

## Ordinary CAD → coefficients generation

The unchanged production `bcod_sim.vessel_generation.simple_pipeline.generate_simple_vessel` processed each final STL as a new arbitrary vessel. Inputs: **52.3 kg**, assumed CG **FRD (0,0,0) m**, fresh-water density **1000 kg/m³**, speed range **0–3 m/s**, default confidence policy and 9-sample restoring LUT. There was no Surveyor-specific coefficient branch, manual drag fitting or trajectory calibration. The pipeline inferred rigid inertia and reference dimensions. The nominal reference is **length 1.779691 m, beam 0.830199 m, draft 0.171194 m**. The [standard nominal coefficient package](coefficients/nominal/coefficient_package.yaml), [generator provenance](coefficients/nominal/provenance.json), and [coefficient manifest](coefficient_manifest.json) identify code hash, input geometry hash, frames, ownership and assumptions. There is no actuator geometry in this CAD package; the yaw open-loop test below applies an external diagnostic moment rather than inventing thrusters.

The package defines body velocity, force and moment in **FRD** about reference point **(0,0,0) m**. Plant6 owns rigid and added-mass Coriolis terms; the runtime payload owns linear damping, nonlinear surge resistance, sectional cross-flow and restoring terms. The independent evaluator verifies those owners' combined dissipative wrench without double counting. No frame transform is required for the FRD input STL.

Nominal hydrostatics: **0.052300 m³** displacement, CB **FRD (−0.052042, 0, 0.146467) m**, waterplane **0.423082 m²**, positive GM roll **1.0820 m** and pitch **1.8067 m** for the stated CG. Nominal strip added-mass diagonal includes sway **59.33 kg**, heave **43.43 kg**, pitch **8.800 kg·m²**, yaw **10.465 kg·m²**; surge added mass is zero in the strip fallback. The guarded Capytaine BEM route rejects this detailed wetted mesh (**234,174 panels versus its 20–1,200 guard**) and the standard strip estimate runs instead. The generator marks the displacement-catamaran classification and relevant drag/form-factor methods as **low confidence**. These are model-form limitations, not fitted Surveyor parameters.

## M1, dynamics and geometry sensitivity

The frozen M1 package schema, FRD frame, signs, force ownership, positive hydrostatics, passivity and independent NumPy evaluator versus Plant6 all pass. The 343-state force-grid maximum residual is **1.14×10⁻¹³ N/N·m** nominal and at most **1.71×10⁻¹³** across variants. Frozen file hashes match; an independent nominal regeneration is **bytewise identical** with canonical package hash `e4c0a03c43daf4a3c15594f848e6b4c5b3e36f8b1755fce8ebc0e54b3b9efadf`. The full relevant M1 and Plant6 repository test selection passed **76 tests**. [M1 results](m1_validation.json), [determinism proof](determinism_check.json), and each package's `validation.json` contain machine results.

| Major nominal prediction | Value | Fuller–finer spread relative to nominal |
|---|---:|---:|
| Surge resistance X at 1.5 m/s | −10.969 N | 0.445% |
| Sway force Y at 0.3 m/s | −41.051 N | 0.274% |
| Yaw moment N at 0.3 rad/s | −4.950 N·m | 0.354% |
| Added mass sway | 59.328 kg | 0.448% |
| Added mass yaw | 10.465 kg·m² | 0.530% |
| Largest reported major-group spread | — | **1.286%** (roll added mass) |

The complete [variant state-grid comparison](m1_validation.json) includes (M_A), (R(u)), (Y) and (N). Under the project's <10% decision threshold, geometry reconstruction uncertainty is secondary for this coefficient run. It does **not** establish agreement with measured Surveyor trajectories.

The [open-loop report](dynamics_report.json) and [plots](figures/open_loop_sanity.png) pass zero-input equilibrium, straight/reverse surge, external yaw moment, coast-down, small yaw-rate decay, combined surge/yaw, and small heave/roll/pitch restoring perturbations. All states remain finite and bounded; unforced kinetic energy decays. Equilibrium uses the package's solved heave/roll/pitch trim, not an arbitrary zero attitude. The [full trajectories](dynamics_trajectories.json) are retained for inspection.

## Final acceptance table

| Item | Decision |
|---|---|
| SOURCE PONTOON DIMENSIONS | **PASS** |
| SOURCE SURFACE FIDELITY OUTSIDE DECLARED MASKS | **PASS** |
| B-REP AND CONFORMAL MESH TOPOLOGY | **PASS** |
| STEP/STL CONSISTENCY | **PASS** |
| HYDROSTATICS AT 52.3 KG | **PASS** |
| SECTION FAIRNESS AND LOCAL SOURCE CHINE | **PASS WITH DOCUMENTED EXCEPTION** |
| PROTECTED GEOMETRY PRESERVATION | **PASS** |
| FROZEN M1 AND PACKAGE CONTRACT | **PASS** |
| OPEN-LOOP NUMERICAL SANITY | **PASS** |
| SURROGATE ACCEPTED | **YES, AS SURVEYOR-DERIVED REPAIRED CAD** |
| M1 | **RUN; PASS** |

The package is suitable for the next real-trajectory comparison as an **uncalibrated CAD-derived prediction with a measured geometry band**. It is not a physical coefficient-validation result until compared with independent Surveyor observations.
