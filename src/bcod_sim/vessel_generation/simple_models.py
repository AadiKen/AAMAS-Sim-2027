"""Conservative, speed-regular passive estimates with explicit method limits."""
from __future__ import annotations

import math
import os
import numpy as np
import trimesh

from .simple_crossflow import _clip_section
from .simple_sections import polygon_properties, section_properties


def classify(mesh: trimesh.Trimesh, draft: float, hull_count: int, speed_max: float,
             displaced_volume: float,
             override: str | None = None) -> dict:
    length, beam, _ = mesh.extents
    lb = length / beam
    froude = speed_max / math.sqrt(9.80665 * length)
    block = displaced_volume / max(length * beam * draft, 1e-12)
    if override:
        kind, confidence = override, "user-provided"
    elif hull_count == 2:
        kind, confidence = "displacement_catamaran", "medium" if froude < .45 else "low"
    elif hull_count > 2:
        kind, confidence = "general_multihull", "low"
    elif lb < 1.5 or block < .08 or block > 1.05:
        kind, confidence = "unknown_exotic", "low"
    elif froude > .8:
        kind, confidence = "planing_candidate", "low"
    elif froude > .45:
        kind, confidence = "semi_displacement", "low"
    elif lb > 7:
        kind, confidence = "slender_displacement_monohull", "medium"
    else:
        kind, confidence = "displacement_monohull", "medium"
    return {"classification": kind, "confidence": confidence,
            "supporting_metrics": {"length_beam": float(lb), "beam_draft": float(beam / draft),
                                   "max_froude": float(froude), "hull_count": hull_count,
                                   "block_coefficient": float(block),
                                   "volume_length_cubed": float(displaced_volume / length ** 3)},
            "method": "geometry_froude_heuristic_v1"}


def extract_stations(mesh: trimesh.Trimesh, waterline: float, *, count: int = 31) -> list[dict]:
    lo, hi = mesh.bounds[:, 0]
    dx = (hi - lo) / count
    stations = []
    for x in np.linspace(lo + dx / 2, hi - dx / 2, count):
        loops = section_properties(mesh, 0, float(x))["loops"]
        for loop in loops:
            coords = _clip_section(loop, waterline)
            if len(coords) < 3:
                continue
            beam = float(np.ptp(coords[:, 1]))
            draft = float(coords[:, 2].max() - waterline)
            if beam > 1e-6 and draft > 1e-6:
                area, yc, _, _, _ = polygon_properties(coords, 1, 2)
                stations.append({"x_m": float(x), "y_m": yc,
                                 "beam_m": beam, "draft_m": draft, "dx_m": float(dx),
                                 "section_area_m2": float(area),
                                 "section_family": "unknown"})
    if len(stations) < 3:
        raise ValueError("Hull cannot be sectioned for strip/cross-flow baseline")
    return stations


def strip_added_mass(stations: list[dict], density: float) -> dict:
    """Area-informed equivalent-ellipse strip estimate with analytic ellipse terms."""
    matrix = np.zeros((6, 6))
    for station in stations:
        x, b, t, dx = (station[k] for k in ("x_m", "beam_m", "draft_m", "dx_m"))
        a = b/2
        if "section_area_m2" in station:
            area = float(station["section_area_m2"])
            if not math.isfinite(area) or area <= 0 or area > b*t*(1+1e-8):
                raise ValueError("Section area must be positive and no larger than its bounding box")
            c = min(t/2, area/(math.pi*a))
        else:
            c = t/2
        sway = density * math.pi * c**2 * dx
        heave = density * math.pi * a**2 * dx
        vy = np.array((0., 1., 0., 0., 0., x))
        vz = np.array((0., 0., 1., 0., -x, 0.))
        matrix += sway * np.outer(vy, vy) + heave * np.outer(vz, vz)
        # An elliptic section gives only a rough roll inertia without appendages.
        matrix[3, 3] += density * math.pi / 8 * (a*a-c*c)**2 * dx
    return {"matrix_6x6": matrix.tolist(), "method": "area_equivalent_ellipse_strip_v2",
            "confidence": "low", "source": "strip", "unavailable_terms": ["surge_added_mass", "off_axis_appendage_couplings"],
            "applicability": "CAD section area and beam define equivalent ellipses; section shape and free-surface effects remain approximated"}


def reduce_bem_closed_mesh(mesh: trimesh.Trimesh, target_faces: int) -> tuple[trimesh.Trimesh, dict]:
    """Make a solver-only, topology-preserving decimation of a closed hull.

    The canonical geometry and hydrostatics mesh are never replaced. Volume,
    wetted area, principal extents, component count, watertightness and winding
    are checked before this solver mesh is admitted.
    """
    if target_faces < 100:
        raise ValueError("BEM decimation target must contain at least 100 faces")
    parts=mesh.split(only_watertight=True)
    if len(parts)!=len(mesh.split(only_watertight=False)):
        raise ValueError("BEM source has an open or non-watertight component")
    weights=np.asarray([max(float(p.area),1e-12) for p in parts]);budgets=np.maximum(40,np.floor(target_faces*weights/weights.sum()).astype(int))
    # Preserve the requested approximate total while allocating no fewer than
    # 40 panels to any disconnected physical pontoon.
    budgets[-1]+=max(0,target_faces-int(budgets.sum()))
    reduced=[]
    for part,budget in zip(parts,budgets):
        if len(part.faces)<=budget:
            reduced.append(part.copy());continue
        candidate=part.simplify_quadric_decimation(face_count=int(budget))
        candidate.merge_vertices(digits_vertex=10)
        candidate.remove_unreferenced_vertices();candidate.fix_normals()
        if not candidate.is_watertight or not candidate.is_winding_consistent or candidate.volume<=0:
            raise ValueError("BEM decimation broke closed component topology")
        reduced.append(candidate)
    out=trimesh.util.concatenate(reduced);out.fix_normals()
    volume_error=abs(float(out.volume)-float(mesh.volume))/max(abs(float(mesh.volume)),1e-12)
    area_error=abs(float(out.area)-float(mesh.area))/max(float(mesh.area),1e-12)
    extent_error=np.max(np.abs(out.extents-mesh.extents)/np.maximum(mesh.extents,1e-12))
    if len(out.split(only_watertight=True))!=len(parts) or not out.is_watertight or not out.is_winding_consistent:
        raise ValueError("BEM decimated mesh component/topology gate failed")
    if volume_error>.01 or area_error>.03 or extent_error>.005:
        raise ValueError(f"BEM decimation exceeded geometry error gates (volume={volume_error:.2%}, area={area_error:.2%}, extent={extent_error:.2%})")
    return out,{"target_faces":int(target_faces),"actual_faces":len(out.faces),"components":len(parts),
        "watertight":bool(out.is_watertight),"winding_consistent":bool(out.is_winding_consistent),
        "volume_relative_error":volume_error,"surface_area_relative_error":area_error,
        "max_extent_relative_error":float(extent_error)}


def _wetted_capytaine_panels(mesh: trimesh.Trimesh, waterline_frd: float, *,
                             omit_waterline_cap: bool = False,
                             filter_tiny_slivers: bool = False,
                             return_quality: bool = False):
    """Clip the wetted hull, optionally excluding a mesh's artificial waterline closure."""
    transform = np.array((1., -1., -1.))
    # STL is float32. Points intended to lie on the waterline can return a few
    # nanometres above/below it, producing nearly zero-area clipping slivers.
    snap_tolerance = 1e-7 * max(float(mesh.extents.max()), 1.)
    vertices = []
    faces = []
    for triangle in mesh.triangles:
        if omit_waterline_cap and np.all(np.abs(triangle[:, 2] - waterline_frd) <= snap_tolerance):
            continue
        poly = [p * transform + np.array((0., 0., waterline_frd)) for p in triangle]
        for point in poly:
            if abs(point[2]) <= snap_tolerance:
                point[2] = 0.
        clipped = []
        for p, q in zip(poly, poly[1:] + poly[:1]):
            inside_p, inside_q = p[2] <= 0, q[2] <= 0
            if inside_p:
                clipped.append(p)
            if inside_p != inside_q:
                clipped.append(p + (-p[2] / (q[2] - p[2])) * (q - p))
        if len(clipped) < 3:
            continue
        for j in range(1, len(clipped) - 1):
            tri = [clipped[0], clipped[j], clipped[j + 1]]
            if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 1e-12:
                continue
            start = len(vertices)
            vertices.extend(tri)
            faces.append((start, start + 1, start + 2, start + 2))
    vertices=np.asarray(vertices);faces=np.asarray(faces,dtype=int)
    quality={"input_clipped_panels":len(faces),"filtered_sliver_panels":0,
             "filtered_sliver_area_fraction":0.}
    if filter_tiny_slivers and len(faces):
        triangles=vertices[faces[:,:3]]
        edge_lengths=np.stack([np.linalg.norm(triangles[:,j]-triangles[:,(j+1)%3],axis=1)
                               for j in range(3)],axis=1)
        aspect=edge_lengths.max(axis=1)/np.maximum(edge_lengths.min(axis=1),1e-15)
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                      triangles[:,2]-triangles[:,0]),axis=1)/2
        slivers=aspect>200.
        quality["filtered_sliver_panels"]=int(slivers.sum())
        quality["filtered_sliver_area_fraction"]=float(areas[slivers].sum()/max(areas.sum(),1e-15))
        if quality["filtered_sliver_area_fraction"]>1e-3:
            raise ValueError("High-aspect BEM slivers carry more than 0.1% of wetted area")
        faces=faces[~slivers]
    return (vertices,faces,quality) if return_quality else (vertices,faces)


def validate_bem_matrix(matrix: np.ndarray, strip: dict | None) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError("Non-finite or malformed BEM matrix")
    asymmetry = np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-12)
    if asymmetry > .1:
        raise ValueError(f"Added-mass asymmetry {asymmetry:.1%}")
    matrix = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < -1e-5 * max(np.linalg.norm(matrix), 1):
        raise ValueError("Added-mass matrix has negative eigenvalue")
    active = eigenvalues[eigenvalues > 1e-6 * max(eigenvalues.max(), 1.)]
    if len(active) >= 2 and active.max() / active.min() > 1e8:
        raise ValueError("BEM matrix is severely ill-conditioned on active modes")
    comparison = {}
    if strip is not None:
        baseline = np.asarray(strip["matrix_6x6"])
        for axis, name in ((1, "Sway"), (2, "Heave"), (3, "Roll"), (4, "Pitch"), (5, "Yaw")):
            if baseline[axis, axis] > 1e-9:
                comparison[name] = float(matrix[axis, axis] / baseline[axis, axis])
        if any(comparison[name] < .3 or comparison[name] > 3. for name in ("Sway", "Heave", "Yaw") if name in comparison):
            raise ValueError(f"BEM/strip diagonal disagreement: {comparison}")
    return matrix, comparison


def hmri_total_sway_prime(mass_kg: float, sway_added_mass_kg: float,
                          density_kg_m3: float, length_m: float, draft_m: float) -> float:
    """(m+m_y)' under the HMRI L,T,U maneuvering force/acceleration scales.

    Y scales with 0.5 rho L T U² and v_dot with U²/L; U cancels.
    """
    if min(mass_kg, density_kg_m3, length_m, draft_m) <= 0:
        raise ValueError("HMRI scales must be positive")
    return (mass_kg + sway_added_mass_kg) / (.5 * density_kg_m3 * length_m**2 * draft_m)


def bem_frame_matrix(matrix: np.ndarray) -> np.ndarray:
    """Map Capytaine's forward-port-up wrench/velocity matrix to FRD."""
    sign = np.diag((1., -1., -1., 1., -1., -1.))
    return sign @ np.asarray(matrix) @ sign


def panel_resolution_change(coarse: np.ndarray, fine: np.ndarray, *, reference_length_m: float = 1.) -> dict:
    indices = {"A11": 0, "A22": 1, "A33": 2, "A44": 3, "A55": 4, "A66": 5}
    changes = {name: float(abs(fine[i, i] - coarse[i, i]) / max(abs(fine[i, i]), 1e-12))
               for name, i in indices.items()}
    q=np.diag((1.,1.,1.,1/reference_length_m,1/reference_length_m,1/reference_length_m))
    coarse_nd=q@np.asarray(coarse)@q;fine_nd=q@np.asarray(fine)@q
    matrix_change=float(np.linalg.norm(fine_nd-coarse_nd)/max(np.linalg.norm(fine_nd),1e-12))
    maximum=max(*changes.values(),matrix_change)
    return {"relative_changes": changes, "nondimensional_matrix_relative_change":matrix_change,
            "maximum_relative_change": maximum,"stable": maximum <= .15}


def try_bem(mesh: trimesh.Trimesh, strip: dict | None, *, enabled: bool,
            waterline_frd: float, density: float,
            refined_mesh: trimesh.Trimesh | None = None,
            formulation_revision: int = 1,
            required_resolution_change: float = .15,
            maximum_panel_count: int = 1200) -> dict:
    if not enabled:
        return {"status": "disabled", "reason_code": "BEM_DISABLED_BY_GENERATION_SETTING",
                "method": "capytaine", "confidence": "unavailable"}
    try:
        import capytaine as cpt
    except (ImportError, OSError, PermissionError) as exc:
        return {"status": "unavailable", "reason_code": "SOLVER_FAILURE",
                "reason": f"Capytaine import failed: {type(exc).__name__}: {exc}", "method": "capytaine",
                "confidence": "unavailable"}
    try:
        names = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
        solver = cpt.BEMSolver()
        def solve(surface: trimesh.Trimesh):
            vertices, faces, clipping_quality = _wetted_capytaine_panels(surface, waterline_frd,
                omit_waterline_cap=formulation_revision >= 2,
                filter_tiny_slivers=formulation_revision >= 2,return_quality=True)
            if len(faces) == 0 or not np.isfinite(vertices).all():
                raise ValueError("Wetted-body clipping produced no finite panels")
            if len(faces) < 20 or len(faces) > maximum_panel_count:
                raise ValueError(f"Wetted panel count {len(faces)} outside guarded BEM range 20–{maximum_panel_count}")
            edges = vertices[faces[:, :3]]
            lengths = np.stack([np.linalg.norm(edges[:, j] - edges[:, (j+1)%3], axis=1) for j in range(3)], axis=1)
            aspect = float((lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-12)).max())
            if aspect > 200:
                raise ValueError(f"Excessive wetted-panel aspect ratio {aspect:.1f}")
            panel_mesh = cpt.Mesh(vertices=vertices, faces=faces)
            tol = 1e-7 * max(float(surface.extents.max()), 1.)
            # A submerged body ending at the design waterline is still open at
            # the free surface. Generate Capytaine's numerical lid from this
            # open boundary; the excluded hydrostatic cap never enters the body mesh.
            piercing = bool(surface.bounds[0, 2] < waterline_frd + tol and
                            surface.bounds[1, 2] >= waterline_frd - tol)
            lid = panel_mesh.generate_lid(z=0.0) if piercing else None
            if piercing and (lid is None or lid.nb_faces == 0):
                raise ValueError("Surface-piercing hull produced no irregular-frequency lid")
            body = cpt.FloatingBody(mesh=panel_mesh, lid_mesh=lid,
                dofs=cpt.rigid_body_dofs(rotation_center=(0., 0., 0.)))
            results = [solver.solve(cpt.RadiationProblem(body=body, radiating_dof=name,
                omega=0.0, rho=density)) for name in names]
            a = bem_frame_matrix(np.array([[float(result.added_masses[influenced]) for result in results]
                      for influenced in names]))
            b = bem_frame_matrix(np.array([[float(result.radiation_dampings[influenced]) for result in results]
                      for influenced in names]))
            if not np.isfinite(b).all():
                raise ValueError("Non-finite radiation damping")
            raw_symmetry = float(np.linalg.norm(a - a.T) / max(np.linalg.norm(a), 1e-12))
            a, comparison = validate_bem_matrix(a, None if formulation_revision >= 2 else strip)
            return a, b, comparison, {"wetted_panels": len(faces), "lid_panels": 0 if lid is None else lid.nb_faces,
                "lid_forces_excluded": lid is not None and body.mesh.nb_faces == len(faces),
                "maximum_panel_edge_ratio": aspect,
                "raw_symmetry_relative": raw_symmetry,**clipping_quality}
        a, b, comparison, coarse_quality = solve(mesh)
        refinement = None
        if refined_mesh is not None:
            fine, _, _, fine_quality = solve(refined_mesh)
            refinement = panel_resolution_change(a, fine, reference_length_m=max(float(mesh.extents[0]),1e-6))
            refinement["coarse"] = coarse_quality
            refinement["fine"] = fine_quality
        severe = any(comparison[name] < .3 or comparison[name] > 3.
                     for name in ("Sway", "Heave", "Yaw") if name in comparison)
        if severe:
            raise ValueError(f"Severe strip disagreement: {comparison}")
        if refinement is not None and refinement["maximum_relative_change"] > required_resolution_change:
            raise ValueError(f"Panel-resolution change exceeds {required_resolution_change:.1%}: {refinement['relative_changes']}")
        moderate = any(r < .7 or r > 1.3 for r in comparison.values())
        return {"status": "accepted", "reason_code": "BEM_SUCCESS", "method": f"capytaine_radiation_v{formulation_revision}", "version": getattr(cpt, "__version__", "unknown"),
                "frequencies_rad_s": [0.0], "added_mass_6x6": [a.tolist()],
                "radiation_damping_6x6": [b.tolist()],
                "strip_agreement_ratios": comparison, "panel_count": coarse_quality["wetted_panels"],
                "panel_quality": coarse_quality, "panel_resolution": refinement,
                "confidence": "high" if refinement and not moderate else "medium",
                "maneuvering_selection": "zero-frequency radiation added mass; radiation damping excluded"}
    except Exception as exc:
        message=str(exc);lower=message.lower()
        reason_code=("INVALID_WETTED_SURFACE" if "wetted-body clipping" in lower else
                     "MESH_FAILURE" if any(x in lower for x in ("panel", "mesh", "lid", "strip disagreement", "eigenvalue", "asymmetry", "ill-conditioned")) else
                     "CONVERGENCE_FAILURE" if any(x in lower for x in ("converg", "singular")) else
                     "SOLVER_FAILURE")
        return {"status": "rejected", "reason_code": reason_code, "reason": f"{type(exc).__name__}: {message}", "version": getattr(cpt, "__version__", "unknown"),
                "method": "capytaine", "confidence": "unavailable",
                "strip_fallback_available": strip is not None}


def crossflow(stations: list[dict], density: float, nu: np.ndarray, cd_scale: float = 1.) -> np.ndarray:
    """Physical fluid-on-body wrench, including pure sway/yaw at zero surge."""
    u, v, w, p, q, r = np.asarray(nu, dtype=float)
    if not np.isfinite(nu).all():
        raise ValueError("Non-finite body velocity")
    tau = np.zeros(6)
    projected_total = sum(s["draft_m"] * s["dx_m"] for s in stations)
    mean_local = sum(s["draft_m"] * s["dx_m"] * (v+s["x_m"]*r)
                     for s in stations) / projected_total
    mean_abs_u = sum(s["draft_m"] * s["dx_m"] * abs(u-s["y_m"]*r)
                     for s in stations) / projected_total
    speed2_mean = mean_abs_u**2 + mean_local**2
    translation_weight = mean_local**2/speed2_mean if speed2_mean else 0.
    for s in stations:
        local = v + s["x_m"] * r
        ratio = s["beam_m"] / (2 * s["draft_m"])
        cd = s.get("cd", float(np.clip(1.5 / math.sqrt(max(ratio, .05)), .55, 2.6))) * cd_scale
        drag = -.5 * density * cd * s.get("lateral_projected_depth_m", s["draft_m"]) * abs(local) * local * s["dx_m"]
        local_u = u-s["y_m"]*r
        lift = -s.get("lift_base_kg_per_m", 0.) * abs(local_u) * local
        if s.get("lateral_force_method") == "translation_shear_v5":
            drag = -.5 * density * cd * s.get("lateral_projected_depth_m", s["draft_m"]) * local * (
                translation_weight*abs(local) + (1-translation_weight)*abs(local-mean_local)) * s["dx_m"]
            lift *= 1-translation_weight
        elif s.get("lateral_force_method") in {"incidence_blend_v3", "shear_incidence_blend_v4"}:
            speed2 = local_u**2 + local**2
            weight = local**2/speed2 if speed2 else 0.
            if s.get("lateral_force_method") == "shear_incidence_blend_v4":
                shear2 = (local-mean_local)**2
                shear_weight = shear2/(shear2+mean_local**2) if shear2+mean_local**2 else 0.
                weight = 1-(1-weight)*(1-shear_weight)
            drag, lift = weight*drag, (1-weight)*lift
        force = drag + lift
        tau[1] += force
        tau[5] += (s["x_m"] - s.get("x_reference_m", 0.)) * force
        if "axial_rotation_area_m2" in s:
            rotational_u = -s["y_m"] * r
            axial = (-.5*density*s.get("axial_rotation_cd", 1.)*
                     s["axial_rotation_area_m2"]*abs(rotational_u)*rotational_u)
            tau[5] -= s["y_m"]*axial
    return tau


def resistance_curve(speeds: np.ndarray, *, length: float, wetted_area: float,
                     density: float, classification: str, viscosity: float = 1.05e-6,
                     wave: dict | None = None) -> dict:
    if wave is not None and len(wave["wave_resistance_n"]) != len(speeds):
        raise ValueError("Wave resistance must match sampled speeds")
    values = []
    components = []
    for index, u in enumerate(speeds):
        speed = abs(float(u))
        if speed == 0:
            values.append(0.)
            components.append({"friction_n": 0., "wave_n": 0., "form_factor": 1.})
            continue
        reynolds = max(speed * length / viscosity, 1e3)
        ittc_applicable = 1e5 <= reynolds <= 1e9
        cf = .075 / (math.log10(reynolds) - 2) ** 2 if reynolds >= 1e5 else 1.328 / math.sqrt(reynolds)
        form = 1.
        friction = .5 * density * wetted_area * cf * speed * speed
        wave_force = 0. if wave is None else float(wave["wave_resistance_n"][index])
        drag = friction + wave_force
        components.append({"friction_n": friction, "wave_n": wave_force,
                           "form_factor": form, "form_increment_n": 0.,
                           "other_corrections_n": 0., "reynolds_number": reynolds,
                           "friction_coefficient_ittc57": cf,
                           "friction_method": "ittc57" if reynolds >= 1e5 else "laminar_plate_extrapolation",
                           "ittc57_applicable": ittc_applicable})
        values.append(-math.copysign(drag, u))
    return {"speed_mps": [float(x) for x in speeds], "force_x_n": values,
            "components": components, "wave": wave,
            "method": "ittc57_friction_plus_guarded_michell_v3" if wave else "ittc57_friction_line_v3", "confidence": "low",
            "source": "analytical_empirical", "reverse_behavior": "odd extension; unvalidated",
            "provider": {"name":"ITTC-1957 smooth-plate correlation line",
                         "form_factor":"not applied; requires measured or validated-CFD input",
                         "applicability":"friction only; guarded Michell wave term only in thin-ship domain",
                         "reynolds_gate":"1e5 <= Re <= 1e9; outside values are reported as extrapolated",
                         "standard_source":"ITTC Recommended Procedure 7.5-02-02-01, Resistance Tests"},
            "limitations": ("Michell thin-ship theory excludes viscous pressure, planing, trim, and shallow-water effects"
                            if wave else "No wave calculation in this regime; form factor, viscous pressure, transom, trim, planing, and multihull interference are not estimated")}


def inoue_linear_maneuvering(*, mass_kg: float, density: float, length_m: float,
                             draft_m: float, block_coefficient: float) -> dict:
    """Inoue-Hirano-Kijima (1981) even-keel linear maneuvering derivatives."""
    vals=(mass_kg,density,length_m,draft_m,block_coefficient)
    if not all(math.isfinite(x) and x>0 for x in vals): raise ValueError("Inoue inputs must be positive and finite")
    if not 8. <= length_m/draft_m <= 20.: raise ValueError("Inoue derivatives gated to 8 <= L/T <= 20 displacement hulls")
    if not .4 <= block_coefficient <= .85: raise ValueError("Inoue derivatives gated to block coefficient 0.4..0.85")
    k=2*draft_m/length_m; mp=mass_kg/(.5*density*length_m**2*draft_m)
    deriv={"Y_v":-.5*math.pi*k-.7*mp,"Y_r":.25*math.pi*k,"N_v":-k,"N_r":-.54*k+k*k}
    fy=.5*density*length_m*draft_m; mn=fy*length_m; d=np.zeros((6,6))
    d[1,1]=-fy*deriv["Y_v"];d[1,5]=-fy*length_m*deriv["Y_r"]
    d[5,1]=-mn*deriv["N_v"];d[5,5]=-mn*length_m*deriv["N_r"]
    eig=np.linalg.eigvalsh((d+d.T)/2)
    if eig.min() < -1e-10*max(1.,np.linalg.norm(d)): raise ValueError("Inoue derivatives are not passive")
    return {"method":"inoue_hirano_kijima_1981_even_keel_v1","confidence":"low",
            "applicability":"single displacement monohull; even keel; 8<=L/T<=20; 0.4<=Cb<=0.85",
            "source":"Inoue, Hirano & Kijima (1981), DOI 10.3233/ISP-1981-2832103",
            "k":k,"mass_prime":mp,"derivatives_prime":deriv,
            "damping_matrix_per_mps":d.tolist(),"symmetric_part_eigenvalues":eig.tolist()}


def inertia_estimate(mass: float, extents: np.ndarray) -> np.ndarray:
    x, y, z = extents
    return np.diag(mass / 12 * np.array((y*y + z*z, x*x + z*z, x*x + y*y)))
