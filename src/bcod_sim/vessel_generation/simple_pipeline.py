"""Default CFD-free mesh-to-passive-vessel path.

Every unsupported high-fidelity method is recorded as unavailable; it is never
silently replaced with a supposedly validated coefficient.
"""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh
import yaml

from .simple_geometry import prepare_geometry
from .simple_crossflow import extract_crossflow_stations
from .simple_wave import guarded_wave_curve, thin_ship_applicable
from .simple_hydro_mesh import reduce_hydrodynamic_mesh, mesh_hash
from .simple_models import (classify, crossflow, extract_stations, inertia_estimate,
                            resistance_curve, strip_added_mass, try_bem)
from .simple_sections import hydrostatic_state, solve_waterline
from .models import CanonicalVessel, ParameterLineage

SCHEMA = "bcod-simple-hydrodynamics-v1"
MODEL_ID = "bcod-passive-phase3a-1.0.0"


def _write(root: Path, name: str, value: dict) -> None:
    (root / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rotation(roll: float, pitch: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([[cp, sr * sp, cr * sp], [0., cr, -sr], [-sp, sr * cp, cr * cp]])


def _restoring_lut(mesh: trimesh.Trimesh, waterline: float, mass: float,
                   cg: np.ndarray, density: float, gravity: float, count: int,
                   motion_fraction: float = .2,
                   center: tuple[float, float, float] = (0., 0., 0.)) -> dict:
    draft = float(mesh.bounds[1, 2] - waterline)
    heaves = np.linspace(-motion_fraction * draft, motion_fraction * draft, count) + center[0]
    rolls = np.linspace(-.15, .15, count) + center[1]
    pitches = np.linspace(-.10, .10, count) + center[2]
    data = np.zeros((count, count, count, 6))
    for i, heave in enumerate(heaves):
        for j, roll in enumerate(rolls):
            for k, pitch in enumerate(pitches):
                data[i, j, k] = _exact_restoring_load(mesh, waterline, mass, cg, density,
                                                       gravity, float(heave), float(roll), float(pitch))
    return {"model": "trilinear_restoring_lut", "axes": {"heave_m": heaves.tolist(),
            "roll_rad": rolls.tolist(), "pitch_rad": pitches.tolist()}, "wrench_frd": data.tolist(),
            "out_of_domain": "error", "interpolation": "trilinear",
            "method": "exact_triangle_clip_v1", "confidence": "medium"}


def _local_restoring_lut(stiffness: np.ndarray, draft: float) -> dict:
    """Build a small linear restoring neighborhood without inventing topsides."""
    heave_limit = max(min(.01 * draft, .01), 1e-4)
    roll_limit = pitch_limit = .01
    axes = (np.linspace(-heave_limit, heave_limit, 3),
            np.linspace(-roll_limit, roll_limit, 3),
            np.linspace(-pitch_limit, pitch_limit, 3))
    grid = np.zeros((3, 3, 3, 6), dtype=float)
    for i, heave in enumerate(axes[0]):
        for j, roll in enumerate(axes[1]):
            for k, pitch in enumerate(axes[2]):
                displacement = np.array((0., 0., heave, roll, pitch, 0.))
                grid[i, j, k] = -(stiffness @ displacement)
    return {"model": "local_design_waterline_linear_restoring_lut",
            "axes": {"heave_m": axes[0].tolist(), "roll_rad": axes[1].tolist(),
                     "pitch_rad": axes[2].tolist()}, "wrench_frd": grid.tolist(),
            "out_of_domain": "error", "interpolation": "trilinear",
            "method": "waterplane_local_stiffness_v1", "confidence": "medium",
            "mode": "local_design_waterline", "nonlinear_vertical_motion_supported": False,
            "validity": {"heave": "local", "roll": "local", "pitch": "local"}}


def _exact_restoring_load(mesh: trimesh.Trimesh, waterline: float, mass: float,
                          cg: np.ndarray, density: float, gravity: float,
                          heave: float, roll: float, pitch: float) -> np.ndarray:
    rot = _rotation(roll, pitch)
    transformed = mesh.copy()
    transformed.vertices = mesh.vertices @ rot.T + np.array((0., 0., heave))
    if not transformed.bounds[0, 2] < waterline < transformed.bounds[1, 2]:
        raise ValueError("Restoring state is fully emerged or submerged")
    state = hydrostatic_state(transformed, waterline, density=density,
                              include_wetted=False, include_waterplane=False)
    buoy_world = np.array((0., 0., -density * gravity * state["volume_m3"]))
    weight_body = rot.T @ np.array((0., 0., mass * gravity))
    buoy_body = rot.T @ buoy_world
    cb_body = rot.T @ (np.asarray(state["center_buoyancy_frd_m"]) - np.array((0., 0., heave)))
    result = np.zeros(6)
    result[:3] = weight_body + buoy_body
    result[3:] = np.cross(cg, weight_body) + np.cross(cb_body, buoy_body)
    return result


def _exact_restoring_equilibrium(mesh: trimesh.Trimesh, waterline: float,
                                 mass: float, cg: np.ndarray, density: float,
                                 gravity: float) -> tuple[float, float, float]:
    """Solve physical heave/roll/pitch equilibrium before constructing the LUT."""
    point = np.zeros(3)
    scales = np.array((mass*gravity, mass*gravity*max(mesh.extents[1], .01),
                       mass*gravity*max(mesh.extents[0], .01)))
    for _ in range(12):
        force = _exact_restoring_load(mesh, waterline, mass, cg, density, gravity, *point)
        residual = force[[2, 3, 4]]/scales
        if np.max(np.abs(residual)) < 1e-8:
            return tuple(float(x) for x in point)
        jacobian = np.zeros((3, 3))
        for column, step in enumerate((max(mesh.extents[2]*1e-4, 1e-6), 1e-4, 1e-4)):
            shifted = point.copy()
            shifted[column] += step
            jacobian[:, column] = ((_exact_restoring_load(mesh, waterline, mass, cg, density,
                                    gravity, *shifted)[[2, 3, 4]]/scales)-residual)/step
        direction = np.linalg.solve(jacobian, -residual)
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            trial = point + fraction*direction
            if abs(trial[1]) > .25 or abs(trial[2]) > .2:
                continue
            try:
                candidate = _exact_restoring_load(mesh, waterline, mass, cg, density,
                                                   gravity, *trial)[[2, 3, 4]]/scales
            except ValueError:
                continue
            if np.linalg.norm(candidate) < np.linalg.norm(residual):
                point, accepted = trial, True
                break
        if not accepted:
            break
    raise ValueError("No stable hydrostatic equilibrium for supplied mass and CG within trim envelope")


def _restoring_equilibrium(restoring: dict, mass: float, length: float) -> dict:
    """Find zero heave force and roll/pitch moment in the exported LUT.

    The requested CG may differ from CB, so zero trim need not be equilibrium.
    Solve against the same trilinear law used by Plant6, not a separate tangent.
    """
    axes = [np.asarray(restoring["axes"][name], dtype=float)
            for name in ("heave_m", "roll_rad", "pitch_rad")]
    grid = np.asarray(restoring["wrench_frd"], dtype=float)

    def wrench(position: np.ndarray) -> np.ndarray:
        indices, weights = [], []
        for value, axis in zip(position, axes):
            if value < axis[0] or value > axis[-1]:
                raise ValueError("Hydrostatic equilibrium is outside the restoring LUT")
            index = int(np.clip(np.searchsorted(axis, value), 1, len(axis)-1))
            indices.append(index-1)
            weights.append((value-axis[index-1])/(axis[index]-axis[index-1]))
        result = np.zeros(6)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    weight = ((weights[0] if a else 1-weights[0]) *
                              (weights[1] if b else 1-weights[1]) *
                              (weights[2] if c else 1-weights[2]))
                    result += weight*grid[indices[0]+a, indices[1]+b, indices[2]+c]
        return result

    point = np.array([axis[len(axis)//2] for axis in axes], dtype=float)
    target = (2, 3, 4)
    tolerance = 1e-7*mass*9.80665*max(1., length)
    for _ in range(12):
        residual = wrench(point)[list(target)]
        if np.max(np.abs(residual)) <= tolerance:
            break
        jacobian = np.zeros((3, 3))
        for column, axis in enumerate(axes):
            delta = 1e-5*(axis[-1]-axis[0])
            plus, minus = point.copy(), point.copy()
            plus[column] = min(axis[-1], point[column]+delta)
            minus[column] = max(axis[0], point[column]-delta)
            jacobian[:, column] = ((wrench(plus)[list(target)] - wrench(minus)[list(target)]) /
                                   (plus[column]-minus[column]))
        direction = np.linalg.solve(jacobian, -residual)
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            trial = point + fraction*direction
            if all(axis[0] <= value <= axis[-1] for value, axis in zip(trial, axes)):
                if np.linalg.norm(wrench(trial)[list(target)]) < np.linalg.norm(residual):
                    point, accepted = trial, True
                    break
        if not accepted:
            raise ValueError("Unable to find hydrostatic equilibrium within restoring LUT")
    final = wrench(point)
    if np.max(np.abs(final[list(target)])) > tolerance:
        raise ValueError("Hydrostatic equilibrium residual exceeds tolerance")
    return {"heave_m": float(point[0]), "roll_rad": float(point[1]),
            "pitch_rad": float(point[2]), "residual_wrench_frd": final.tolist(),
            "method": "trilinear_restoring_root_v1"}


def _validate(mass: float, inertia: np.ndarray, added: np.ndarray, linear: np.ndarray,
              stations: list[dict], density: float, curve: dict, hydro: dict) -> dict:
    failures = []
    if not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
        failures.append("rigid inertia is not positive definite")
    if not np.isfinite(added).all() or not np.allclose(added, added.T, atol=1e-8) or np.linalg.eigvalsh(added).min() < -1e-7:
        failures.append("added mass is not positive semidefinite")
    if not np.isfinite(linear).all() or np.linalg.eigvalsh((linear + linear.T) / 2).min() < -1e-7:
        failures.append("linear damping is not dissipative")
    checks = []
    for u in (-2., -1., 0., 1., 2.):
        for v in (-1., 0., 1.):
            for r in (-.5, 0., .5):
                nu = np.array((u, v, 0., 0., 0., r))
                force = crossflow(stations, density, nu)
                force -= linear @ nu
                drag = float(np.interp(u, curve["speed_mps"], curve["force_x_n"]))
                force[0] += drag
                power = float(nu @ force)
                checks.append(power)
                if power > 1e-7 or not np.isfinite(power):
                    failures.append("passive energy generation")
    if hydro["waterplane_area_m2"] <= 0 or hydro["volume_m3"] <= 0:
        failures.append("invalid hydrostatics")
    # Small, deterministic passive free-response screening of the exported
    # tangent model. The integration is only a stability check, not validation.
    base = np.diag([mass, mass, mass, *np.diag(inertia)]) + added
    stiffness = np.asarray(hydro["stiffness_6x6"])
    decays = {}
    for axis, name in enumerate(("surge", "sway", "heave", "roll", "pitch", "yaw")):
        eta = np.zeros(6)
        nu = np.zeros(6)
        nu[axis] = .1
        initial = float(.5 * nu @ base @ nu)
        for _ in range(200):
            force = -linear @ nu - stiffness @ eta + crossflow(stations, density, nu)
            force[0] += float(np.interp(nu[0], curve["speed_mps"], curve["force_x_n"]))
            nu = nu + .01 * np.linalg.solve(base, force)
            eta = eta + .01 * nu
        final = float(.5 * nu @ base @ nu + .5 * eta @ stiffness @ eta)
        decays[name] = {"initial_energy_j": initial, "final_energy_j": final}
        if not np.isfinite(final) or final > initial * 1.01:
            failures.append(f"{name} free response grew")
    return {"passed": not failures, "failures": sorted(set(failures)),
            "maximum_passive_power_w": max(checks), "tested_states": len(checks),
            "free_response": decays}


def generate_simple_vessel(*, geometry: str | Path, output: str | Path, mass_kg: float,
                           cg_frd_m: tuple[float, float, float], units: str | None = None,
                           known_length_m: float | None = None, source_frame: str = "FRD",
                           reference_length_m: float | None = None,
                           water_kinematic_viscosity_m2_s: float = 1.05e-6,
                           draft_m: float | None = None, water_density_kg_m3: float = 1025.,
                           inertia_cg_kg_m2: np.ndarray | None = None,
                           speed_range_mps: tuple[float, float] = (0., 3.),
                           geometry_mode: str = "full_hull",
                           reference_draft_m: float | None = None,
                           classification: str | None = None, disable_bem: bool = False,
                           lut_samples: int = 9, confidence_policy: str = "allow_low",
                           bem_panel_target: int = 900) -> Path:
    if not math.isfinite(mass_kg) or mass_kg <= 0 or len(cg_frd_m) != 3 or not np.isfinite(cg_frd_m).all():
        raise ValueError("Positive mass and finite FRD CG are required")
    if water_density_kg_m3 <= 0 or speed_range_mps[0] < 0 or speed_range_mps[1] <= speed_range_mps[0]:
        raise ValueError("Invalid water density or speed range")
    if not math.isfinite(water_kinematic_viscosity_m2_s) or water_kinematic_viscosity_m2_s <= 0:
        raise ValueError("Water kinematic viscosity must be positive")
    if lut_samples < 3 or lut_samples % 2 != 1:
        raise ValueError("LUT sample count must be odd and at least 3")
    if confidence_policy not in {"allow_low", "strict"}:
        raise ValueError("confidence_policy must be allow_low or strict")
    if geometry_mode not in {"full_hull", "design_waterline_submerged_hull"}:
        raise ValueError("geometry_mode must be full_hull or design_waterline_submerged_hull")
    if geometry_mode == "design_waterline_submerged_hull" and (reference_draft_m is None or reference_draft_m <= 0):
        raise ValueError("design_waterline_submerged_hull requires a positive reference_draft_m")
    if bem_panel_target < 20 or bem_panel_target > 1200:
        raise ValueError("bem_panel_target must be in [20, 1200]")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    module_root = Path(__file__).parent
    code_hash = sha256(b"".join((module_root / name).read_bytes() for name in
        ("simple_pipeline.py", "simple_geometry.py", "simple_sections.py",
         "simple_hydro_mesh.py", "simple_models.py", "simple_crossflow.py", "simple_wave.py",
         "volume_clip.py", "coefficient_package.py")) +
         (module_root.parent / "dynamics" / "crossflow.py").read_bytes()).hexdigest()
    try:
        capytaine_version = version("capytaine")
    except PackageNotFoundError:
        capytaine_version = None
    try:
        simplification_version = version("fast-simplification")
    except PackageNotFoundError:
        simplification_version = None
    request = {"schema": SCHEMA, "code_sha256": code_hash,
        "geometry_sha256": sha256(Path(geometry).read_bytes()).hexdigest(),
        "mass_kg": mass_kg, "cg_frd_m": list(cg_frd_m), "units": units,
        "known_length_m": known_length_m, "reference_length_m": reference_length_m,
        "water_kinematic_viscosity_m2_s": water_kinematic_viscosity_m2_s,
        "source_frame": source_frame,
        "draft_m": draft_m, "water_density_kg_m3": water_density_kg_m3,
        "geometry_mode": geometry_mode, "reference_draft_m": reference_draft_m,
        "inertia_cg_kg_m2": None if inertia_cg_kg_m2 is None else np.asarray(inertia_cg_kg_m2).tolist(),
        "speed_range_mps": list(speed_range_mps), "classification": classification,
        "disable_bem": disable_bem, "lut_samples": lut_samples,
        "bem_panel_target": bem_panel_target,
        "confidence_policy": confidence_policy, "capytaine_version": capytaine_version,
        "fast_simplification_version": simplification_version}
    request_hash = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    required = ("vessel.yaml", "runtime_payload.json", "coefficients.json", "provenance.json",
                "validation.json", "restoring_lut.json")
    if all((root / name).exists() for name in required):
        previous = json.loads((root / "provenance.json").read_text())
        if previous.get("generation_request_sha256") == request_hash:
            if not (root / "normalized_geometry.stl").exists() and (root / "processed_geometry.stl").exists():
                (root / "normalized_geometry.stl").write_bytes((root / "processed_geometry.stl").read_bytes())
            from .coefficient_package import write_coefficient_package
            if not (root / "coefficient_package.yaml").exists():
                write_coefficient_package(root)
            return root
    timings = {}
    begin = perf_counter()
    prepared = prepare_geometry(geometry, units=units, known_length_m=known_length_m,
                                source_frame=source_frame)
    mesh = prepared.mesh
    timings["geometry_s"] = perf_counter() - begin
    begin = perf_counter()
    if geometry_mode == "design_waterline_submerged_hull":
        waterline = float(mesh.bounds[0, 2])
        mesh_draft = float(mesh.bounds[1, 2] - waterline)
        draft_error = abs(mesh_draft - float(reference_draft_m)) / float(reference_draft_m)
        if draft_error > .01:
            raise ValueError(f"Capped geometry depth contradicts reference draft: {draft_error:.2%}")
        hydro = hydrostatic_state(mesh, waterline, density=water_density_kg_m3,
            include_wetted=True, include_waterplane=True, waterline_is_boundary=True)
        error = abs(hydro["displacement_kg"] - mass_kg) / mass_kg
        if error > .05:
            raise ValueError(f"Design-waterline displaced mass contradicts loading: {error:.1%}")
        hydro.update({"mode": "local_design_waterline", "reference_draft_m": float(reference_draft_m),
            "nonlinear_vertical_motion_supported": False,
            "validity": {"heave": "local", "roll": "local", "pitch": "local"},
            "waterline_cap_excluded_from_wetted_area": True})
    elif draft_m is None:
        hydro = solve_waterline(mesh, mass_kg, water_density_kg_m3)
    else:
        level = float(mesh.bounds[1, 2] - draft_m)
        hydro = hydrostatic_state(mesh, level, density=water_density_kg_m3)
        error = abs(hydro["displacement_kg"] - mass_kg) / mass_kg
        if error > .05:
            raise ValueError(f"Supplied draft contradicts mass: displacement error {error:.1%}")
    waterline = hydro["waterline_z_frd_m"]
    draft = float(mesh.bounds[1, 2] - waterline)
    hydro["draft_m"] = draft
    hydro["equilibrium_mass_residual_kg"] = hydro["displacement_kg"] - mass_kg
    volume = hydro["volume_m3"]
    cb = np.asarray(hydro["center_buoyancy_frd_m"])
    gm_roll = cb[2] - cg_frd_m[2] + hydro["waterplane_i_roll_m4"] / volume
    gm_pitch = cb[2] - cg_frd_m[2] + hydro["waterplane_i_pitch_m4"] / volume
    stiffness = np.zeros((6, 6))
    stiffness[2, 2] = water_density_kg_m3 * 9.80665 * hydro["waterplane_area_m2"]
    stiffness[3, 3] = water_density_kg_m3 * 9.80665 * volume * gm_roll
    stiffness[4, 4] = water_density_kg_m3 * 9.80665 * volume * gm_pitch
    hydro["gm_roll_m"], hydro["gm_pitch_m"] = gm_roll, gm_pitch
    hydro["stiffness_6x6"] = stiffness.tolist()
    timings["hydrostatics_s"] = perf_counter() - begin
    if min(gm_roll, gm_pitch) <= 0:
        raise ValueError("Unstable hydrostatic equilibrium for supplied CG")
    begin = perf_counter()
    hydromesh, reduction = reduce_hydrodynamic_mesh(mesh, waterline, hydro,
        target_faces=3000 if disable_bem else bem_panel_target)
    timings["hydrodynamic_mesh_s"] = perf_counter() - begin
    begin = perf_counter()
    components = mesh.split(only_watertight=True)
    largest_component = max(float(component.volume) for component in components)
    hull_components = [component for component in components if component.volume >= .1 * largest_component]
    hull_count = len(hull_components)
    bare_hulls = trimesh.util.concatenate(hull_components)
    regime = classify(mesh, draft, hull_count, speed_range_mps[1], volume, classification)
    try:
        stations = extract_stations(bare_hulls, waterline)
    except ValueError as exc:
        raise ValueError(f"No defensible strip fallback: {exc}") from exc
    strip = strip_added_mass(stations, water_density_kg_m3)
    crossflow_stations, crossflow_provenance = extract_crossflow_stations(
        bare_hulls, waterline, max_froude=regime["supporting_metrics"]["max_froude"],
        density=water_density_kg_m3)
    if not disable_bem:
        os.environ.setdefault("CAPYTAINE_CACHE_DIR", str(root / ".capytaine_cache"))
    bem_key = sha256(json.dumps({"mesh": mesh_hash(hydromesh), "waterline": waterline,
        "density": water_density_kg_m3, "version": capytaine_version,
        "panel_target": bem_panel_target, "code": code_hash}, sort_keys=True).encode()).hexdigest()
    bem_cache = root / "bem_cache.json"
    if len(hull_components) != len(components) and not disable_bem:
        bem = {"status": "rejected", "method": "capytaine", "confidence": "unavailable",
               "reason": "Unlabeled appendage candidates require explicit BEM inclusion decision"}
    elif not disable_bem and bem_cache.exists() and json.loads(bem_cache.read_text()).get("key") == bem_key:
        bem = json.loads(bem_cache.read_text())["result"]
        bem["cache_hit"] = True
    else:
        refined = None
        if not disable_bem and len(mesh.faces) > bem_panel_target:
            refined, _ = reduce_hydrodynamic_mesh(mesh, waterline, hydro,
                target_faces=min(1200, int(bem_panel_target * 1.15)))
            if mesh_hash(refined) == mesh_hash(hydromesh):
                refined = None
        bem = try_bem(hydromesh, strip, enabled=not disable_bem,
                      waterline_frd=waterline, density=water_density_kg_m3,
                      refined_mesh=refined)
        if not disable_bem:
            _write(root, "bem_cache.json", {"key": bem_key, "result": bem})
    added = np.asarray(bem["added_mass_6x6"][0] if bem["status"] == "accepted" else strip["matrix_6x6"])
    added = (added + added.T) / 2
    eigenvalues, vectors = np.linalg.eigh(added)
    added = (vectors * np.maximum(eigenvalues, 0.)) @ vectors.T
    added = (added + added.T) / 2
    added_source = "bem" if bem["status"] == "accepted" else "strip"
    timings["added_mass_s"] = perf_counter() - begin
    inertia = (inertia_estimate(mass_kg, mesh.extents) if inertia_cg_kg_m2 is None
               else np.asarray(inertia_cg_kg_m2, dtype=float))
    if inertia.shape != (3, 3):
        raise ValueError("Inertia tensor must be 3x3")
    linear = np.zeros((6, 6))
    # Sway/yaw lift is speed-dependent and lives in the sectional model.
    # The quadratic separated-flow term has no arbitrary tangent velocity.
    # Conservative critical-damping fractions are explicit low-confidence
    # maneuvering-time estimates; radiation damping is never inserted here.
    effective = np.diag(inertia)
    linear[2, 2] = .04 * math.sqrt(max(stiffness[2, 2], 0.) * (mass_kg + added[2, 2]))
    linear[3, 3] = .10 * math.sqrt(max(stiffness[3, 3], 0.) * (effective[0] + added[3, 3]))
    linear[4, 4] = .04 * math.sqrt(max(stiffness[4, 4], 0.) * (effective[1] + added[4, 4]))
    begin = perf_counter()
    speeds = np.linspace(speed_range_mps[0], speed_range_mps[1], 17)
    speeds = np.unique(np.r_[-speeds[::-1], 0., speeds])
    wave = None
    wave_domain = thin_ship_applicable(
        length_beam=regime["supporting_metrics"]["length_beam"],
        block_coefficient=regime["supporting_metrics"]["block_coefficient"],
        max_froude=regime["supporting_metrics"]["max_froude"], hull_count=hull_count)
    if regime["classification"] == "slender_displacement_monohull" and wave_domain:
        try:
            wave = guarded_wave_curve(bare_hulls, waterline, speeds, water_density_kg_m3)
        except ValueError as exc:
            wave = None
            wave_rejection = str(exc)
    else:
        wave_rejection = "Outside guarded thin/fine displacement domain (L/B>=10, Cb<=0.55, Fn<=0.45, monohull)"
    resistance_length = (float(reference_length_m) if reference_length_m is not None else
                         float(known_length_m) if known_length_m is not None else float(mesh.extents[0]))
    resistance = resistance_curve(speeds, length=resistance_length,
        wetted_area=hydro["wetted_area_m2"], density=water_density_kg_m3,
        viscosity=water_kinematic_viscosity_m2_s,
        classification=regime["classification"], wave=wave)
    resistance["wave_selection"] = "accepted" if wave is not None else "fallback"
    if wave is None:
        resistance["wave_rejection"] = wave_rejection
    timings["resistance_s"] = perf_counter() - begin
    begin = perf_counter()
    if geometry_mode == "design_waterline_submerged_hull":
        exact_center = (0., 0., 0.)
        restoring = _local_restoring_lut(stiffness, draft)
        equilibrium = {"heave_m": 0., "roll_rad": 0., "pitch_rad": 0.,
                       "method": "reference_design_waterline_local_equilibrium"}
    else:
        exact_center = _exact_restoring_equilibrium(mesh, waterline, mass_kg,
            np.asarray(cg_frd_m), water_density_kg_m3, 9.80665)
        restoring = _restoring_lut(mesh, waterline, mass_kg, np.asarray(cg_frd_m),
                                   water_density_kg_m3, 9.80665, lut_samples, center=exact_center)
        equilibrium = _restoring_equilibrium(restoring, mass_kg, float(mesh.extents[0]))
    timings["restoring_lut_s"] = perf_counter() - begin
    validation = _validate(mass_kg, inertia, added, linear, crossflow_stations,
                           water_density_kg_m3, resistance, hydro)
    if not validation["passed"]:
        raise ValueError("Passive validation failed: " + ", ".join(validation["failures"]))
    reduced_mesh_hash = mesh_hash(hydromesh)
    confidence = {"geometry": "medium", "hydrostatics": "medium", "inertia": "low" if inertia_cg_kg_m2 is None else "high",
                  "added_mass": bem["confidence"] if added_source == "bem" else "low", "surge_resistance": "low", "sway_yaw": "low",
                  "roll": "low", "vertical_plane": "low", "overall_passive_model": "low",
                  "reasons": ["BEM is potential flow and requires vessel-specific physics validation" if added_source == "bem" else "Capytaine BEM unavailable or rejected",
                              "Resistance lacks validated viscous-pressure, planing, and multihull-interference terms",
                              "Strip added mass assumes elliptical cross-sections" if added_source != "bem" else "BEM zero-frequency added mass excludes viscous maneuvering effects",
                              "Global self-intersections were not tested"]}
    confidence["geometry"] = "low"  # Global self-intersection check is unavailable.
    if confidence_policy == "strict" and confidence["overall_passive_model"] == "low":
        raise ValueError("Strict confidence policy rejected low-confidence passive model")
    provenance = {"schema": SCHEMA, "model_identifier": MODEL_ID,
                  "original_geometry_sha256": prepared.source_hash,
                  "processed_geometry_sha256": prepared.processed_hash,
                  "hydrodynamic_mesh_sha256": reduced_mesh_hash,
                  "hydrodynamic_mesh_reduction": reduction,
                  "input_mass_kg": mass_kg, "input_cg_frd_m": list(cg_frd_m),
                  "geometry_mode": geometry_mode, "reference_draft_m": reference_draft_m,
                  "reference_length_m": resistance_length,
                  "water_kinematic_viscosity_m2_s": water_kinematic_viscosity_m2_s,
                  "input_inertia_cg_kg_m2": None if inertia_cg_kg_m2 is None else inertia.tolist(),
                  "water_density_kg_m3": water_density_kg_m3, "source_frame": source_frame,
                  "canonical_frame": "FRD", "classification": regime, "bem": bem,
                  "strip": {"method": strip["method"], "stations": len(stations)},
                  "resistance_method": resistance["method"],
                  "restoring_equilibrium": equilibrium,
                  "crossflow_method": "sectional_translation_shear_v5",
                  "crossflow_provenance": crossflow_provenance,
                  "lut": {"samples_per_axis": 3 if geometry_mode == "design_waterline_submerged_hull" else lut_samples,
                          "out_of_domain": "error"},
                  "calibration_hooks": ["surge_resistance_scale", "crossflow_cd_scale", "linear_sway_scale",
                                        "linear_yaw_scale", "added_mass_scale", "roll_damping_scale"],
                  "confidence_policy": confidence_policy,
                  "generation_request_sha256": request_hash,
                  "timings_s": timings}
    cg = np.asarray(cg_frd_m, dtype=float)
    skew = np.array(((0., -cg[2], cg[1]), (cg[2], 0., -cg[0]), (-cg[1], cg[0], 0.)))
    rigid = np.block([[mass_kg * np.eye(3), -mass_kg * skew],
                      [mass_kg * skew, inertia - mass_kg * skew @ skew]])
    parameter_metadata = {"M_A": {"value": added.tolist(), "source": added_source,
        "method": bem["method"] if added_source == "bem" else strip["method"],
        "applicability": "Guarded radiation mesh and strip cross-check" if added_source == "bem" else strip["applicability"],
        "confidence": confidence["added_mass"], "provenance": prepared.processed_hash},
        "M_RB": {"value": rigid.tolist(), "source": "geometry" if inertia_cg_kg_m2 is None else "manual",
                 "method": "rigid_body_mass_matrix", "applicability": "positive mass and positive inertia",
                 "confidence": confidence["inertia"], "provenance": prepared.source_hash},
        "linear_damping_matrix": {"value": linear.tolist(), "source": "analytical",
            "method": "vertical_critical_fraction_only; sway_yaw_lift_in_section_model", "applicability": "vertical critical damping approximation",
            "confidence": "low", "provenance": prepared.processed_hash},
        "surge_resistance": {"value": resistance["force_x_n"], "source": "analytical",
            "method": resistance["method"], "applicability": resistance["limitations"],
            "confidence": "low", "provenance": prepared.processed_hash},
        "crossflow": {"value": {"station_count": len(crossflow_stations)}, "source": "geometry",
            "method": "sectional_translation_shear_v5", "applicability": "sectionable wetted hull",
            "confidence": "low", "provenance": prepared.processed_hash},
        "hydrostatics": {"value": hydro["stiffness_6x6"], "source": "geometry",
            "method": "section_quadrature_v1", "applicability": "closed hull, valid waterline",
            "confidence": "medium", "provenance": prepared.processed_hash}}
    coefficients = {"M_RB": rigid.tolist(), "M_A": added.tolist(),
                    "linear_damping_matrix": linear.tolist(),
                    "maneuvering_radiation_damping": "excluded",
                    "parameter_metadata": parameter_metadata}
    # Runtime package retains the current BCOD schema and adds optional
    # nonlinear payloads consumed by Plant6's model interfaces.
    vessel = {"schema_version": 1, "id": root.name, "version": "1.0.0", "mass_kg": mass_kg,
              "cg_frd_m": list(cg_frd_m), "inertia_cg_kg_m2": inertia.tolist(),
              "added_mass_kg": added.tolist(), "linear_damping": [0.] * 6,
              "quadratic_damping": [0.] * 6,
              "linear_damping_matrix": linear.tolist(),
              "hydrostatics": restoring,
              "crossflow": {"model": "sectional_stations", "stations": crossflow_stations,
                            "water_density_kg_m3": water_density_kg_m3},
              "surge_resistance": resistance,
              "max_abs_nu": [max(5., speed_range_mps[1] * 2)] * 3 + [2.] * 3,
              "min_substep_s": 1e-6, "max_substep_s": .1,
              "collision": {"kind": "box", "half_extents_m": (mesh.extents / 2).tolist()},
              "environment_loads": [0.] * 4,
              "geometry": {"format": Path(geometry).suffix[1:].lower(), "content_hash": prepared.source_hash,
                           "length_m": float(mesh.extents[0]), "beam_m": float(mesh.extents[1]), "draft_m": draft},
              "equilibrium_heave_roll_pitch": [equilibrium["heave_m"], equilibrium["roll_rad"], equilibrium["pitch_rad"]],
              "buoyancy_n": mass_kg * 9.80665,
              "center_buoyancy_frd_m": hydro["center_buoyancy_frd_m"]}
    source_id = prepared.source_hash
    generated_id = prepared.processed_hash
    lineage = {
        "mass_kg": ParameterLineage(source_kind="manual", source_id=source_id),
        "cg_frd_m": ParameterLineage(source_kind="manual", source_id=source_id),
        "inertia_cg_kg_m2": ParameterLineage(source_kind="manual" if inertia_cg_kg_m2 is not None else "empirical", source_id=source_id),
        "added_mass_kg": ParameterLineage(source_kind=added_source, source_id=generated_id),
        "linear_damping": ParameterLineage(source_kind="empirical", source_id=generated_id),
        "quadratic_damping": ParameterLineage(source_kind="geometry-derived", source_id=generated_id),
        "buoyancy_n": ParameterLineage(source_kind="geometry-derived", source_id=generated_id),
    }
    canonical = CanonicalVessel.model_validate({**vessel, "provenance": lineage})
    _write(root, "hydrostatics.json", hydro)
    _write(root, "added_mass.json", {"selected": {"source": added_source, "matrix_6x6": added.tolist()},
                                     "strip": strip, "bem": bem})
    _write(root, "resistance.json", resistance)
    _write(root, "maneuvering.json", {"crossflow": vessel["crossflow"],
             "crossflow_provenance": crossflow_provenance,
             "linear_damping": {"method": "forward_speed_dependent_guarded_lift_in_section_model", "confidence": "low"},
             "roll_damping": {"method": "critical_fraction_placeholder", "fraction": .05, "confidence": "low"},
             "heave_pitch_damping": {"method": "critical_fraction_placeholder", "fraction": .02, "confidence": "low"},
             "calibration_hooks": provenance["calibration_hooks"]})
    _write(root, "restoring_lut.json", restoring)
    _write(root, "coefficients.json", coefficients)
    _write(root, "provenance.json", provenance)
    _write(root, "confidence.json", confidence)
    _write(root, "validation.json", validation)
    _write(root, "geometry.json", prepared.report)
    _write(root, "hydrodynamic_mesh.json", {"sha256": reduced_mesh_hash, **reduction})
    hydromesh.export(root / "hydrodynamic_mesh.stl")
    mesh.export(root / "processed_geometry.stl")
    (root / "normalized_geometry.stl").write_bytes((root / "processed_geometry.stl").read_bytes())
    _write(root, "runtime_payload.json", canonical.simulator_definitions()[0]["payload"])
    (root / "vessel.yaml").write_text(yaml.safe_dump(canonical.model_dump(mode="json"), sort_keys=False))
    lines = [f"# {root.name}: passive vessel generation", "",
             f"Schema: `{SCHEMA}`; frame: FRD; OpenFOAM invoked: no.", "",
             f"Classification: {regime['classification']} ({regime['confidence']}).",
             f"Equilibrium residual: {hydro['equilibrium_mass_residual_kg']:.6g} kg.",
             f"Passive energy checks: {validation['tested_states']} states, passed.",
             f"Total generation time: {sum(timings.values()):.2f} s.", "",
             "## Limitations", "",
             "BEM is optional and may be unavailable/rejected. Wave/planing/interference resistance is unvalidated. Linear sway/yaw and roll/vertical damping are conservative, unvalidated estimates with low confidence.",
             "Hydrodynamic mesh reduction is applied only when topology and hydrostatic tolerances pass; otherwise the full cleaned mesh is retained."]
    (root / "validation_report.md").write_text("\n".join(lines) + "\n")
    from .coefficient_package import write_coefficient_package
    write_coefficient_package(root)
    return root
