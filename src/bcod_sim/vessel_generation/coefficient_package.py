"""Frozen passive coefficient contract and executable package validation.

The numerical reference below is intentionally separate from Plant6/Torch.
It evaluates the serialized terms, so a round-trip catches loader mistakes.
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import RestoringLUT
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.web.runtime_factory import VesselRuntime

SCHEMA = "manta-hydrodynamics-v1"


def _canonical_hash(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def package_from_generated(root: str | Path) -> dict:
    root = Path(root)
    runtime = json.loads((root / "runtime_payload.json").read_text())
    spec = VesselRuntime.model_validate(runtime)
    hydro = json.loads((root / "hydrostatics.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    added = json.loads((root / "added_mass.json").read_text())["selected"]
    coefficients = json.loads((root / "coefficients.json").read_text())
    geometry = json.loads((root / "geometry.json").read_text())
    package = {
        "schema_version": SCHEMA,
        "frames": {"body": "FRD", "force": "FRD", "velocity": "FRD", "moment": "FRD",
                   "reference_point_frd_m": [0., 0., 0.]},
        "force_ownership": {"rigid_coriolis": "Plant6", "added_mass_coriolis": "Plant6",
                            "linear_damping": "runtime_payload.linear_damping_matrix plus speed-dependent Inoue derivatives",
                            "nonlinear_damping": "runtime_payload.surge_resistance",
                            "crossflow": "runtime_payload.crossflow",
                            "restoring": "runtime_payload.hydrostatics"},
        "reference": {"length_m": spec.geometry["length_m"], "beam_m": spec.geometry["beam_m"],
                      "draft_m": hydro["draft_m"],
                      "water_density_kg_m3": spec.crossflow["water_density_kg_m3"],
                      "gravity_mps2": 9.80665},
        "validity": {"max_abs_nu": list(spec.max_abs_nu),
                     "notes": "Low-confidence uncalibrated baseline; internal validation only"},
        "technical_status": {"hydrostatics": "validated_internal",
                             "added_mass": "validated_internal" if added["source"] == "bem" else "estimated",
                             "surge": "estimated", "sway_yaw": "estimated"},
        "hydrostatics": {"mode": hydro.get("mode", "full_hull_nonlinear"),
                           "reference_draft_m": hydro.get("reference_draft_m", hydro.get("draft_m")),
                           "nonlinear_vertical_motion_supported": hydro.get("nonlinear_vertical_motion_supported", True),
                           "validity": hydro.get("validity", {"heave": "nonlinear", "roll": "nonlinear", "pitch": "nonlinear"}),
                           "volume_m3": hydro["volume_m3"],
                           "center_buoyancy_frd_m": hydro["center_buoyancy_frd_m"],
                           "waterplane_area_m2": hydro["waterplane_area_m2"],
                           "waterplane_i_roll_m4": hydro["waterplane_i_roll_m4"],
                           "waterplane_i_pitch_m4": hydro["waterplane_i_pitch_m4"],
                           "restoring": spec.hydrostatics},
        "mass": {"rigid_body_mass_kg": spec.mass_kg, "cg_frd_m": list(spec.cg_frd_m),
                 "inertia_cg_kg_m2": spec.inertia_cg_kg_m2},
        "added_mass": {"matrix_6x6": spec.added_mass_kg, "method": added["source"],
                       "frequency_assumption": "zero_frequency" if added["source"] == "bem" else "not_applicable_strip_estimate",
                       "reference_point_frd_m": [0., 0., 0.]},
        "surge": {"model": "tabulated_sign_aware_resistance", "curve": spec.surge_resistance},
        "maneuvering": {"linear_damping_matrix": spec.linear_damping_matrix,
                        "speed_dependent_linear_damping_matrix_per_mps": spec.speed_dependent_linear_damping_matrix_per_mps,
                        "crossflow": spec.crossflow, "crossflow_includes_forward_speed_lift": False},
        "provenance": {"geometry_hash": spec.geometry["content_hash"],
                       "normalized_geometry_hash": provenance.get("processed_geometry_sha256"),
                       "generation_request_hash": provenance.get("generation_request_sha256"),
                       "source_per_term": coefficients["parameter_metadata"],
                       "warnings": geometry.get("warnings", [])},
        "runtime_payload": runtime,
    }
    input_definition = root / "input_definition.yaml"
    if input_definition.exists():
        normalized = yaml.safe_load(input_definition.read_text())
        package["provenance"]["input_config_sha256"] = normalized["original_config_sha256"]
        package["provenance"]["loading_sources"] = normalized["sources"]
    package["canonical_sha256"] = _canonical_hash(package)
    return package


def write_coefficient_package(root: str | Path) -> Path:
    root = Path(root)
    path = root / "coefficient_package.yaml"
    package = package_from_generated(root)
    path.write_text(yaml.safe_dump(package, sort_keys=True))
    return path


def load_coefficient_package(path: str | Path) -> dict:
    package = yaml.safe_load(Path(path).read_text())
    digest = package.pop("canonical_sha256")
    if package.get("schema_version") != SCHEMA or package.get("frames", {}).get("body") != "FRD":
        raise ValueError("unsupported coefficient schema or frame")
    if _canonical_hash(package) != digest:
        raise ValueError("coefficient package hash mismatch")
    runtime = package["runtime_payload"]
    VesselRuntime.model_validate(runtime)
    if (package["added_mass"]["matrix_6x6"] != runtime["added_mass_kg"] or
        package["maneuvering"]["linear_damping_matrix"] != runtime["linear_damping_matrix"] or
        package["maneuvering"].get("speed_dependent_linear_damping_matrix_per_mps") != runtime.get("speed_dependent_linear_damping_matrix_per_mps") or
        package["maneuvering"]["crossflow"] != runtime["crossflow"] or
        package["surge"]["curve"] != runtime["surge_resistance"] or
        package["hydrostatics"]["restoring"] != runtime["hydrostatics"]):
        raise ValueError("coefficient summary and runtime payload disagree")
    package["canonical_sha256"] = digest
    if package["runtime_payload"].get("maneuvering_surface") is not None:
        raise ValueError("baseline coefficient package cannot own a captive force surface")
    if package["runtime_payload"].get("added_mass_coriolis_enabled", True) is not True:
        raise ValueError("baseline requires analytic added-mass Coriolis ownership")
    return package


def _reference_damping(package: dict, nu: np.ndarray) -> np.ndarray:
    p = package["runtime_payload"]
    matrix = np.asarray(p["linear_damping_matrix"], float)
    speed_matrix=p.get("speed_dependent_linear_damping_matrix_per_mps")
    if speed_matrix is not None:
        matrix=matrix+abs(float(nu[0]))*np.asarray(speed_matrix,float)
    result = -matrix @ nu
    speeds = np.asarray(p["surge_resistance"]["speed_mps"], float)
    forces = np.asarray(p["surge_resistance"]["force_x_n"], float)
    u = nu[0]
    if u < speeds[0] and speeds[0] < 0:
        result[0] += forces[0] * abs(u / speeds[0]) ** 2
    elif u > speeds[-1] and speeds[-1] > 0:
        result[0] += forces[-1] * abs(u / speeds[-1]) ** 2
    else:
        result[0] += np.interp(u, speeds, forces)
    return result


def _reference_crossflow(package: dict, nu: np.ndarray) -> np.ndarray:
    item = package["runtime_payload"]["crossflow"]
    stations = item["stations"]
    x = np.array([s["x_m"] for s in stations]); y = np.array([s["y_m"] for s in stations])
    depth = np.array([s["draft_m"] for s in stations]); dx = np.array([s["dx_m"] for s in stations])
    cd = np.array([s["cd"] for s in stations]); lift_base = np.array([s["lift_base_kg_per_m"] for s in stations])
    local = nu[1] + x * nu[5]
    local_u = nu[0] - y * nu[5]
    projected = depth * dx
    mean_local = np.sum(projected * local) / projected.sum()
    mean_u = np.sum(projected * np.abs(local_u)) / projected.sum()
    speed2 = mean_u**2 + mean_local**2
    weight = mean_local**2 / speed2 if speed2 > 0 else 0.
    shear = local - mean_local
    drag = -.5 * item["water_density_kg_m3"] * cd * depth * local * (weight * np.abs(local) + (1-weight) * np.abs(shear)) * dx
    lift = -(1-weight) * lift_base * np.abs(local_u) * local
    fy = drag + lift
    result = np.zeros(6)
    result[1] = fy.sum(); result[5] = np.dot(x, fy)
    if all("axial_rotation_area_m2" in s and "axial_rotation_cd" in s for s in stations):
        area = np.array([s["axial_rotation_area_m2"] for s in stations])
        axial_cd = np.array([s["axial_rotation_cd"] for s in stations])
        rotational_u = -y * nu[5]
        axial = -.5 * item["water_density_kg_m3"] * axial_cd * area * np.abs(rotational_u) * rotational_u
        result[5] -= np.dot(y, axial)
    return result


def reference_wrench(package: dict, nu: np.ndarray) -> np.ndarray:
    """Independent NumPy evaluation of serialized dissipative terms."""
    nu = np.asarray(nu, float)
    if nu.shape != (6,) or not np.isfinite(nu).all():
        raise ValueError("expected finite body-FRD six-velocity")
    return _reference_damping(package, nu) + _reference_crossflow(package, nu)


def _runtime_plant(package: dict) -> Plant6:
    p = VesselRuntime.model_validate(package["runtime_payload"])
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    h = p.hydrostatics
    return Plant6(
        MassProperties(p.mass_kg, tensor(p.cg_frd_m), tensor(p.inertia_cg_kg_m2), tensor(p.added_mass_kg)),
        Damping(tensor(p.linear_damping), tensor(p.quadratic_damping), tensor(p.linear_damping_matrix),
                surge_resistance_curve=(tensor(p.surge_resistance["speed_mps"]), tensor(p.surge_resistance["force_x_n"])),
                speed_dependent_linear_matrix_per_mps=(tensor(p.speed_dependent_linear_damping_matrix_per_mps)
                    if p.speed_dependent_linear_damping_matrix_per_mps is not None else None)),
        RestoringLUT(tensor(h["axes"]["heave_m"]), tensor(h["axes"]["roll_rad"]),
                     tensor(h["axes"]["pitch_rad"]), tensor(h["wrench_frd"])),
        OperatingEnvelope(tensor(p.max_abs_nu), p.min_substep_s, p.max_substep_s),
        crossflow=SectionalCrossflow.from_stations(p.crossflow["stations"],
                     density=p.crossflow["water_density_kg_m3"], dtype=torch.float64))


def validate_coefficients(path: str | Path, *, grid_size: int = 7) -> dict:
    if grid_size < 3 or grid_size % 2 != 1:
        raise ValueError("grid_size must be odd and at least 3")
    package = load_coefficient_package(path)
    plant = _runtime_plant(package)
    p = package["runtime_payload"]
    zeros = {name: torch.zeros(6, dtype=torch.float64) for name in EXTERNAL_TERMS}
    samples = []
    maximum = 0.
    vmax = min(1., p["max_abs_nu"][1] / 3)
    rmax = min(.3, p["max_abs_nu"][5] / 3)
    for u in np.linspace(-.5, 1., grid_size):
        for v in np.linspace(-vmax, vmax, grid_size):
            for r in np.linspace(-rmax, rmax, grid_size):
                nu = np.array([u, v, 0., 0., 0., r])
                reference = reference_wrench(package, nu)
                state = VesselState(torch.zeros(3, dtype=torch.float64),
                                    torch.tensor([1., 0., 0., 0.], dtype=torch.float64),
                                    torch.as_tensor(nu, dtype=torch.float64))
                ledger = plant.diagnostics(state, zeros)
                actual = sum((ledger.terms[k].numpy() for k in ("linear_damping", "nonlinear_damping", "crossflow")))
                error = float(np.max(np.abs(reference - actual)))
                maximum = max(maximum, error)
                power = float(nu @ reference)
                samples.append((u, v, r, reference[0], reference[1], reference[5], error, power))
                if power > 1e-7:
                    raise ValueError("coefficient damping injects energy")
    result = {"schema": SCHEMA, "package_sha256": package["canonical_sha256"],
              "state_count": len(samples), "max_runtime_error": maximum,
              "passed": maximum < 1e-7, "physical_accuracy": "not externally validated"}
    root = Path(path).parent
    diagnostics = root / "diagnostics"; diagnostics.mkdir(exist_ok=True)
    for name, value in (("geometry.json", json.loads((root / "geometry.json").read_text())),
                        ("hydrostatics.json", package["hydrostatics"]),
                        ("added_mass.json", package["added_mass"]),
                        ("sections.json", package["maneuvering"]["crossflow"]["stations"]),
                        ("warnings.json", package["provenance"]["warnings"])):
        (diagnostics / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with (diagnostics / "state_grid.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("u_mps", "v_mps", "r_rad_s", "X_n", "Y_n", "N_nm", "runtime_error", "damping_power_w"))
        writer.writerows(samples)
    (diagnostics / "coefficient_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (diagnostics / "coefficient_validation.md").write_text(
        f"# Coefficient validation\n\nState count: {len(samples)}\n\nMaximum runtime discrepancy: {maximum:.4g}\n\n"
        f"Internal round-trip: {'PASS' if result['passed'] else 'FAIL'}\n\nPhysical accuracy: not externally validated.\n")
    _write_plots(root, package, vmax, rmax)
    report = root / "report"
    (report / "generation_report.md").write_text(
        f"# CAD to coefficient validation\n\nPackage SHA-256: `{package['canonical_sha256']}`\n\n"
        f"Geometry SHA-256: `{package['provenance']['geometry_hash']}`\n\n"
        f"Frame and moment origin: FRD, {package['frames']['reference_point_frd_m']} m.\n\n"
        f"Added mass: {package['added_mass']['method']} ({package['added_mass']['frequency_assumption']}).\n\n"
        f"State-grid runtime discrepancy: {maximum:.4g} N or N m across {len(samples)} states.\n\n"
        f"Internal coefficient round-trip: {'PASS' if result['passed'] else 'FAIL'}.\n\n"
        "Physical coefficient accuracy: NOT YET VALIDATED against independent experiments.\n")
    return result


def _write_plots(root: Path, package: dict, vmax: float, rmax: float) -> None:
    """Produce stable, inspectable visual diagnostics without affecting the package hash."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(root / "diagnostics" / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "report" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    u = np.linspace(-1., 1., 101)
    curves = np.array([reference_wrench(package, [speed, 0, 0, 0, 0, 0]) for speed in u])
    fig, ax = plt.subplots(); ax.plot(u, curves[:, 0]); ax.axhline(0, color="gray", lw=.5)
    ax.set(xlabel="u (m/s)", ylabel="X (N)", title="Surge resistance")
    fig.savefig(plots / "surge.png", dpi=120); plt.close(fig)
    for kind, values, ix in (("sway", np.linspace(-vmax, vmax, 101), 1),
                             ("yaw", np.linspace(-rmax, rmax, 101), 5)):
        series = np.array([reference_wrench(package, [0.5, val if ix == 1 else 0., 0, 0, 0,
                                                       val if ix == 5 else 0.]) for val in values])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for ax, col, label in zip(axes, (1, 5), ("Y (N)", "N (N m)")):
            ax.plot(values, series[:, col]); ax.set(xlabel=f"{kind} velocity", ylabel=label)
        fig.tight_layout(); fig.savefig(plots / f"pure_{kind}.png", dpi=120); plt.close(fig)
    vs = np.linspace(-vmax, vmax, 41); rs = np.linspace(-rmax, rmax, 41)
    field = np.array([[reference_wrench(package, [0.5, v, 0, 0, 0, r]) for r in rs] for v in vs])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, col, label in zip(axes, (1, 5), ("Y (N)", "N (N m)")):
        panel = ax.imshow(field[:, :, col], origin="lower", aspect="auto",
                          extent=(rs[0], rs[-1], vs[0], vs[-1]))
        ax.set(xlabel="r (rad/s)", ylabel="v (m/s)", title=label)
        fig.colorbar(panel, ax=ax)
    fig.tight_layout(); fig.savefig(plots / "combined_sway_yaw.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots()
    matrix = np.asarray(package["added_mass"]["matrix_6x6"])
    panel = ax.imshow(matrix, cmap="coolwarm")
    fig.colorbar(panel, ax=ax); ax.set(title="Added-mass matrix", xlabel="velocity axis", ylabel="wrench axis")
    fig.savefig(plots / "added_mass.png", dpi=120); plt.close(fig)
    geometry_path = root / "processed_geometry.stl"
    if geometry_path.exists():
        import trimesh
        mesh = trimesh.load_mesh(geometry_path)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        vertices = np.asarray(mesh.vertices)
        axes[0].scatter(vertices[:, 0], vertices[:, 1], s=1)
        axes[0].set(xlabel="x FRD (m)", ylabel="y FRD (m)", title="Plan view")
        axes[1].scatter(vertices[:, 0], vertices[:, 2], s=1)
        waterline = vertices[:, 2].max() - package["reference"]["draft_m"]
        axes[1].axhline(waterline, color="blue", lw=1, label="waterline")
        axes[1].scatter([package["mass"]["cg_frd_m"][0]], [package["mass"]["cg_frd_m"][2]], c="red", label="CG")
        axes[1].scatter([package["hydrostatics"]["center_buoyancy_frd_m"][0]],
                        [package["hydrostatics"]["center_buoyancy_frd_m"][2]], c="green", label="CB")
        axes[1].legend(fontsize=7); axes[1].set(xlabel="x FRD (m)", ylabel="z FRD (m)", title="Profile")
        fig.tight_layout(); fig.savefig(plots / "geometry.png", dpi=120); plt.close(fig)
