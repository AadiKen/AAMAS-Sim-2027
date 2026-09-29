"""Fast geometry-to-runtime checks; no CFD or network dependencies."""
import json
import numpy as np
import pytest
import torch
import trimesh

from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel
from bcod_sim.vessel_generation.simple_models import (crossflow, validate_bem_matrix,
    bem_frame_matrix, panel_resolution_change, hmri_total_sway_prime, try_bem)
from bcod_sim.vessel_generation.simple_geometry import prepare_geometry
from bcod_sim.vessel_generation.simple_geometry import _remove_degenerate_faces_preserving_closure


def test_degenerate_facet_is_retained_when_it_closes_surface():
    # A collapsed stern facet can have zero area while its indexed edges still
    # close the source volume. Blind removal leaves the volume open.
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 0., 1.]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight
    cleaned, removed, retained = _remove_degenerate_faces_preserving_closure(mesh)
    assert cleaned.is_watertight
    assert removed == 0
    assert retained
from bcod_sim.vessel_generation.simple_hydro_mesh import reduce_hydrodynamic_mesh
from bcod_sim.vessel_generation.simple_sections import hydrostatic_state
from bcod_sim.vessel_generation.simple_calibration import apply_passive_scales
from bcod_sim.web.runtime_factory import VesselRuntime
from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.dynamics.restoring import RestoringLUT
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.state.vessel_state import VesselState


def _make(mesh, tmp_path, name, mass, **kwargs):
    path = tmp_path / f"{name}.stl"
    mesh.export(path)
    root = tmp_path / f"{name}_package"
    generate_simple_vessel(geometry=path, output=root, units="m", mass_kg=mass,
                           cg_frd_m=(0., 0., 0.), disable_bem=True,
                           lut_samples=3, **kwargs)
    return root


def test_box_complete_package_and_runtime(tmp_path):
    root = _make(trimesh.creation.box(extents=(4., 2., 1.)), tmp_path, "box", 2050.)
    hydro = json.loads((root / "hydrostatics.json").read_text())
    assert abs(hydro["volume_m3"] - 2.) < .01
    assert abs(hydro["draft_m"] - .25) < .01
    assert abs(hydro["equilibrium_mass_residual_kg"]) < .01
    lut = json.loads((root / "restoring_lut.json").read_text())
    loads = np.asarray(lut["wrench_frd"])
    center = len(loads) // 2
    assert abs(loads[center, center, center, 2]) < 1.
    assert loads[center-1, center, center, 2] > 0 > loads[center+1, center, center, 2]
    assert loads[center, center-1, center, 3] > 0 > loads[center, center+1, center, 3]
    validation = json.loads((root / "validation.json").read_text())
    assert validation["passed"] and len(validation["free_response"]) == 6
    spec = VesselRuntime.model_validate_json((root / "runtime_payload.json").read_text())
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    h = spec.hydrostatics
    hydro_runtime = RestoringLUT(t(h["axes"]["heave_m"]), t(h["axes"]["roll_rad"]),
                                 t(h["axes"]["pitch_rad"]), t(h["wrench_frd"]))
    cf = SectionalCrossflow.from_stations(spec.crossflow["stations"], dtype=torch.float64)
    curve = spec.surge_resistance
    damping = Damping(t(spec.linear_damping), t(spec.quadratic_damping),
        t(spec.linear_damping_matrix), surge_resistance_curve=(t(curve["speed_mps"]), t(curve["force_x_n"])))
    plant = Plant6(MassProperties(spec.mass_kg, t(spec.cg_frd_m), t(spec.inertia_cg_kg_m2),
        t(spec.added_mass_kg)), damping, hydro_runtime,
        OperatingEnvelope(t(spec.max_abs_nu), spec.min_substep_s, spec.max_substep_s), crossflow=cf)
    zero = {name: t([0.] * 6) for name in EXTERNAL_TERMS}
    state = VesselState(t([0., 0., 0.]), t([1., 0., 0., 0.]), t([0., .2, 0., 0., 0., .1]))
    assert torch.isfinite(plant.acceleration(state, zero)).all().item()
    step = plant.step(state, zero, .01)
    assert torch.isfinite(step.state.nu_body).all().item()


def test_exported_equilibrium_accounts_for_longitudinal_cg_offset(tmp_path):
    mesh = trimesh.creation.box(extents=(4., 2., 1.))
    path = tmp_path / "offset_box.stl"
    mesh.export(path)
    root = tmp_path / "offset_box_package"
    generate_simple_vessel(geometry=path, output=root, units="m", mass_kg=2050.,
        cg_frd_m=(.05, 0., 0.), disable_bem=True, lut_samples=3)
    spec = VesselRuntime.model_validate_json((root / "runtime_payload.json").read_text())
    heave, roll, pitch = spec.equilibrium_heave_roll_pitch
    assert abs(pitch) > 1e-4
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    q = t([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])
    state = VesselState(t([0., 0., heave]), q, t([0.]*6))
    h = spec.hydrostatics
    model = RestoringLUT(t(h["axes"]["heave_m"]), t(h["axes"]["roll_rad"]),
                         t(h["axes"]["pitch_rad"]), t(h["wrench_frd"]))
    load = model.evaluate(state, spec.mass_kg, t(spec.cg_frd_m)).tau_body
    assert torch.max(torch.abs(load[[2, 3, 4]])).item() < .01


def test_catamaran_zero_surge_and_fallback(tmp_path):
    left = trimesh.creation.box(extents=(4., .6, 1.))
    right = left.copy()
    left.apply_translation((0., -1., 0.))
    right.apply_translation((0., 1., 0.))
    root = _make(trimesh.util.concatenate((left, right)), tmp_path, "cat", 1230., speed_range_mps=(0., 1.))
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["classification"]["classification"] == "displacement_catamaran"
    assert provenance["bem"]["status"] == "disabled"
    assert len(json.loads((root / "maneuvering.json").read_text())["crossflow"]["stations"]) > 40
    stations = json.loads((root / "maneuvering.json").read_text())["crossflow"]["stations"]
    for nu in ((0., .5, 0., 0., 0., 0.), (0., 0., 0., 0., 0., .3),
               (0., .3, 0., 0., 0., -.2), (-1., .3, 0., 0., 0., .2)):
        state = np.asarray(nu)
        force = crossflow(stations, 1025., state)
        assert np.isfinite(force).all() and state @ force <= 1e-9


def test_scale_and_explicit_failure(tmp_path):
    mesh = trimesh.creation.box(extents=(4000., 2000., 1000.))
    path = tmp_path / "millimetres.obj"
    mesh.export(path)
    prepared = prepare_geometry(path, units="mm")
    assert np.allclose(prepared.mesh.extents, (4., 2., 1.))
    known = prepare_geometry(path, known_length_m=4.)
    assert np.allclose(known.mesh.extents, (4., 2., 1.))
    flipped = prepare_geometry(path, units="mm", source_frame="FPU")
    assert flipped.report["canonical_frame"] == "FRD"
    assert np.allclose(flipped.mesh.extents, (4., 2., 1.))
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    broken = tmp_path / "open.stl"
    mesh.export(broken)
    with pytest.raises(ValueError, match="closed orientable"):
        prepare_geometry(broken, units="mm")


def test_planing_candidate_downgrades_confidence(tmp_path):
    root = _make(trimesh.creation.box(extents=(4., 2., 1.)), tmp_path, "fast", 2050., speed_range_mps=(0., 7.))
    provenance = json.loads((root / "provenance.json").read_text())
    confidence = json.loads((root / "confidence.json").read_text())
    assert provenance["classification"]["classification"] == "planing_candidate"
    assert confidence["surge_resistance"] == "low"


def test_repeat_generation_reuses_content_addressed_result(tmp_path):
    mesh = trimesh.creation.box(extents=(4., 2., 1.))
    path = tmp_path / "repeat.stl"
    mesh.export(path)
    output = tmp_path / "package"
    args = dict(geometry=path, output=output, units="m", mass_kg=2050.,
                cg_frd_m=(0., 0., 0.), disable_bem=True, lut_samples=3)
    generate_simple_vessel(**args)
    before = (output / "coefficients.json").read_bytes()
    provenance_before = (output / "provenance.json").read_bytes()
    generate_simple_vessel(**args)
    assert (output / "coefficients.json").read_bytes() == before
    assert (output / "provenance.json").read_bytes() == provenance_before
    with pytest.raises(ValueError, match="Strict confidence policy"):
        generate_simple_vessel(**args, confidence_policy="strict")


def test_bem_validation_rejects_bad_results_and_accepts_consistent_matrix():
    baseline = np.diag([0., 10., 20., 5., 10., 8.])
    strip = {"matrix_6x6": baseline.tolist()}
    matrix, comparison = validate_bem_matrix(baseline, strip)
    assert np.allclose(matrix, baseline)
    assert comparison["Sway"] == 1.
    with pytest.raises(ValueError, match="Non-finite"):
        bad = baseline.copy(); bad[0, 0] = np.nan
        validate_bem_matrix(bad, strip)
    with pytest.raises(ValueError, match="asymmetry"):
        bad = baseline.copy(); bad[1, 2] = 10.
        validate_bem_matrix(bad, strip)
    with pytest.raises(ValueError, match="negative eigenvalue"):
        bad = baseline.copy(); bad[3, 3] = -100.
        validate_bem_matrix(bad, strip)
    with pytest.raises(ValueError, match="disagreement"):
        bad = baseline.copy(); bad[1, 1] = 100.
        validate_bem_matrix(bad, strip)


def test_bem_frame_resolution_and_hmri_total_mass():
    a = np.eye(6)
    a[0, 1] = a[1, 0] = 2.
    mapped = bem_frame_matrix(a)
    assert mapped[0, 1] == mapped[1, 0] == -2.
    assert np.allclose(bem_frame_matrix(mapped), a)
    fine = np.diag([1., 100., 200., 10., 20., 30.])
    coarse = fine.copy(); coarse[1, 1] *= .95
    assert panel_resolution_change(coarse, fine)["stable"]
    coarse[1, 1] *= .5
    assert not panel_resolution_change(coarse, fine)["stable"]
    denominator = .5 * 1025. * 5.75**2 * .27
    assert hmri_total_sway_prime(833.38, 739.9, 1025., 5.75, .27) == pytest.approx((833.38 + 739.9) / denominator)
    assert hmri_total_sway_prime(833.38, 0., 1025., 5.75, .27) == pytest.approx(833.38 / denominator)


def test_bem_disabled_returns_without_adapter_invocation():
    assert try_bem(trimesh.creation.box(), None, enabled=False,
                   waterline_frd=0., density=1025.)["status"] == "disabled"


def test_surface_piercing_lid_and_strip_comparison():
    pytest.importorskip("capytaine")
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale((3., .6, .5))
    from bcod_sim.vessel_generation.simple_models import extract_stations, strip_added_mass
    strip = strip_added_mass(extract_stations(mesh, 0.), 1025.)
    result = try_bem(mesh, strip, enabled=True, waterline_frd=0., density=1025.)
    assert result["status"] == "accepted"
    assert result["panel_quality"]["lid_panels"] > 0
    assert result["panel_quality"]["lid_forces_excluded"]
    assert 1.5 < result["strip_agreement_ratios"]["Sway"] < 2.5
    assert result["confidence"] == "medium"


def test_bem_cache_reuses_and_invalidates(tmp_path, monkeypatch):
    import bcod_sim.vessel_generation.simple_pipeline as module
    calls = []
    def rejected(*args, **kwargs):
        calls.append(kwargs["waterline_frd"])
        return {"status": "rejected", "method": "capytaine", "reason": "quality gate"}
    monkeypatch.setattr(module, "try_bem", rejected)
    path = tmp_path / "hull.stl"
    trimesh.creation.box(extents=(4., 2., 1.)).export(path)
    args = dict(geometry=path, output=tmp_path / "package", units="m", mass_kg=2050.,
                cg_frd_m=(0., 0., 0.), lut_samples=3)
    generate_simple_vessel(**args)
    assert len(calls) == 1
    generate_simple_vessel(**args)
    assert len(calls) == 1
    generate_simple_vessel(**(args | {"speed_range_mps": (0., 2.)}))
    assert len(calls) == 1
    generate_simple_vessel(**(args | {"bem_panel_target": 800}))
    assert len(calls) == 2
    generate_simple_vessel(**(args | {"mass_kg": 2100., "bem_panel_target": 800}))
    assert len(calls) == 3 and calls[0] != calls[-1]
    trimesh.creation.box(extents=(4.1, 2., 1.)).export(path)
    generate_simple_vessel(**(args | {"bem_panel_target": 800}))
    assert len(calls) == 4


def test_forced_bem_failure_falls_back_to_strip(tmp_path, monkeypatch):
    import bcod_sim.vessel_generation.simple_pipeline as module
    monkeypatch.setattr(module, "try_bem", lambda *args, **kwargs:
                        {"status": "rejected", "method": "capytaine", "reason": "forced failure"})
    path = tmp_path / "hull.stl"
    trimesh.creation.box(extents=(4., 2., 1.)).export(path)
    root = tmp_path / "fallback"
    generate_simple_vessel(geometry=path, output=root, units="m", mass_kg=2050.,
                           cg_frd_m=(0., 0., 0.), lut_samples=3)
    added = json.loads((root / "added_mass.json").read_text())
    assert added["selected"]["source"] == "strip"
    assert added["bem"]["reason"] == "forced failure"


def test_convex_hydrodynamic_mesh_reduction_preserves_hydrostatics():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    mesh.apply_scale((3., .6, .5))
    reference = hydrostatic_state(mesh, 0.)
    reduced, report = reduce_hydrodynamic_mesh(mesh, 0., reference, target_faces=3000)
    assert report["status"] == "reduced"
    assert len(reduced.faces) <= 3000
    assert all(report["errors"][key] <= limit for key, limit in report["tolerances"].items())


def test_stage3b_scales_preserve_passivity(tmp_path):
    root = _make(trimesh.creation.box(extents=(4., 2., 1.)), tmp_path, "scales", 2050.)
    original = json.loads((root / "runtime_payload.json").read_text())
    scaled = apply_passive_scales(original, {"surge_resistance_scale": 1.2,
        "crossflow_cd_scale": .8, "linear_sway_scale": 1.1,
        "linear_yaw_scale": .9, "added_mass_scale": 1.05,
        "roll_damping_scale": 1.3})
    assert VesselRuntime.model_validate(scaled)
    assert scaled["crossflow"]["cd_scale"] == .8
    assert np.linalg.eigvalsh(scaled["linear_damping_matrix"]).min() >= -1e-8
    assert np.linalg.eigvalsh(scaled["added_mass_kg"]).min() >= -1e-8
    assert original["crossflow"].get("cd_scale") is None
    with pytest.raises(ValueError, match="positive"):
        apply_passive_scales(original, {"added_mass_scale": -1.})


def test_default_generate_cli_does_not_construct_openfoam(tmp_path, monkeypatch):
    from bcod_sim import cli
    monkeypatch.setattr(cli, "OpenFOAMAdapter", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("CFD path invoked")))
    path = tmp_path / "cli_hull.stl"
    trimesh.creation.box(extents=(4., 2., 1.)).export(path)
    output = tmp_path / "cli_package"
    assert cli.main(["vessel", "generate", "--geometry", str(path), "--units", "m",
                     "--mass", "2050", "--cg", "0", "0", "0", "--output", str(output),
                     "--lut-samples", "3", "--disable-bem"]) == 0
    assert (output / "runtime_payload.json").exists()
