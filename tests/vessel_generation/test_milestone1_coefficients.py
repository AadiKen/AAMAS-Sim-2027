"""Manufactured CAD fixtures through the production generator and Plant6 loader."""
import json

import numpy as np
import pytest
import trimesh
import yaml
import torch

from bcod_sim.vessel_generation.coefficient_package import (
    _canonical_hash, load_coefficient_package, reference_wrench, validate_coefficients,
)
from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel
from bcod_sim.vessel_generation.simple_geometry import prepare_geometry
from bcod_sim.vessel_generation.input_contract import GenerationFailure, generate_from_config
from bcod_sim.vessel_generation.simple_models import strip_added_mass
from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.state.vessel_state import VesselState


def _generate(tmp_path, name, mesh, mass):
    source = tmp_path / f"{name}.stl"
    mesh.export(source)
    return generate_simple_vessel(geometry=source, output=tmp_path / name,
        mass_kg=mass, cg_frd_m=(0., 0., 0.), units="m", disable_bem=True,
        lut_samples=3)


def test_box_analytic_roundtrip_and_determinism(tmp_path):
    root = _generate(tmp_path, "barge", trimesh.creation.box(extents=(4, 2, 1)), 2050.)
    hydro = json.loads((root / "hydrostatics.json").read_text())
    assert hydro["volume_m3"] == pytest.approx(2., rel=.005)
    assert hydro["waterplane_area_m2"] == pytest.approx(8., rel=.01)
    assert hydro["center_buoyancy_frd_m"] == pytest.approx([0., 0., .375], abs=.01)
    first = (root / "coefficient_package.yaml").read_bytes()
    generate_simple_vessel(geometry=tmp_path / "barge.stl", output=root,
        mass_kg=2050., cg_frd_m=(0., 0., 0.), units="m", disable_bem=True,
        lut_samples=3)
    assert (root / "coefficient_package.yaml").read_bytes() == first
    result = validate_coefficients(root / "coefficient_package.yaml")
    assert result["passed"] and result["state_count"] == 343
    assert result["max_runtime_error"] < 1e-9
    assert (root / "diagnostics/state_grid.csv").exists()
    assert (root / "report/plots/geometry.png").exists()
    assert (root / "report/plots/combined_sway_yaw.png").exists()


def test_catamaran_symmetry_and_sections(tmp_path):
    port = trimesh.creation.box(extents=(4, .6, 1))
    starboard = port.copy()
    port.apply_translation((0, -1, 0))
    starboard.apply_translation((0, 1, 0))
    root = _generate(tmp_path, "catamaran", trimesh.util.concatenate((port, starboard)), 1230.)
    package = load_coefficient_package(root / "coefficient_package.yaml")
    stations = package["maneuvering"]["crossflow"]["stations"]
    assert len({s["hull_id"] for s in stations}) == 2
    assert min(s["y_m"] for s in stations) < 0 < max(s["y_m"] for s in stations)
    for u, v, r in ((0., .2, .1), (.5, -.3, .2), (1., .1, -.2)):
        a = reference_wrench(package, np.array([u, v, 0, 0, 0, r]))
        b = reference_wrench(package, np.array([u, -v, 0, 0, 0, -r]))
        assert a[[1, 5]] == pytest.approx(-b[[1, 5]], abs=1e-8)
    assert validate_coefficients(root / "coefficient_package.yaml", grid_size=3)["passed"]


def test_low_speed_continuity_and_tamper_detection(tmp_path):
    root = _generate(tmp_path, "low_speed", trimesh.creation.box(extents=(4, 2, 1)), 2050.)
    path = root / "coefficient_package.yaml"
    package = load_coefficient_package(path)
    values = [reference_wrench(package, np.array([u, .1, 0, 0, 0, .03]))
              for u in (-1e-8, 0., 1e-8)]
    assert all(np.isfinite(x).all() for x in values)
    assert np.max(np.abs(values[0] - values[2])) < 1e-4
    text = path.read_text().replace("manta-hydrodynamics-v1", "manta-hydrodynamics-v2")
    path.write_text(text)
    with pytest.raises(ValueError):
        load_coefficient_package(path)


def test_package_rejects_duplicate_surface_owner(tmp_path):
    root = _generate(tmp_path, "ownership", trimesh.creation.box(extents=(4, 2, 1)), 2050.)
    path = root / "coefficient_package.yaml"
    package = load_coefficient_package(path)
    package.pop("canonical_sha256")
    package["runtime_payload"]["maneuvering_surface"] = {"pretend": "complete captive surface"}
    package["canonical_sha256"] = _canonical_hash(package)
    path.write_text(yaml.safe_dump(package))
    with pytest.raises(ValueError, match="cannot own a captive force surface"):
        load_coefficient_package(path)


def test_open_mesh_rejected(tmp_path):
    mesh = trimesh.creation.box(extents=(4, 2, 1))
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    source = tmp_path / "open.stl"
    mesh.export(source)
    with pytest.raises(ValueError, match="closed orientable"):
        generate_simple_vessel(geometry=source, output=tmp_path / "bad", mass_kg=2050.,
            cg_frd_m=(0., 0., 0.), units="m", disable_bem=True, lut_samples=3)


def test_thousandfold_scale_error_rejected(tmp_path):
    trimesh.creation.box(extents=(4000, 2000, 1000)).export(tmp_path / "wrong_units.stl")
    with pytest.raises(ValueError, match="INVALID_SCALE"):
        generate_simple_vessel(geometry=tmp_path / "wrong_units.stl", output=tmp_path / "bad_scale",
            mass_kg=2050., cg_frd_m=(0., 0., 0.), units="m", disable_bem=True, lut_samples=3)


def test_axis_swap_is_flagged(tmp_path):
    trimesh.creation.box(extents=(2., 4., 1.)).export(tmp_path / "swapped.stl")
    report = prepare_geometry(tmp_path / "swapped.stl", units="m").report
    assert any("source axes" in warning for warning in report["warnings"])


def test_draft_driven_config_provenance(tmp_path):
    trimesh.creation.box(extents=(4, 2, 1)).export(tmp_path / "barge.stl")
    config = {"geometry": {"file": "barge.stl", "units": "m", "source_frame": "FRD"},
              "loading": {"draft_m": .25},
              "operating_envelope": {"max_forward_speed_mps": 1., "reverse_required": False,
                                     "max_expected_sway_mps": .5, "max_expected_yaw_rate_rad_s": .2},
              "generation": {"disable_bem": True, "lut_samples": 3}}
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(config))
    root = generate_from_config(path, tmp_path / "from_config")
    info = yaml.safe_load((root / "input_definition.yaml").read_text())
    assert info["mass_kg"] == pytest.approx(2050., rel=.001)
    assert info["sources"]["mass"] == "geometry_derived"
    assert info["sources"]["cg"] == "estimated"
    assert load_coefficient_package(root / "coefficient_package.yaml")["provenance"]["loading_sources"]["mass"] == "geometry_derived"
    assert validate_coefficients(root / "coefficient_package.yaml", grid_size=3)["passed"]


def test_config_reports_structured_loading_failure(tmp_path):
    trimesh.creation.box(extents=(4, 2, 1)).export(tmp_path / "barge.stl")
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump({"geometry": {"file": "barge.stl", "units": "m"},
        "loading": {"mass_kg": 2050., "draft_m": .9, "cg_frd_m": [0, 0, 0]},
        "operating_envelope": {"max_forward_speed_mps": 1.},
        "generation": {"disable_bem": True, "lut_samples": 3}}))
    with pytest.raises(GenerationFailure) as caught:
        generate_from_config(path, tmp_path / "bad")
    assert caught.value.code == "NO_EQUILIBRIUM_WATERLINE" or caught.value.code == "INVALID_LOADING"


def test_rectangular_crossflow_against_independent_integrals():
    length, draft, density, cd = 4., .5, 1025., 2.
    count = 201
    dx = length / count
    stations = [{"x_m": -length/2 + (i+.5)*dx, "y_m": 0., "beam_m": 2.,
                 "draft_m": draft, "dx_m": dx, "cd": cd, "lift_base_kg_per_m": 0.,
                 "lateral_force_method": "translation_shear_v5"} for i in range(count)]
    model = SectionalCrossflow.from_stations(stations, density=density)
    def force(v, r):
        state = VesselState(torch.zeros(3, dtype=torch.float64),
            torch.tensor([1., 0., 0., 0.], dtype=torch.float64),
            torch.tensor([0., v, 0., 0., 0., r], dtype=torch.float64))
        return model.evaluate(state).tau_body.numpy()
    sway = force(.3, 0.)
    assert sway[1] == pytest.approx(-.5*density*cd*draft*length*.3**2, rel=.01)
    assert abs(sway[5]) < 1e-9
    yaw = force(0., .2)
    assert yaw[5] == pytest.approx(-.5*density*cd*draft*.2**2*length**4/32, rel=.01)
    assert abs(yaw[1]) < 1e-9
    assert .3*sway[1] < 0 and .2*yaw[5] < 0


@pytest.mark.parametrize("name,scale", [
    ("prolate_spheroid", (3., .6, .5)),
    ("displacement_monohull", (2.5, .5, .4)),
    ("short_wide_asv", (1.5, .75, .4)),
])
def test_ellipsoid_family_through_production_generator(tmp_path, name, scale):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale(scale)
    root = _generate(tmp_path, name, mesh, mesh.volume * 1025 / 2)
    package = load_coefficient_package(root / "coefficient_package.yaml")
    assert package["mass"]["rigid_body_mass_kg"] > 0
    assert validate_coefficients(root / "coefficient_package.yaml", grid_size=3)["passed"]


def test_small_geometry_perturbation_is_continuous(tmp_path):
    base = _generate(tmp_path, "base", trimesh.creation.box(extents=(4, 2, 1)), 2050.)
    changed = _generate(tmp_path, "length_plus_point_one_percent",
                        trimesh.creation.box(extents=(4.004, 2, 1)), 2050.)
    first = load_coefficient_package(base / "coefficient_package.yaml")
    second = load_coefficient_package(changed / "coefficient_package.yaml")
    for nu in (np.array([.5, .1, 0, 0, 0, .05]), np.array([1., -.2, 0, 0, 0, -.1])):
        a = reference_wrench(first, nu)
        b = reference_wrench(second, nu)
        assert np.linalg.norm(b-a) / max(np.linalg.norm(a), 1) < .2


def test_strip_added_mass_constant_section_closed_form():
    length, beam, draft, density, count = 4., 2., .5, 1025., 1000
    dx = length / count
    stations = [{"x_m": -length/2 + (i+.5)*dx, "beam_m": beam,
                 "draft_m": draft, "dx_m": dx} for i in range(count)]
    matrix = np.asarray(strip_added_mass(stations, density)["matrix_6x6"])
    sway_per_length = density * np.pi * (draft/2)**2
    assert matrix[1, 1] == pytest.approx(sway_per_length * length, rel=1e-12)
    assert matrix[5, 5] == pytest.approx(sway_per_length * length**3/12, rel=1e-5)
    assert abs(matrix[1, 5]) < 1e-9
    assert np.allclose(matrix, matrix.T)


def test_strip_added_mass_translation_mirror_and_scale():
    stations = [{"x_m": x, "beam_m": 1., "draft_m": .4, "dx_m": .2}
                for x in (-.25, -.05, .15, .35)]
    base = np.asarray(strip_added_mass(stations, 1025.)["matrix_6x6"])
    translated = np.asarray(strip_added_mass([{**s, "x_m": s["x_m"]+2} for s in stations], 1025.)["matrix_6x6"])
    mirrored = np.asarray(strip_added_mass([{**s, "x_m": -s["x_m"]} for s in stations], 1025.)["matrix_6x6"])
    scaled = np.asarray(strip_added_mass([{**s, "x_m": 2*s["x_m"],
        "beam_m": 2*s["beam_m"], "draft_m": 2*s["draft_m"], "dx_m": 2*s["dx_m"]}
        for s in stations], 1025.)["matrix_6x6"])
    assert translated[1, 1] == pytest.approx(base[1, 1])
    assert mirrored[1, 1] == pytest.approx(base[1, 1])
    assert mirrored[5, 5] == pytest.approx(base[5, 5])
    assert mirrored[1, 5] == pytest.approx(-base[1, 5])
    assert scaled[1, 1] == pytest.approx(8*base[1, 1])
    assert scaled[5, 5] == pytest.approx(32*base[5, 5])


def _half_ellipsoid_translational_added_mass(axes, density):
    """Independent potential-flow result: rigid-lid half ellipsoid at omega=0."""
    roots, weights = np.polynomial.legendre.leggauss(256)
    s = (roots + 1) / 2
    t = s / (1-s)
    volume = 4*np.pi*np.prod(axes)/3
    result = []
    for axis in axes:
        denominator = (axis*axis+t)*np.sqrt(np.prod([a*a+t for a in axes], axis=0))
        shape_integral = np.prod(axes)*np.dot(weights/2, 1/(denominator*(1-s)**2))
        result.append(.5*density*volume*shape_integral/(2-shape_integral))
    return np.array(result)


def test_production_bem_spheroid_analytic_and_refinement(tmp_path, monkeypatch):
    pytest.importorskip("capytaine")
    monkeypatch.setenv("CAPYTAINE_CACHE_DIR", str(tmp_path / "capytaine_cache"))
    axes = (3., .6, .5)
    expected = _half_ellipsoid_translational_added_mass(axes, 1025.)
    generated = []
    for level in (10, 14, 18):
        mesh = trimesh.creation.uv_sphere(radius=1, count=(level, level))
        mesh.apply_scale(axes)
        source = tmp_path / f"spheroid_{level}.stl"
        mesh.export(source)
        root = generate_simple_vessel(geometry=source, output=tmp_path / f"bem_{level}",
            mass_kg=mesh.volume*1025/2, cg_frd_m=(0., 0., 0.), units="m",
            disable_bem=False, lut_samples=3, bem_panel_target=900)
        selected = json.loads((root / "added_mass.json").read_text())["selected"]
        assert selected["source"] == "bem"
        matrix = np.asarray(selected["matrix_6x6"])
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() > -1e-6
        assert matrix[0, 0] == pytest.approx(expected[0], rel=.05)
        assert matrix[1, 1] == pytest.approx(expected[1], rel=.05)
        generated.append(matrix)
    for axis in (0, 1):
        coarse_change = abs(generated[1][axis, axis]-generated[0][axis, axis])/generated[0][axis, axis]
        fine_change = abs(generated[2][axis, axis]-generated[1][axis, axis])/generated[1][axis, axis]
        assert coarse_change < .02 and fine_change < .02


def test_catamaran_spacing_changes_rotational_yaw_drag(tmp_path):
    moments = []
    for spacing in (.7, 1., 1.3):
        port = trimesh.creation.box(extents=(4, .6, 1))
        starboard = port.copy()
        port.apply_translation((0, -spacing, 0))
        starboard.apply_translation((0, spacing, 0))
        root = _generate(tmp_path, f"spacing_{spacing}",
            trimesh.util.concatenate((port, starboard)), 1230.)
        package = load_coefficient_package(root / "coefficient_package.yaml")
        moment = reference_wrench(package, np.array([0., 0., 0., 0., 0., .2]))[5]
        moments.append(moment)
    assert moments[0] > moments[1] > moments[2]  # increasingly negative damping


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_dimension_sweep_stays_finite(tmp_path, axis):
    values = []
    for factor in (.75, 1., 1.25):
        size = [4., 2., 1.]
        size[axis] *= factor
        mesh = trimesh.creation.box(extents=size)
        root = _generate(tmp_path, f"axis_{axis}_{factor}", mesh,
                         1025 * size[0]*size[1]*size[2] / 4)
        package = load_coefficient_package(root / "coefficient_package.yaml")
        wrench = reference_wrench(package, np.array([.5, .1, 0, 0, 0, .1]))
        assert np.isfinite(wrench).all()
        values.append(package["hydrostatics"]["volume_m3"])
    assert values[0] < values[1] < values[2]


def test_section_fullness_sweep_stays_finite(tmp_path):
    values = []
    for exponent in (.75, 1., 1.25):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices = np.sign(mesh.vertices) * np.abs(mesh.vertices)**exponent
        mesh.apply_scale((2., 1., .5))
        root = _generate(tmp_path, f"fullness_{exponent}", mesh, mesh.volume*1025/2)
        package = load_coefficient_package(root / "coefficient_package.yaml")
        assert np.isfinite(reference_wrench(package, np.array([.5, .1, 0, 0, 0, .1]))).all()
        values.append(package["hydrostatics"]["volume_m3"])
    assert len(set(round(v, 4) for v in values)) == 3


def test_small_vertex_noise_and_waterline_shift(tmp_path):
    base_mesh = trimesh.creation.icosphere(subdivisions=2)
    base_mesh.apply_scale((2., .8, .5))
    baseline = _generate(tmp_path, "noise_base", base_mesh, base_mesh.volume*1025/2)
    noisy_mesh = base_mesh.copy()
    rng = np.random.default_rng(11)
    noisy_mesh.vertices += rng.normal(0, 1e-4, noisy_mesh.vertices.shape)
    noisy = _generate(tmp_path, "noise_small", noisy_mesh, base_mesh.volume*1025/2)
    first = load_coefficient_package(baseline / "coefficient_package.yaml")
    second = load_coefficient_package(noisy / "coefficient_package.yaml")
    nu = np.array([.5, .1, 0, 0, 0, .05])
    a, b = reference_wrench(first, nu), reference_wrench(second, nu)
    assert np.linalg.norm(a-b)/max(np.linalg.norm(a), 1.) < .2
    source = tmp_path / "waterline.stl"
    trimesh.creation.box(extents=(4., 2., 1.)).export(source)
    shifted = generate_simple_vessel(geometry=source, output=tmp_path / "waterline_shift",
        mass_kg=2050., cg_frd_m=(0., 0., 0.), units="m", draft_m=.2501,
        disable_bem=True, lut_samples=3)
    original = generate_simple_vessel(geometry=source, output=tmp_path / "waterline_base",
        mass_kg=2050., cg_frd_m=(0., 0., 0.), units="m", draft_m=.25,
        disable_bem=True, lut_samples=3)
    first = load_coefficient_package(original / "coefficient_package.yaml")
    second = load_coefficient_package(shifted / "coefficient_package.yaml")
    a, b = reference_wrench(first, nu), reference_wrench(second, nu)
    assert np.linalg.norm(a-b)/max(np.linalg.norm(a), 1.) < .2
