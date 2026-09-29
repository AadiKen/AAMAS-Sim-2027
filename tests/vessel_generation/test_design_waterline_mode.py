"""Regression coverage for closed hulls capped at their design waterline."""
import json
import math

import numpy as np
import pytest
import trimesh
import torch

from bcod_sim.vessel_generation.coefficient_package import (
    _runtime_plant, load_coefficient_package, reference_wrench,
)
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel


def _clip_and_cap_at_zero(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clip a convex fixture to z>=0 and add a planar triangulated waterplane."""
    vertex_rows = []
    faces = []
    index = {}

    def add(point):
        key = tuple(np.round(point, 10))
        if key not in index:
            index[key] = len(vertex_rows)
            vertex_rows.append(np.asarray(point, dtype=float))
        return index[key]

    boundary = {}
    for triangle in mesh.triangles:
        polygon = [point.copy() for point in triangle]
        clipped = []
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            p_in, q_in = p[2] >= -1e-10, q[2] >= -1e-10
            if p_in:
                point = p.copy()
                point[2] = 0. if abs(point[2]) <= 1e-10 else point[2]
                clipped.append(point)
            if p_in != q_in:
                cross = p + (-p[2] / (q[2] - p[2])) * (q - p)
                cross[2] = 0.
                clipped.append(cross)
        clean = []
        for point in clipped:
            if not clean or np.linalg.norm(point-clean[-1]) > 1e-9:
                clean.append(point)
        if len(clean) > 1 and np.linalg.norm(clean[0]-clean[-1]) <= 1e-9:
            clean.pop()
        clipped = clean
        if len(clipped) >= 3:
            ids = [add(point) for point in clipped]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                if abs(vertex_rows[a][2]) < 1e-9 and abs(vertex_rows[b][2]) < 1e-9:
                    boundary[a] = vertex_rows[a]
                    boundary[b] = vertex_rows[b]
            faces.extend((ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids)-1))
    ring = list(boundary)
    center = np.mean([vertex_rows[i][:2] for i in ring], axis=0)
    ring.sort(key=lambda i: math.atan2(vertex_rows[i][1]-center[1], vertex_rows[i][0]-center[0]))
    center_id = add((center[0], center[1], 0.))
    faces.extend((center_id, ring[i], ring[(i+1) % len(ring)]) for i in range(len(ring)))
    capped = trimesh.Trimesh(vertices=np.asarray(vertex_rows), faces=np.asarray(faces), process=True)
    capped.fix_normals()
    assert capped.is_watertight
    return capped


def _generate_pair(full_mesh, clipped_mesh, tmp_path, *, draft, density, known_length):
    full_path, clipped_path = tmp_path / "full.stl", tmp_path / "waterline.stl"
    full_mesh.export(full_path)
    clipped_mesh.export(clipped_path)
    volume = float(clipped_mesh.volume)
    mass = density * volume
    common = dict(units="m", reference_length_m=known_length, mass_kg=mass,
                  cg_frd_m=(0., 0., .25), water_density_kg_m3=density,
                  speed_range_mps=(0., 1.2), disable_bem=True, lut_samples=3)
    full = generate_simple_vessel(geometry=full_path, output=tmp_path / "full_package",
                                  draft_m=draft, **common)
    clipped = generate_simple_vessel(geometry=clipped_path, output=tmp_path / "capped_package",
        geometry_mode="design_waterline_submerged_hull", reference_draft_m=draft, **common)
    return full, clipped


def _hydro(root):
    return json.loads((root / "hydrostatics.json").read_text())


def test_rectangular_barge_local_hydrostatics_matches_full_hull(tmp_path):
    full_mesh = trimesh.creation.box(extents=(4., 2., 1.))
    clipped_mesh = trimesh.creation.box(extents=(4., 2., .5))
    clipped_mesh.apply_translation((0., 0., .25))
    full, capped = _generate_pair(full_mesh, clipped_mesh, tmp_path,
                                  draft=.5, density=1000., known_length=4.)
    a, b = _hydro(full), _hydro(capped)
    for field in ("volume_m3", "waterplane_area_m2", "waterplane_i_roll_m4",
                  "waterplane_i_pitch_m4", "wetted_area_m2"):
        assert b[field] == pytest.approx(a[field], rel=1e-8, abs=1e-8)
    assert b["center_buoyancy_frd_m"] == pytest.approx(a["center_buoyancy_frd_m"], abs=1e-8)
    assert b["waterplane_centroid_frd_m"] == pytest.approx(a["waterplane_centroid_frd_m"], abs=1e-8)
    assert b["waterplane_area_m2"] == pytest.approx(8.)
    assert b["waterplane_i_roll_m4"] == pytest.approx(8./3.)
    assert b["waterplane_i_pitch_m4"] == pytest.approx(32./3.)
    assert b["stiffness_6x6"][2][2] == pytest.approx(1000*9.80665*8.)
    assert b["stiffness_6x6"][3][3] == pytest.approx(1000*9.80665*4*(2./3))
    assert b["mode"] == "local_design_waterline"
    assert b["nonlinear_vertical_motion_supported"] is False


def test_curved_ellipsoid_fixture_and_horizontal_wrench_invariance(tmp_path):
    full_mesh = trimesh.creation.icosphere(subdivisions=2)
    full_mesh.apply_scale((2., 1., .5))
    clipped_mesh = _clip_and_cap_at_zero(full_mesh)
    assert clipped_mesh.is_watertight and clipped_mesh.is_winding_consistent
    full, capped = _generate_pair(full_mesh, clipped_mesh, tmp_path,
                                  draft=.5, density=1000., known_length=4.)
    a, b = _hydro(full), _hydro(capped)
    for field in ("volume_m3", "waterplane_area_m2", "waterplane_i_roll_m4",
                  "waterplane_i_pitch_m4", "wetted_area_m2"):
        assert b[field] == pytest.approx(a[field], rel=2e-6, abs=2e-7)
    assert b["center_buoyancy_frd_m"] == pytest.approx(a["center_buoyancy_frd_m"], abs=2e-7)
    assert b["waterplane_centroid_frd_m"] == pytest.approx(a["waterplane_centroid_frd_m"], abs=2e-7)
    assert b["stiffness_6x6"][2][2] == pytest.approx(a["stiffness_6x6"][2][2], rel=2e-6)
    assert b["stiffness_6x6"][3][3] == pytest.approx(a["stiffness_6x6"][3][3], rel=2e-6)
    assert b["stiffness_6x6"][4][4] == pytest.approx(a["stiffness_6x6"][4][4], rel=2e-6)

    package_a = load_coefficient_package(full / "coefficient_package.yaml")
    package_b = load_coefficient_package(capped / "coefficient_package.yaml")
    np.testing.assert_allclose(package_b["added_mass"]["matrix_6x6"],
                               package_a["added_mass"]["matrix_6x6"], rtol=3e-6, atol=2e-6)
    plant_a, plant_b = _runtime_plant(package_a), _runtime_plant(package_b)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    zeros = {name: t([0.] * 6) for name in EXTERNAL_TERMS}
    for u in (.2, .8, 1.2):
        for beta in (-.2, -.1, 0., .1, .2):
            for yaw_rate in (-.1, 0., .1):
                nu = np.array((u, u*math.sin(beta), 0., 0., 0., yaw_rate))
                assert reference_wrench(package_b, nu) == pytest.approx(
                    reference_wrench(package_a, nu), rel=3e-6, abs=2e-6)
                state = VesselState(t([0.,0.,0.]),t([1.,0.,0.,0.]),t(nu))
                da, db = plant_a.diagnostics(state,zeros), plant_b.diagnostics(state,zeros)
                keys=("added_mass_coriolis","linear_damping","nonlinear_damping","crossflow")
                wa=sum((da.terms[k].numpy() for k in keys),np.zeros(6))
                wb=sum((db.terms[k].numpy() for k in keys),np.zeros(6))
                assert wb[[0,1,5]] == pytest.approx(wa[[0,1,5]],rel=3e-6,abs=2e-6)
