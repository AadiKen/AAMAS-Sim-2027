"""Generic physics gates that run before the locked KCS comparison."""
from dataclasses import replace
import math

import numpy as np
import torch
import trimesh

from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.vessel_generation.simple_crossflow import extract_crossflow_stations
from bcod_sim.vessel_generation.simple_models import crossflow


def _fixture(mesh, waterline=0., froude=.2):
    stations, provenance = extract_crossflow_stations(mesh, waterline, max_froude=froude)
    model = SectionalCrossflow.from_stations(stations)
    model.validate(dtype=torch.float64, device=torch.device("cpu"))
    return stations, provenance, model


def _force(model, u, v, r):
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    state = VesselState(t([0., 0., 0.]), t([1., 0., 0., 0.]), t([u, v, 0., 0., 0., r]))
    return model.evaluate(state)


def test_barge_section_area_cd_and_direct_integral():
    stations, provenance, model = _fixture(trimesh.creation.box(extents=(4., 2., 1.)))
    assert provenance["linear_lift"][0]["method"] == "unsupported_lift_crossflow_only"
    assert len(stations) == 31
    for s in stations:
        assert np.isclose(s["submerged_section_area_m2"], 1.)
        assert np.isclose(s["lateral_projected_depth_m"], .5)
        assert np.isclose(s["section_fullness"], 1.)
        assert s["cd"] == 2.
    result = _force(model, 0., .4, .2)
    expected = np.array([-.5*1025*s["cd"]*s["draft_m"]*
        abs(.4+s["x_m"]*.2)*(.4+s["x_m"]*.2)*s["dx_m"] for s in stations])
    assert np.allclose(result.diagnostics["section_force_n"], expected)
    assert np.isclose(result.tau_body[1], expected.sum())
    assert np.isclose(result.tau_body[5], sum(s["x_m"]*f for s, f in zip(stations, expected)))


def test_slender_analytic_lift_and_symmetry():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale((3., .35, .4))
    stations, provenance, model = _fixture(mesh)
    hull = provenance["linear_lift"][0]
    assert hull["method"] == "inoue_guarded_displacement_v1"
    expected = math.pi/2*(2*hull["draft_m"]/hull["length_m"]) + \
        1.4*hull["sectional_block_coefficient"]*hull["beam_m"]/hull["length_m"]
    assert np.isclose(hull["lift_coefficient"], expected)
    k = .5*1025*hull["length_m"]*hull["draft_m"]*expected
    result = _force(model, 1., 1e-6, 0.)
    assert np.isclose(float(result.tau_body[1])/1e-6, -k, rtol=1e-4)
    for v, r in ((.2, 0.), (0., .3), (.2, .1)):
        positive = _force(model, 1., v, r).tau_body
        negative = _force(model, 1., -v, -r).tau_body
        assert np.allclose(positive, -negative, atol=1e-8)


def test_catamaran_zero_speed_reverse_and_reference_shift():
    left = trimesh.creation.box(extents=(4., .7, 1.))
    right = left.copy()
    left.apply_translation((0., -1., 0.))
    right.apply_translation((0., 1., 0.))
    stations, provenance, model = _fixture(trimesh.util.concatenate((left, right)))
    assert provenance["hull_components"] == 2
    assert {s["hull_id"] for s in stations} == {0, 1}
    assert min(s["y_m"] for s in stations) < -.9
    assert max(s["y_m"] for s in stations) > .9
    sway = _force(model, 0., .4, 0.).tau_body
    assert abs(float(sway[5])) < 1e-8
    assert float(_force(model, 0., 0., .2).tau_body[5]) < 0
    for u in (-2., -1., 0., 1., 2.):
        for v in (-1., -.4, 0., .4, 1.):
            for r in (-.5, -.2, 0., .2, .5):
                force = _force(model, u, v, r).tau_body
                assert np.isfinite(force.numpy()).all()
                assert v*float(force[1]) + r*float(force[5]) <= 1e-8
    shifted = replace(model, moment_reference_x_m=.35)
    old = _force(model, 0., .4, .2).tau_body
    new = _force(shifted, 0., .4, .2).tau_body
    assert np.isclose(float(new[1]), float(old[1]))
    assert np.isclose(float(new[5]), float(old[5])-.35*float(old[1]))
    assert np.allclose(crossflow(stations, 1025., np.array((0., .4, 0., 0., 0., .2)))[[1,5]],
                       old.numpy()[[1,5]])


def test_incidence_transition_continuity_and_energy():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale((3., .35, .4))
    stations, _, model = _fixture(mesh)
    assert all(s["lateral_force_method"] == "translation_shear_v5" for s in stations)
    for u in (-2., -.1, -1e-8, 0., 1e-8, .1, 2.):
        for v in (-1., -.2, -1e-8, 0., 1e-8, .2, 1.):
            for r in (-.3, 0., .3):
                result = _force(model, u, v, r)
                tau = result.tau_body.numpy()
                assert np.isfinite(tau).all()
                assert v*tau[1] + r*tau[5] <= 1e-8
                assert np.allclose(tau, crossflow(stations, 1025.,
                    np.array([u, v, 0., 0., 0., r])), atol=1e-8)
                weights = result.diagnostics["station_crossflow_weight"].numpy()
                assert np.all((weights >= 0) & (weights <= 1))
    for u in (-.5, 0., .5):
        for v in (-.2, 0., .2):
            a = _force(model, u, v-1e-8, .1).tau_body.numpy()
            b = _force(model, u, v+1e-8, .1).tau_body.numpy()
            assert np.max(np.abs(a-b)) < 1e-3


def test_fine_ended_and_hard_chine_sections_remain_bounded():
    rounded = trimesh.creation.icosphere(subdivisions=2)
    rounded.apply_scale((3., .35, .4))
    chine = trimesh.creation.box(extents=(4., 1.4, .8))
    for mesh in (rounded, chine):
        stations, _, model = _fixture(mesh)
        cds = np.array([s["cd"] for s in stations])
        assert np.isfinite(cds).all()
        assert cds.min() > 0 and cds.max() <= 2
        assert np.isfinite(_force(model, 1., .3, .2).tau_body.numpy()).all()
    rounded_depths = [s["draft_m"] for s in _fixture(rounded)[0]]
    assert min(rounded_depths) < .3*max(rounded_depths)
