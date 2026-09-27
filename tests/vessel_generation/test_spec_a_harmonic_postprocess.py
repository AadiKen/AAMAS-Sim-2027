from __future__ import annotations

import math

import numpy as np
import pytest

from bcod_sim.vessel_generation.spec_a.harmonic_postprocess import (
    FLIP, body_kinematics, harmonic, read_forces, require_compatible, world_to_body_cg,
)


def _world_body(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ FLIP


@pytest.mark.parametrize("yaw", [-.7, 0., .53])
@pytest.mark.parametrize("cg,origin", [((.4, -.3, .2), (0., 0., 0.)),
                                            ((-1.2, .5, .1), (.8, -2., .3))])
def test_rotating_translating_wrench_and_reference_invariance(yaw, cg, origin):
    fb, mb = np.array([17., -4., 2.]), np.array([3., -7., 11.])
    rot = _world_body(yaw)
    fw, mcg_world = rot @ fb, rot @ mb
    cg, origin = np.asarray(cg), np.asarray(origin)
    mo = mcg_world + np.cross(cg-origin, fw)
    actual_f, actual_m = world_to_body_cg(fw, mo, origin_world=origin, cg_world=cg,
                                           yaw_foam_rad=yaw)
    np.testing.assert_allclose(actual_f, fb, atol=1e-12)
    np.testing.assert_allclose(actual_m, mb, atol=1e-12)
    shifted_origin = origin + np.array([2., -.6, .9])
    shifted_moment = mcg_world + np.cross(cg-shifted_origin, fw)
    _, shifted = world_to_body_cg(fw, shifted_moment, origin_world=shifted_origin,
                                   cg_world=cg, yaw_foam_rad=yaw)
    np.testing.assert_allclose(shifted, mb, atol=1e-12)


def test_harmonic_signed_linear_cubic_and_acceleration_irregular_timestamps():
    rng = np.random.default_rng(4)
    t = np.r_[np.linspace(6.25, 18.75, 3200), rng.uniform(6.25, 18.75, 1700)]
    t = np.unique(np.sort(t))
    omega = 2*math.pi*.08
    a, linear, cubic, accel = .4, -12., -8., 3.5
    rate = a*np.cos(omega*t)
    signal = linear*rate+cubic*rate**3-accel*np.sin(omega*t)
    h = harmonic(t, signal)
    assert h["rate_1"] == pytest.approx(linear*a+.75*cubic*a**3, abs=1e-5)
    assert h["rate_3"] == pytest.approx(.25*cubic*a**3, abs=1e-5)
    assert h["accel_1"] == pytest.approx(accel, abs=1e-5)
    cubic_recovered = 4*h["rate_3"]/a**3
    linear_recovered = (h["rate_1"]-.75*cubic_recovered*a**3)/a
    assert (linear_recovered, cubic_recovered) == pytest.approx((linear, cubic), abs=1e-5)


def test_restart_latest_owns_overlap(tmp_path):
    for name, rows in (("0", [(0, 1), (1, 2), (2, 3)]),
                       ("1.5", [(1.5, 9), (2, 10), (3, 11)])):
        p = tmp_path / "postProcessing/forces" / name / "forces.dat"
        p.parent.mkdir(parents=True)
        p.write_text("# CofR : (0 0 0)\n" + "".join(
            f"{t} (({v} 0 0) (0 0 0)) ((0 0 0) (0 0 0))\n" for t, v in rows))
    data = read_forces(tmp_path)
    assert data[:, 0].tolist() == [0, 1, 1.5, 2, 3]
    assert data[:, 1].tolist() == [1, 2, 9, 10, 11]


def test_reject_unresolved_and_incompatible_reference_metadata():
    good = {"load_kind": "physical", "frame": "FRD", "moment_origin": "CG",
            "cg_world_m": [0, 0, 0], "phase": "rate_cos", "normalization": "U0"}
    require_compatible(good, dict(good))
    with pytest.raises(ValueError):
        require_compatible(good, {**good, "cg_world_m": None})
    with pytest.raises(ValueError):
        require_compatible(good, {**good, "frame": "world"})
    with pytest.raises(ValueError):
        require_compatible(good, {**good, "cg_world_m": [0.1, 0, 0]})
    with pytest.raises(ValueError):
        world_to_body_cg([1, 0, 0], [0, 0, 0], origin_world=[0]*3,
                         cg_world=[0]*3, yaw_foam_rad=0, input_frame="body_frd")


def test_pure_yaw_kinematics_preserves_nearly_zero_sway_but_changes_u():
    theta = np.array([0., -.25])
    u, v, r = body_kinematics(theta, 1.953*np.tan(theta), np.array([0., -.1]))
    np.testing.assert_allclose(v, 0., atol=1e-12)
    np.testing.assert_allclose(u, 1.953/np.cos(theta), atol=1e-12)
    assert r[-1] == pytest.approx(.1)
