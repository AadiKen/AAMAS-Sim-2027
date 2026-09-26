import numpy as np
import pytest
import torch

from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.vessel_generation.twodt.level_a import (
    SectionalTwoDt, cylinder_startup_ratio, sectional_v5_arrays,
)


def _stations(x=(-1., 0., 1.), hull=0, y=0.):
    return [dict(x_m=float(xi), y_m=float(y), draft_m=.4, dx_m=1., cd=1.2,
                 lift_base_kg_per_m=10., hull_id=hull,
                 lateral_force_method="translation_shear_v5", beam_m=.8)
            for xi in x]


@pytest.mark.parametrize("u,v,r", [(1., .15, 0.), (1., 0., .2), (1., .1, -.12), (0., .3, .2), (-1., .2, .1)])
def test_instantaneous_mode_reproduces_frozen_v5(u, v, r):
    stations = _stations()
    model = SectionalCrossflow.from_stations(stations)
    nu = torch.tensor([u, v, 0., 0., 0., r], dtype=torch.float64)
    state = VesselState(torch.zeros(3, dtype=torch.float64),
                        torch.tensor([1., 0., 0., 0.], dtype=torch.float64), nu)
    frozen = model.evaluate(state).tau_body.numpy()
    instant = sectional_v5_arrays(stations, u, v, r)
    assert np.allclose([instant["Y_n"], instant["N_nm"]], frozen[[1, 5]], atol=1e-10)


def test_published_startup_shape_and_forward_age():
    ratio = cylinder_startup_ratio(np.array([0., 5., 9., 25., 100.]))
    assert np.isclose(ratio[0], .07339/1.2)
    assert ratio[0] < ratio[1] < ratio[2]
    assert ratio[2] > ratio[3]
    assert ratio[3] == ratio[4]
    out = SectionalTwoDt(_stations()).steady_captive(2., .2, 0.)
    assert np.all(np.diff(out["t_star"]) < 0)
    assert np.all(out["dY_n"] < 0)
    assert np.isfinite(out["N_nm"])


def test_drift_yaw_symmetry_reference_and_multihull():
    stations = _stations(hull=0, y=-.8) + _stations(hull=1, y=.8)
    model = SectionalTwoDt(stations)
    a, b = model.steady_captive(2., .2, 0.), model.steady_captive(2., -.2, 0.)
    assert np.isclose(a["Y_n"], -b["Y_n"])
    assert np.isclose(a["N_nm"], -b["N_nm"])
    # A bow-to-stern startup history is longitudinally asymmetric even on a
    # geometrically symmetric hull; the two equal hulls must contribute equally.
    assert np.isclose(a["N_nm"], 2*SectionalTwoDt(_stations()).steady_captive(2., .2, 0.)["N_nm"])
    spun = model.steady_captive(2., 0., .1)
    assert np.isfinite(spun["N_nm"])
    shifted = SectionalTwoDt(stations, reference_x=.25).steady_captive(2., .2, 0.)
    assert np.isclose(shifted["N_nm"], a["N_nm"]-.25*a["Y_n"])


def test_unsupported_axial_speed_is_rejected():
    model = SectionalTwoDt(_stations())
    with pytest.raises(ValueError, match="forward-plane"):
        model.steady_captive(0., .1, 0.)
    with pytest.raises(ValueError, match="forward-plane"):
        model.steady_captive(-1., .1, 0.)
