import numpy as np

from bcod_sim.vessel_generation.sqcm.vortex import (
    horseshoe_velocity, make_horseshoes, qcm_longitudinal_positions,
    segment_velocity,
)


def test_biot_savart_segment_sign_and_oddness():
    point = np.array([[1., 0., 0.]])
    a, b = np.array([0., -1., 0.]), np.array([0., 1., 0.])
    velocity = segment_velocity(point, a, b)[0]
    assert velocity[2] < 0
    assert np.allclose(velocity, -segment_velocity(point, b, a)[0])
    assert np.isclose(np.linalg.norm(velocity), np.sqrt(2)/(4*np.pi))


def test_qcm_chebyshev_positions_and_wake_topologies():
    x, weights = qcm_longitudinal_positions(-1., 1., 8)
    assert np.all(np.diff(x) > 0)
    assert np.isclose(x[0], -x[-1])
    assert np.isclose(weights.sum(), 0., atol=1e-14)
    base = dict(x_stern=-1., x_bow=1., z_waterline=0., z_keel=0.3,
                longitudinal=8, vertical=2, wake_elements=5)
    one = make_horseshoes(**base, model=1)
    two = make_horseshoes(**base, model=2)
    assert len(one) == len(two) == 16
    assert np.isclose(one[-1].leg_end[1, 0], -1.)
    assert np.isclose(two[-1].leg_end[1, 0], two[-1].bound_end[0]-6/5)
    assert two[-1].leg_end[1, 0] > one[-1].leg_end[1, 0]
    velocity = horseshoe_velocity(np.array([[0., 0.1, 0.2]]), two, core_radius=.01)
    assert velocity.shape == (1, 16, 3)
    assert np.isfinite(velocity).all()
