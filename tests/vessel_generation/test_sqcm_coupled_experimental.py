import math

import numpy as np

from bcod_sim.vessel_generation.sqcm.coupled_solver import (
    mirror_horseshoes, solve_coupled, vortex_collocation,
)
from bcod_sim.vessel_generation.sqcm.vortex import make_horseshoes
from bcod_sim.vessel_generation.sqcm.wigley import wigley_source_panels


def _run(beta_degrees, model):
    panels, physical = wigley_source_panels(longitudinal=8, vertical=2)
    angle = math.radians(beta_degrees)
    horseshoes = make_horseshoes(x_stern=-1.25, x_bow=1.25,
                                z_waterline=0., z_keel=.156,
                                longitudinal=8, vertical=2,
                                wake_elements=10, drift_angle=angle, model=model)
    horseshoes += mirror_horseshoes(horseshoes)
    points, normals = vortex_collocation(horseshoes, x_stern=-1.25,
                                         x_bow=1.25, longitudinal=8)
    ambient = np.array([-.5*math.cos(angle), .5*math.sin(angle), 0.])
    return solve_coupled(panels, physical, horseshoes, points, normals,
                         ambient, core_radius=.01)


def test_fixed_wake_coupled_boundary_signs_and_symmetry():
    for model in (1, 2):
        positive, negative = _run(10, model), _run(-10, model)
        assert positive.force[1] > 0
        assert np.isclose(positive.force[1], -negative.force[1], atol=1e-10)
        assert np.isclose(positive.moment[2], -negative.moment[2], atol=1e-10)
        assert max(abs(positive.normal_residual)) < 1e-10
        assert max(abs(positive.vortex_plane_residual)) < 1e-10
