"""Identification mathematics, independent of CFD and experimental tuning."""
import math

import numpy as np

from tools.validate_kcs_hmri_passive import fit_odd, harmonics


def test_odd_polynomial_recovery():
    x = np.array([-.5, -.3, -.1, .1, .3, .5])
    assert np.allclose(fit_odd(x, -.2*x - .04*x**3), (-.2, -.04))


def test_pmm_first_harmonic_and_cycle_mean():
    phases = np.linspace(0., 2*math.pi, 128, endpoint=False)
    amplitude = .4
    def synthetic(v, r):
        vector = np.zeros(6)
        vector[1] = -.2*r - .05*r**3 + .7*v*r**2 + .3*v*v*r
        return {name: vector for name in ("physical", "rhs", "linear", "nonlinear",
                                   "crossflow", "added_coriolis", "rigid_coriolis")}
    result = harmonics(synthetic, -.1, amplitude, phases)["physical"]
    expected_sine = -.2*amplitude - .75*.05*amplitude**3 + .3*.01*amplitude
    expected_mean = .5*.7*(-.1)*amplitude**2
    np.testing.assert_allclose(result["sine_first"][1], expected_sine, atol=1e-12)
    np.testing.assert_allclose(result["mean"][1], expected_mean, atol=1e-12)
