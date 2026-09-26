"""Analytical thin-ship offset and numerical convergence checks."""
import numpy as np

from bcod_sim.vessel_generation.simple_wave import michell_wave_resistance, thin_ship_applicable


def test_wigley_michell_wave_resistance_refines_and_stays_positive():
    # Standard Wigley form has Cb=4/9; no experimental resistance enters this test.
    length, beam, draft, speed = 3., .3, .1875, 1.627
    depth = np.linspace(0., draft, 33)
    values = []
    for segments in (40, 80):
        x = np.linspace(-length/2, length/2, segments+1)
        half_width = beam/2 * (1-(2*x[:, None]/length)**2) * (1-(depth[None, :]/draft)**2)
        values.append(michell_wave_resistance(x, depth, half_width, speed, 1000.))
    assert values[0] > 0
    assert abs(values[1]-values[0])/values[1] < .01
    assert 3.5 < values[1] < 4.1


def test_thin_ship_domain_rejects_fuller_and_multihull_forms():
    assert thin_ship_applicable(length_beam=10., block_coefficient=4/9,
                                max_froude=.3, hull_count=1)
    assert not thin_ship_applicable(length_beam=6., block_coefficient=.7,
                                    max_froude=.26, hull_count=1)
    assert not thin_ship_applicable(length_beam=12., block_coefficient=.4,
                                    max_froude=.3, hull_count=2)
