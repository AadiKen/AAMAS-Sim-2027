"""Focused checks for the passive SST diagnostic postprocessor."""

import numpy as np

from tools.analyze_stage3_sst_diagnostic_probes import interface_elevation, lag_scan


def test_interface_position_from_vertical_probe_triplet() -> None:
    xy = [(float(i), 0.0) for i in range(30)]
    points = np.asarray([(x, y, z) for x, y in xy for z in (-.125, 0., .125)])
    alpha = np.asarray([[.625, .5, .375] * 30, [.635, .51, .385] * 30])
    locations, eta = interface_elevation(alpha, points)
    assert locations.shape == (30, 2)
    assert np.allclose(eta[0], 0.)
    assert np.allclose(eta[1], .01)


def test_positive_lag_means_downstream_signal_follows() -> None:
    dt = .1
    time = np.arange(200) * dt
    upstream = np.sin(2 * np.pi * .33 * time)
    downstream = np.sin(2 * np.pi * .33 * (time - .4))
    result = lag_scan(upstream, downstream, dt, .8)
    assert result["valid"]
    assert abs(result["best_lag_s"] - .4) <= dt
    assert result["best_correlation"] > .99
