"""Analytic gate for the experimental SQCM source-panel component."""
import numpy as np
import pytest

from bcod_sim.vessel_generation.sqcm.source import cube_sphere_quads, solve_source
from bcod_sim.vessel_generation.sqcm.wigley import wigley_source_panels


@pytest.mark.parametrize("divisions", [2, 4, 6])
def test_cubed_sphere_source_boundary_and_symmetry(divisions):
    panels = cube_sphere_quads(1., divisions)
    result = solve_source(panels, np.array([1., 0., 0.]))
    assert np.max(np.abs(result["normal_residual"])) < 1e-11
    assert result["condition_number"] < 10
    cp = result["cp"]
    force = np.sum((-cp * panels.areas)[:, None] * panels.normals, axis=0)
    assert np.linalg.norm(force) < 1e-10
    exact_cp = -1.25 + 2.25 * panels.normals[:, 0]**2
    assert np.sqrt(np.mean((cp-exact_cp)**2)) < .08


def test_invalid_panel_rejected():
    with pytest.raises(ValueError, match="Degenerate"):
        from bcod_sim.vessel_generation.sqcm.source import SourcePanels
        SourcePanels.from_corners(np.zeros((1, 4, 3)))


@pytest.mark.parametrize("nx,nz", [(8, 3), (16, 4), (30, 5)])
def test_wigley_source_only_straight_ahead_symmetry(nx, nz):
    panels, physical = wigley_source_panels(longitudinal=nx, vertical=nz)
    result = solve_source(panels, np.array([-.5, 0., 0.]))
    panel_force = (-result["cp"][physical] * panels.areas[physical])[:, None] * panels.normals[physical]
    force = panel_force.sum(axis=0)
    moment = np.cross(panels.centers[physical], panel_force).sum(axis=0)
    assert abs(force[1]) < 1e-10
    assert abs(moment[2]) < 1e-10
    assert np.max(abs(result["normal_residual"])) < 1e-10
