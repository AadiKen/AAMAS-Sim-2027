from pathlib import Path

import numpy as np
import trimesh

from bcod_sim.vessel_generation.twodt.sections import (
    extract_section_contours, representative_sections,
)


ROOT = Path(__file__).resolve().parents[2]


def test_catamaran_section_components_and_deterministic_family_map():
    mesh = trimesh.load(ROOT / 'stage3_results/simple_hydrodynamics/phase3a/final_fleet_equilibrium/catamaran/processed_geometry.stl', force='mesh')
    sections = extract_section_contours(mesh, 0., count=11)
    assert len({s.hull_id for s in sections}) == 2
    assert all(s.draft_m > 0 and s.beam_m > 0 and 0 < s.fullness <= 1. for s in sections)
    assert all(s.contour_yz_m.shape == (64, 2) for s in sections)
    first = representative_sections(sections, maximum=3)
    second = representative_sections(sections, maximum=3)
    assert first == second
    assert first['family_count'] == 6
    assert len(first['section_to_family']) == len(sections)
    assert np.isfinite([s.descriptor for s in sections]).all()
