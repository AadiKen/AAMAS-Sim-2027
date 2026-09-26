from pathlib import Path

import trimesh

from bcod_sim.vessel_generation.sqcm.geometry import reconstruct


ROOT = Path(__file__).resolve().parents[2]


def test_kcs_structured_geometry_preserves_nominal_submerged_shape():
    source = ROOT / 'stage3_results/simple_hydrodynamics/phase3a/final_kcs/canonical/processed_geometry.stl'
    hull = trimesh.load(source, force='mesh')
    result = reconstruct(hull, 0., longitudinal=30, vertical=5)
    assert result.preservation['accepted']
    assert result.preservation['relative_volume_error'] < .005
    assert result.preservation['relative_wetted_area_error'] < .03
    assert len(result.component_properties) == 1


def test_catamaran_components_remain_separate():
    source = ROOT / 'stage3_results/simple_hydrodynamics/phase3a/final_fleet_centered/catamaran/processed_geometry.stl'
    hull = trimesh.load(source, force='mesh')
    result = reconstruct(hull, 0., longitudinal=12, vertical=3)
    assert result.preservation['accepted']
    assert len(result.component_properties) == 2
    centers = sorted(row['y_center'] for row in result.component_properties)
    assert centers[0] < 0 < centers[1]
