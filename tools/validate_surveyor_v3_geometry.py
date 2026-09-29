"""Audit v3 source fidelity and fresh-water hydrostatics without weakening gates."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
from qualify_surveyor_envelopes import _surface_distances, _to_frd
from bcod_sim.vessel_generation.simple_sections import solve_waterline

BASE = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'
SOURCE = ROOT / 'docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/source/mixed_low_faces_source_mm.stl'


def stats(distances: np.ndarray, areas_mm2: np.ndarray) -> dict:
    return {'sample_count': len(distances), 'area_m2': float(areas_mm2.sum()/1e6),
            'median_mm': float(np.percentile(distances, 50)),
            'p90_mm': float(np.percentile(distances, 90)),
            'p95_mm': float(np.percentile(distances, 95)),
            'p99_mm': float(np.percentile(distances, 99)),
            'max_mm': float(distances.max()),
            'area_above_2mm_m2': float(areas_mm2[distances>2].sum()/1e6),
            'area_above_5mm_m2': float(areas_mm2[distances>5].sum()/1e6)}


def run() -> dict:
    source = trimesh.load_mesh(SOURCE, process=True)
    port = source.submesh([np.where(source.triangles_center[:, 0] < 0)[0]],
                          append=True, repair=False)
    centers = port.triangles_center
    areas = port.area_faces
    source_z = centers[:, 2]
    wet = centers[:, 1] <= -75
    ordinary = wet & ((source_z < 900) | (source_z > 1200))
    declared = {'aft_v3_cap': {'source_z_mm': [5, 95]},
                'forward_v2_cap': {'source_z_mm': [1750, 1779]},
                'local_v2_correction': {'source_z_mm': [500, 525],
                                        'source_x_mm': [-385, -360],
                                        'source_y_mm': [-260, -205]},
                'original_broken_bridge_boundary': {'source_z_mm': [325, 350]}}
    protected = ordinary.copy()
    masks = {}
    mask_arrays = {}
    for label, bounds in declared.items():
        low, high = bounds['source_z_mm']
        mask = ordinary & (source_z >= low) & (source_z < high)
        for axis, key in ((0, 'source_x_mm'), (1, 'source_y_mm')):
            if key in bounds:
                mask &= (centers[:, axis] >= bounds[key][0]) & (centers[:, axis] < bounds[key][1])
        mask_arrays[label] = mask
        masks[label] = {**bounds, 'source_area_m2': float(areas[mask].sum()/1e6),
                        'source_triangle_count': int(mask.sum())}
        protected &= ~mask
    original_step = Path.home() / 'Downloads' / 'Mock_Surveyor_1A (1).STEP'
    rows = {'schema': 'surveyor-v3-validation-1', 'original_step_sha256':
            sha256(original_step.read_bytes()).hexdigest(),
            'source_selected_faces_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
            'density_kg_m3': 1000, 'nominal_mass_kg': 52.3,
            'protected_source_masks': masks, 'variants': {}}
    for name in ('nominal', 'fuller', 'finer'):
        path = BASE / 'meshes' / f'surveyor_{name}_v3_aft_repair.stl'
        mesh = trimesh.load_mesh(path, process=True)
        d = _surface_distances(_to_frd(centers), mesh, 128)*1000
        hydro = solve_waterline(mesh, 52.3, 1000.)
        draft = float(mesh.bounds[1, 2] - hydro['waterline_z_frd_m'])
        waterline = mesh.section(plane_origin=[0, 0, hydro['waterline_z_frd_m']],
                                 plane_normal=[0, 0, 1])
        loops = waterline.discrete
        port_waterline = np.vstack([loop for loop in loops if loop[:, 1].mean() > 0])
        all_waterline = np.vstack(loops)
        wetted_dimensions = {
            'source_z_aft_mm': float(port_waterline[:, 0].min()*1000+915),
            'source_z_forward_mm': float(port_waterline[:, 0].max()*1000+915),
            'port_length_mm': float(np.ptp(port_waterline[:, 0])*1000),
            'port_width_mm': float(np.ptp(port_waterline[:, 1])*1000),
            'combined_beam_mm': float(np.ptp(all_waterline[:, 1])*1000),
            'pontoon_centerline_spacing_mm': float((port_waterline[:, 1].min()+
                                                     port_waterline[:, 1].max())*1000),
        }
        source_length, source_beam, source_spacing = 1745.0419971309304, 823.0786132619094, 676.0418111570184
        wetted_dimensions['length_error_percent'] = abs(wetted_dimensions['port_length_mm']/source_length-1)*100
        wetted_dimensions['beam_error_percent'] = abs(wetted_dimensions['combined_beam_mm']/source_beam-1)*100
        wetted_dimensions['spacing_error_mm'] = abs(wetted_dimensions['pontoon_centerline_spacing_mm']-source_spacing)
        wetted_dimensions['source_dimension_gate_pass'] = (wetted_dimensions['length_error_percent']<.5 and
            wetted_dimensions['beam_error_percent']<.5 and wetted_dimensions['spacing_error_mm']<2)
        row = {'stl_sha256': sha256(path.read_bytes()).hexdigest(),
               'watertight': bool(mesh.is_watertight),
               'winding_consistent': bool(mesh.is_winding_consistent),
               'component_count': len(mesh.split(only_watertight=False)),
               'mesh_volume_m3': float(mesh.volume),
               'mesh_center_frd_m': mesh.center_mass.tolist(),
               'bounds_frd_m': mesh.bounds.tolist(),
               'ordinary_all': stats(d[ordinary], areas[ordinary]),
               'ordinary_protected': stats(d[protected], areas[protected]),
               'declared_masks': {label: stats(d[mask], areas[mask])
                                  for label, mask in mask_arrays.items()},
               'hydrostatics_52p3kg': hydro, 'draft_m': draft,
               'wetted_pontoon_dimensions': wetted_dimensions}
        row['source_fidelity_pass'] = (row['ordinary_protected']['p95_mm'] < 2 and
                                      row['ordinary_protected']['max_mm'] < 5 and
                                      row['ordinary_protected']['area_above_5mm_m2'] == 0)
        rows['variants'][name] = row
        print(name, 'P95', row['ordinary_protected']['p95_mm'],
              'max', row['ordinary_protected']['max_mm'],
              'draft', draft, 'pass', row['source_fidelity_pass'], flush=True)
    (BASE / 'geometry_validation.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    run()
