"""Build and audit the Surveyor-constrained hydrodynamic surrogate.

The source chains and their pocket hypotheses are frozen inputs from the
section-recovery study. STEP is the authoritative loft; STL is its tessellation.
This script deliberately stops before M1 when any required CAD gate fails.
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bcod_sim.vessel_generation.simple_sections import solve_waterline
from qualify_surveyor_envelopes import _sample_source, _surface_distances

SOURCE_STEP = Path.home() / 'Downloads' / 'Mock_Surveyor_1A (1).STEP'
SOURCE_STL = ROOT / 'docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/source/mixed_low_faces_source_mm.stl'
INPUT = ROOT / 'docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/chain_hypotheses'
OUT = ROOT / 'docs/surveyor_cad_validation/surrogate_cad'
VARIANTS = {'nominal': 'smooth', 'fuller': 'outset', 'finer': 'inset'}
STATION_INDICES = slice(None, None, 2)  # 10 mm, including both end stations
ARC_COUNT = 8
POINTS_PER_ARC = 32


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def loft(variant: str, contours: np.ndarray, step_path: Path) -> float:
    """Ruled OCC solid through eight B-spline arcs per closed transverse wire."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add(f'surveyor_{variant}')
        wires = []
        for source_z, contour in contours:
            ring = contour[::2]
            point_tags = [gmsh.model.occ.addPoint(source_z-915, -x, -y)
                          for x, y in ring]
            curves = [gmsh.model.occ.addBSpline(
                [point_tags[(j*POINTS_PER_ARC+k) % len(ring)]
                 for k in range(POINTS_PER_ARC+1)])
                for j in range(ARC_COUNT)]
            wires.append(gmsh.model.occ.addWire(curves, checkClosed=True))
        dimtags = gmsh.model.occ.addThruSections(wires, makeSolid=True, makeRuled=True)
        solids = [tag for dim, tag in dimtags if dim == 3]
        if len(solids) != 1:
            raise RuntimeError(f'Expected one port solid, got {solids}')
        port = solids[0]
        starboard = gmsh.model.occ.copy([(3, port)])
        gmsh.model.occ.mirror(starboard, 0, 1, 0, 0)
        gmsh.model.occ.synchronize()
        mass = sum(gmsh.model.occ.getMass(3, tag) for tag in (port, starboard[0][1])) / 1e9
        if mass <= 0:
            raise RuntimeError('Nonpositive B-rep mass')
        gmsh.write(str(step_path))
        return mass
    finally:
        gmsh.finalize()


def tessellate(step_path: Path, output: Path, minimum: float, maximum: float) -> trimesh.Trimesh:
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add(output.stem)
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMin', minimum)
        gmsh.option.setNumber('Mesh.MeshSizeMax', maximum)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(2)
        raw = output.with_suffix('.mm.stl')
        gmsh.write(str(raw))
        mesh = trimesh.load_mesh(raw, process=True)
        mesh.vertices /= 1000
        mesh.fix_normals()
        if mesh.volume < 0:
            mesh.invert()
        mesh.export(output)
        raw.unlink()
        return mesh
    finally:
        gmsh.finalize()


def mesh_check(mesh: trimesh.Trimesh, brep_volume: float) -> dict:
    faces = np.sort(mesh.faces, axis=1)
    duplicate = int(len(faces)-len(np.unique(faces, axis=0)))
    components = mesh.split(only_watertight=False)
    return {'watertight': bool(mesh.is_watertight),
            'winding_consistent': bool(mesh.is_winding_consistent),
            'component_count': len(components), 'duplicate_faces': duplicate,
            'volume_m3': float(mesh.volume), 'brep_volume_m3': brep_volume,
            'brep_mesh_volume_relative_difference': float(abs(mesh.volume-brep_volume)/brep_volume),
            'bounds_frd_m': mesh.bounds.tolist(), 'face_count': len(mesh.faces),
            'pass': bool(mesh.is_watertight and mesh.is_winding_consistent and
                         len(components) == 2 and duplicate == 0 and mesh.volume > 0)}


def source_check(mesh: trimesh.Trimesh, points: np.ndarray, waterline: float) -> dict:
    distances = _surface_distances(points, mesh)
    source_z = points[:, 0]*1000+915
    outside = ((source_z < 900)|(source_z > 1200)) & ((source_z < 325)|(source_z > 350))
    assessed = distances[outside & (points[:, 2] >= waterline)]
    return {'assessed_points': len(assessed),
            'median_mm': float(np.median(assessed)*1000),
            'p95_mm': float(np.percentile(assessed, 95)*1000),
            'max_mm': float(assessed.max()*1000),
            'pass': bool(np.percentile(assessed, 95) < .002 and assessed.max() < .005),
            'excluded_source_Z_mm': [[325, 350], [900, 1200]]}


def section_table(name: str, stations: np.ndarray, contour: np.ndarray) -> None:
    path = OUT / 'sections' / f'{name}.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_Z_mm', 'frd_x_m', 'area_mm2', 'centroid_source_X_mm',
                         'centroid_source_Y_mm', 'max_half_beam_mm', 'draft_extent_mm',
                         'perimeter_mm', 'provenance'])
        for z, ring in zip(stations, contour):
            a = ring[:, 0]*np.roll(ring[:, 1], -1)-np.roll(ring[:, 0], -1)*ring[:, 1]
            area = abs(a.sum())/2
            perimeter = np.linalg.norm(np.roll(ring, -1, axis=0)-ring, axis=1).sum()
            writer.writerow([z, (z-915)/1000, area, ring[:, 0].mean(), ring[:, 1].mean(),
                             abs(ring[:, 0]).max(), -ring[:, 1].min(), perimeter,
                             'SOURCE_STEP+INTERPOLATED+POCKET_SURROGATE' if 900 <= z <= 1200
                             else 'SOURCE_STEP+INTERPOLATED'])


def run() -> dict:
    for folder in ('source', 'reconstruction', 'validation', 'exports', 'cad', 'meshes',
                   'hydrostatics', 'sections', 'source_comparison', 'renders', 'm1', 'comparison'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    imported = trimesh.load_mesh(SOURCE_STL, process=True)
    port = next(part for part in imported.split(only_watertight=False)
                if part.bounds[1, 0] < 0)
    source_points = _sample_source(port, 50, 1750)
    result = {'schema': 'surveyor-constrained-hydrodynamic-surrogate-v1',
              'source_step_sha256': digest(SOURCE_STEP),
              'source_section_mesh_sha256': digest(SOURCE_STL),
              'source_units': 'inch AP214; Gmsh import in mm',
              'cad_units': 'mm', 'stl_units': 'm',
              'transform': 'FRD x=(source Z-915)/1000; y=-source X/1000; z=-source Y/1000',
              'authoritative_representation': 'OCC B-rep STEP; STL is triangulation',
              'script_sha256': digest(Path(__file__)), 'water_density_kg_m3': 1000,
              'mass_kg': 52.3, 'variants': {}, 'm1_run': False}
    for name, hypothesis in VARIANTS.items():
        input_path = INPUT / f'{hypothesis}_sections.npz'
        data = np.load(input_path)
        stations = data['stations_mm'][STATION_INDICES]
        rings = data['port_contours_source_xy_mm'][STATION_INDICES]
        section_table(name, stations, rings)
        combined = [(float(z), ring) for z, ring in zip(stations, rings)]
        step_path = OUT / 'cad' / f'surveyor_{name}.step'
        brep_volume = loft(name, combined, step_path)
        mesh_path = OUT / 'meshes' / f'{name}.stl'
        mesh = tessellate(step_path, mesh_path, 5, 10)
        hydro = solve_waterline(mesh, 52.3, 1000)
        draft = float(mesh.bounds[1, 2]-hydro['waterline_z_frd_m'])
        stat = mesh_check(mesh, brep_volume)
        source = source_check(mesh, source_points, hydro['waterline_z_frd_m'])
        # Published 1.83 m x 0.91 m describes the whole craft. Compare the
        # hydrodynamic solids only with the trusted, submerged STEP pontoon.
        source_wet = port.vertices[port.vertices[:, 1] <= -1000*hydro['waterline_z_frd_m']]
        surrogate_wet = mesh.vertices[(mesh.vertices[:, 1] > 0) &
                                      (mesh.vertices[:, 2] >= hydro['waterline_z_frd_m'])]
        source_length = float(np.ptp(source_wet[:, 2])/1000)
        surrogate_length = float(np.ptp(surrogate_wet[:, 0]))
        source_beam = float(-2*source_wet[:, 0].min()/1000)
        surrogate_beam = float(2*surrogate_wet[:, 1].max())
        source_spacing = float(-(source_wet[:, 0].min()+source_wet[:, 0].max())/1000)
        surrogate_spacing = float(surrogate_wet[:, 1].min()+surrogate_wet[:, 1].max())
        dimension = {'source_pontoon_length_m': source_length,
                     'surrogate_pontoon_length_m': surrogate_length,
                     'source_symmetric_beam_m': source_beam,
                     'surrogate_symmetric_beam_m': surrogate_beam,
                     'source_centerline_spacing_m': source_spacing,
                     'surrogate_centerline_spacing_m': surrogate_spacing,
                     'length_error_fraction': abs(surrogate_length-source_length)/source_length,
                     'beam_error_fraction': abs(surrogate_beam-source_beam)/source_beam,
                     'spacing_error_m': abs(surrogate_spacing-source_spacing),
                     'published_whole_vehicle_sanity_m': {'length': 1.83, 'beam': .91}}
        row = {'input_section_sha256': digest(input_path), 'step_sha256': digest(step_path),
               'stl_sha256': digest(mesh_path), 'mesh': stat, 'hydrostatics': hydro,
               'draft_m': draft, 'draft_gate_pass': bool(.16 <= draft <= .18),
               'source_pontoon_dimensions': dimension, 'source_pontoon_dimensions_gate_pass': bool(
                   dimension['length_error_fraction'] < .005 and
                   dimension['beam_error_fraction'] < .005 and
                   dimension['spacing_error_m'] < .002),
               'source_deviation': source}
        result['variants'][name] = row
        print(name, 'BREP/mesh', round(brep_volume, 5), round(mesh.volume, 5),
              'draft', round(draft, 5), 'source p95/max',
              round(source['p95_mm'], 2), round(source['max_mm'], 2), flush=True)
    nominal_step = OUT / 'cad/surveyor_nominal.step'
    coarse = tessellate(nominal_step, OUT / 'meshes/nominal_coarse.stl', 10, 20)
    fine = tessellate(nominal_step, OUT / 'meshes/nominal_fine.stl', 2.5, 5)
    nominal = trimesh.load_mesh(OUT / 'meshes/nominal.stl', process=True)
    nominal_hydro = result['variants']['nominal']['hydrostatics']
    fine_hydro = solve_waterline(fine, 52.3, 1000)
    convergence = {'coarse_volume_m3': float(coarse.volume),
                   'nominal_volume_m3': float(nominal.volume),
                   'fine_volume_m3': float(fine.volume),
                   'nominal_to_fine_volume_relative': float(abs(nominal.volume-fine.volume)/fine.volume),
                   'nominal_to_fine_cb_m': float(np.linalg.norm(
                       np.array(nominal_hydro['center_buoyancy_frd_m'])-
                       np.array(fine_hydro['center_buoyancy_frd_m']))),
                   'nominal_to_fine_waterplane_relative': float(abs(
                       nominal_hydro['waterplane_area_m2']-fine_hydro['waterplane_area_m2'])/
                       fine_hydro['waterplane_area_m2'])}
    convergence['pass'] = bool(convergence['nominal_to_fine_volume_relative'] < .005 and
                               convergence['nominal_to_fine_cb_m'] < .001 and
                               convergence['nominal_to_fine_waterplane_relative'] < .01)
    result['mesh_convergence'] = convergence
    all_gates = all(row['mesh']['pass'] and row['draft_gate_pass'] and
                    row['source_pontoon_dimensions_gate_pass'] and row['source_deviation']['pass']
                    for row in result['variants'].values()) and convergence['pass']
    result['cad_validation_pass'] = all_gates
    (OUT / 'provenance_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUT / 'hydrostatics/comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'volume_m3', 'draft_m', 'waterline_z_frd_m',
                         'cb_x_m', 'cb_y_m', 'cb_z_m', 'waterplane_area_m2',
                         'waterplane_I_roll_m4', 'waterplane_I_pitch_m4'])
        for name, row in result['variants'].items():
            h = row['hydrostatics']
            writer.writerow([name, h['volume_m3'], row['draft_m'], h['waterline_z_frd_m'],
                             *h['center_buoyancy_frd_m'], h['waterplane_area_m2'],
                             h['waterplane_i_roll_m4'], h['waterplane_i_pitch_m4']])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, color in [('nominal', 'black'), ('fuller', 'green'), ('finer', 'orange')]:
        rows = np.genfromtxt(OUT / 'sections' / f'{name}.csv', delimiter=',', names=True,
                             dtype=None, encoding=None)
        axes[0].plot(rows['frd_x_m'], rows['area_mm2']/1e6, color=color, label=name)
        axes[1].plot(rows['frd_x_m'], rows['perimeter_mm']/1000, color=color, label=name)
    axes[0].set(xlabel='FRD x (m)', ylabel='Pontoon section area (m²)')
    axes[1].set(xlabel='FRD x (m)', ylabel='Pontoon perimeter (m)')
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'renders/section_fairness.png', dpi=160); plt.close(fig)
    (OUT / 'comparison/decision_manifest.json').write_text(json.dumps({
        'cad_validation_pass': all_gates, 'm1_run': False,
        'reason': 'M1 withheld until every required CAD validation gate passes' if not all_gates
                  else 'CAD gates pass; M1 not yet executed'}, indent=2) + '\n')
    return result


if __name__ == '__main__':
    run()
