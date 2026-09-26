"""Deeply submerged sphere: full six-DOF Capytaine convergence check."""
import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault('CAPYTAINE_CACHE_DIR', '/private/tmp/bcod-capy-cache')
import capytaine as cpt
import numpy as np
import trimesh

ROOT = Path('stage3_results/simple_hydrodynamics/bem_qualification')
ROOT.mkdir(parents=True, exist_ok=True)
RHO = 1025.
REFERENCE = 2 * math.pi * RHO / 3
NAMES = ('Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw')
results = []
for subdivision in (2, 3, 4):
    source = trimesh.creation.icosphere(subdivisions=subdivision, radius=1.)
    source.apply_translation((0., 0., -5.))
    faces = np.column_stack((source.faces, source.faces[:, 2]))
    mesh = cpt.Mesh(vertices=source.vertices, faces=faces)
    body = cpt.FloatingBody(mesh=mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0., 0., -5.)))
    solver = cpt.BEMSolver()
    start = perf_counter()
    solutions = solver.solve_all([cpt.RadiationProblem(body=body, radiating_dof=name,
                                  omega=1., rho=RHO, free_surface=np.inf) for name in NAMES],
                                  progress_bar=False)
    duration = perf_counter() - start
    matrix = np.array([[float(s.added_masses[influenced]) for s in solutions]
                       for influenced in NAMES])
    damping = np.array([[float(s.radiation_dampings[influenced]) for s in solutions]
                        for influenced in NAMES])
    symmetry = np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix)
    row = {'subdivision': subdivision, 'panels': len(source.faces),
           'surge_added_mass_kg': float(matrix[0, 0]), 'sway_added_mass_kg': float(matrix[1, 1]),
           'heave_added_mass_kg': float(matrix[2, 2]), 'analytical_kg': REFERENCE,
           'surge_error_percent': float(100 * (matrix[0, 0] / REFERENCE - 1)),
           'symmetry_relative': float(symmetry),
           'max_translation_coupling_kg': float(np.max(np.abs(matrix[:3, :3] - np.diag(np.diag(matrix[:3, :3]))))),
           'duration_s': duration, 'added_mass_6x6': matrix.tolist(),
           'radiation_damping_6x6': damping.tolist()}
    results.append(row)
    (ROOT / f'sphere_{len(source.faces)}.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps({k: v for k, v in row.items() if k not in ('added_mass_6x6', 'radiation_damping_6x6')}), flush=True)
(ROOT / 'sphere_convergence.json').write_text(json.dumps(results, indent=2) + '\n')
