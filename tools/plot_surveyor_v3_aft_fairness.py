"""Section fairness plot across the Surveyor v3 aft-cap transition."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bcod_sim.vessel_generation.simple_sections import section_properties

BASE = ROOT / 'docs/surveyor_cad_validation/surrogate_cad'
OLD = BASE / 'v2_targeted_patch/meshes/surveyor_nominal_v2_bridge_refit.stl'
NEW = BASE / 'v3_fallback_repair/meshes/surveyor_nominal_v3_aft_repair.stl'
OUT = BASE / 'v3_fallback_repair/figures'


def run() -> None:
    meshes = {'v2': trimesh.load_mesh(OLD, process=True),
              'v3': trimesh.load_mesh(NEW, process=True)}
    stations = np.arange(10, 121, 5)
    results = {}
    for name, mesh in meshes.items():
        port = mesh.submesh([np.where(mesh.triangles_center[:, 1] > 0)[0]],
                            append=True, repair=False)
        rows = []
        for z in stations:
            prop = section_properties(port, 0, (z-915)/1000)
            loops = prop['loops']
            vertices = np.vstack(loops) if loops else np.zeros((0,3))
            length = sum(float(np.linalg.norm(np.diff(loop,axis=0),axis=1).sum()) for loop in loops)
            rows.append({'source_z_mm': float(z), 'area_m2': prop['area'],
                         'centroid_y_m': prop['centroid'][0],
                         'centroid_z_m': prop['centroid'][1],
                         'half_beam_m': float((vertices[:,1].max()-vertices[:,1].min())/2) if len(vertices) else 0.,
                         'depth_m': float(vertices[:,2].max()-vertices[:,2].min()) if len(vertices) else 0.,
                         'perimeter_m': length})
        results[name] = rows
        print(name, 'sections', len(rows), flush=True)
    (OUT / 'aft_section_fairness.json').write_text(json.dumps(results, indent=2)+'\n')
    keys = [('area_m2','Area (m²)'),('centroid_z_m','Vertical centroid FRD (m)'),
            ('half_beam_m','Half beam (m)'),('depth_m','Depth (m)'),
            ('perimeter_m','Perimeter (m)')]
    fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True)
    for ax,(key,label) in zip(axes.ravel(),keys):
        for name,rows in results.items():
            ax.plot(stations,[r[key] for r in rows],marker='.',label=name)
        ax.axvspan(5,95,color='grey',alpha=.12)
        ax.set_ylabel(label)
    axes.ravel()[0].legend()
    axes.ravel()[-1].axis('off')
    for ax in axes[-1,:]: ax.set_xlabel('Source longitudinal Z (mm)')
    fig.tight_layout()
    fig.savefig(OUT / 'aft_section_fairness.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    run()
