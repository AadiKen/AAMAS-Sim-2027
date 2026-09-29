"""Plot original STEP source-face distance to pre/post aft-repair meshes."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from qualify_surveyor_envelopes import _surface_distances, _to_frd

SRC = ROOT / 'docs/surveyor_cad_validation/geometry_recovery/section_reconstruction/source/mixed_low_faces_source_mm.stl'
OLD = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair/provenance/v2_targeted_conformal_nominal.stl'
NEW = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair/meshes/surveyor_nominal_v3_aft_repair.stl'
OUT = NEW.parents[1] / 'figures'


def run() -> None:
    OUT.mkdir(exist_ok=True)
    source = trimesh.load_mesh(SRC, process=True)
    port = source.submesh([np.where(source.triangles_center[:, 0] < 0)[0]], append=True, repair=False)
    points = port.triangles_center
    wet = points[:, 1] <= -75
    ordinary = wet & ((points[:, 2] < 900) | (points[:, 2] > 1200))
    points = points[ordinary]
    meshes = [trimesh.load_mesh(path, process=True) for path in (OLD, NEW)]
    distances = [_surface_distances(_to_frd(points), mesh, 128)*1000 for mesh in meshes]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, distance, title in zip(axes, distances, ('v2 target, conformal remesh', 'v3 aft-cap repair')):
        im = ax.scatter(points[:, 2], points[:, 0], c=np.clip(distance, 0, 10),
                        s=2, cmap='turbo', vmin=0, vmax=10, rasterized=True)
        ax.axvspan(5, 95, color='black', alpha=.08, label='v3 aft mask')
        ax.set(ylabel='Source lateral X (mm)', title=title)
    axes[-1].set_xlabel('Source longitudinal Z (mm)')
    fig.colorbar(im, ax=axes, label='Source-to-STL distance (mm; clipped at 10)')
    fig.savefig(OUT / 'source_distance_before_after.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    run()
