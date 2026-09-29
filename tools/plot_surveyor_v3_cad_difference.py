"""Sample a v2-to-v3 surface difference map; B-rep Boolean proof is separate."""
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
sys.path.insert(0, str(ROOT/'tools'))
from qualify_surveyor_envelopes import _surface_distances

BASE = ROOT/'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'
OLD = BASE/'provenance/v2_targeted_conformal_nominal.stl'
NEW = BASE/'meshes/surveyor_nominal_v3_aft_repair.stl'


def run() -> None:
    old = trimesh.load_mesh(OLD, process=True)
    new = trimesh.load_mesh(NEW, process=True)
    centers = old.triangles_center
    port = centers[:,1] > 0
    ids = np.where(port)[0][::8]
    points = centers[ids]
    distance = _surface_distances(points,new,128)*1000
    outside = (points[:,0]<-.910)|(points[:,0]>-.820)
    data = {'sample_count':int(len(points)), 'mask_frd_x_mm':[-910,-820],
            'outside_sample_count':int(outside.sum()),
            'outside_p95_mm':float(np.percentile(distance[outside],95)),
            'outside_max_mm':float(distance[outside].max()),
            'inside_max_mm':float(distance[~outside].max())}
    (BASE/'cad_difference_map.json').write_text(json.dumps(data,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(12,4.5))
    im=ax.scatter(points[:,0]*1000,points[:,1]*1000,c=np.clip(distance,0,10),s=2,
                  cmap='turbo',vmin=0,vmax=10,rasterized=True)
    ax.axvspan(-910,-820,color='black',alpha=.07)
    ax.set(xlabel='FRD longitudinal X (mm)',ylabel='Port lateral Y (mm)',
           title='v2-to-v3 CAD surface displacement (conformal STL sampling)')
    fig.colorbar(im,ax=ax,label='Nearest-surface distance (mm; clipped at 10)')
    fig.savefig(BASE/'figures/cad_difference_map.png',dpi=170,bbox_inches='tight')
    plt.close(fig)
    print(data,flush=True)


if __name__=='__main__':
    run()
