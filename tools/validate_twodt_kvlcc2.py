"""Independent frozen Level A static-drift comparison with published KVLCC2."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import trimesh

from bcod_sim.vessel_generation.simple_crossflow import extract_crossflow_stations
from bcod_sim.vessel_generation.twodt.level_a import SectionalTwoDt, sectional_v5_arrays

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'stage3_results/twodt'
GEOM=OUT/'kvlcc2_independent/kvlcc2_underwater.stl'
SOURCE='https://www.tandfonline.com/doi/full/10.1080/20464177.2019.1680076'
EFD={0:(-.00004,-.00006),3:(.01263,.00612),6:(.02560,.01392),
     9:(.04553,.01940),12:(.07082,.02539)}
L,T,U,RHO=5.5172,.3586,1.047,1025.


def main():
    freeze=json.loads((OUT/'level_a_freeze.json').read_text())
    for rel,expected in freeze['code_sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Level A code changed after pre-KCS freeze: '+rel)
    mesh=trimesh.load(GEOM,force='mesh')
    stations,provenance=extract_crossflow_stations(mesh,0.,count=31,
                       max_froude=U/math.sqrt(9.80665*L),density=RHO)
    model=SectionalTwoDt(stations,density=RHO)
    force_scale=.5*RHO*L*T*U**2
    rows=[]
    for beta,(yref,nref) in EFD.items():
        u=U*math.cos(math.radians(beta));v=-U*math.sin(math.radians(beta))
        base=sectional_v5_arrays(stations,u,v,0.)
        candidate=model.steady_captive(u,v,0.)
        row={'beta_deg':beta,'EFD_CY':yref,'EFD_CN':nref,
             'V5_CY':base['Y_n']/force_scale,'V5_CN':base['N_nm']/(force_scale*L),
             'level_A_CY':candidate['Y_n']/force_scale,
             'level_A_CN':candidate['N_nm']/(force_scale*L),
             'max_t_star':float(max(candidate['t_star']))}
        if beta:
            for name in ('V5','level_A'):
                row[name+'_Y_error_percent']=100*abs(row[name+'_CY']-yref)/abs(yref)
                row[name+'_N_error_percent']=100*abs(row[name+'_CN']-nref)/abs(nref)
        rows.append(row)
    positives=rows[1:]
    summary={key:float(np.median([row[key] for row in positives])) for key in
             ('V5_Y_error_percent','V5_N_error_percent',
              'level_A_Y_error_percent','level_A_N_error_percent')}
    output={'status':'INDEPENDENT_STATIC_DRIFT_TEST','reference':SOURCE,
            'reference_description':'Yang, Yin & Lian (2021), Tables 5-6, MOERI experimental CY/CN; their 4.97 m geometry is geometrically similar to this official 5.5172 m NMRI model',
            'normalization':'CY=Y/(0.5 rho U^2 L T), CN=N/(0.5 rho U^2 L^2 T)',
            'geometry':json.loads((OUT/'kvlcc2_independent/geometry.json').read_text()),
            'station_count':len(stations),'section_provenance':provenance,
            'conditions':rows,'summary':summary,
            'limitations':['reference scale differs; nondimensional geometry assumed similar',
                           'bare hull only, zero yaw',
                           'source uses an external MOERI EFD table; uncertainty varies by condition']}
    (OUT/'level_a_independent_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
