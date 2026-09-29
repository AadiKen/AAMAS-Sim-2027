"""Validate frozen M1 contracts, passivity and variant spread for Surveyor v3."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from bcod_sim.vessel_generation.coefficient_package import (
    load_coefficient_package, reference_wrench, validate_coefficients,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'

STATES = {
    'surge_0p75_X_N': ([.75, 0, 0, 0, 0, 0], 0),
    'surge_1p5_X_N': ([1.5, 0, 0, 0, 0, 0], 0),
    'surge_3_X_N': ([3., 0, 0, 0, 0, 0], 0),
    'sway_0p3_Y_N': ([0, .3, 0, 0, 0, 0], 1),
    'sway_0p3_N_Nm': ([0, .3, 0, 0, 0, 0], 5),
    'yaw_0p3_Y_N': ([0, 0, 0, 0, 0, .3], 1),
    'yaw_0p3_N_Nm': ([0, 0, 0, 0, 0, .3], 5),
    'combined_Y_N': ([1., .3, 0, 0, 0, .1], 1),
    'combined_N_Nm': ([1., .3, 0, 0, 0, .1], 5),
}


def run() -> None:
    packages = {}
    validations = {}
    metrics = {}
    for name in ('nominal', 'fuller', 'finer'):
        path = BASE / 'coefficients' / name / 'coefficient_package.yaml'
        package = load_coefficient_package(path)
        result = validate_coefficients(path, grid_size=7)
        packages[name], validations[name] = package, result
        matrix = np.asarray(package['added_mass']['matrix_6x6'], dtype=float)
        row = {f'added_mass_{axis}': float(matrix[index, index]) for axis, index in
               (('surge',0),('sway',1),('heave',2),('roll',3),('pitch',4),('yaw',5))}
        for key, (state, component) in STATES.items():
            row[key] = float(reference_wrench(package, np.asarray(state))[component])
        row.update({'draft_m': float(package['reference']['draft_m']),
                    'waterplane_area_m2': float(package['hydrostatics']['waterplane_area_m2']),
                    'displaced_volume_m3': float(package['hydrostatics']['volume_m3'])})
        metrics[name] = row
        print(name, result, flush=True)
    rows = []
    for metric in metrics['nominal']:
        values = {name: item[metric] for name,item in metrics.items()}
        nominal = values['nominal']
        spread = (max(values.values())-min(values.values()))/max(abs(nominal),1e-9)*100
        rows.append({'metric':metric, **values, 'spread_percent_of_nominal':spread})
    frozen = json.loads((ROOT / 'docs/surveyor_cad_validation/m1_freeze_manifest.json').read_text())
    hashes = {path:sha256((ROOT/path).read_bytes()).hexdigest() for path in frozen['m1_file_sha256']}
    unchanged = hashes == frozen['m1_file_sha256']
    report = {'schema':'surveyor-v3-m1-validation-1',
              'frozen_m1_files_unchanged':unchanged,
              'frozen_m1_hashes':hashes,
              'package_validation':validations,
              'state_grid_metrics':metrics,
              'variant_spread':rows,
              'geometry_uncertainty_rule':'<10% secondary; 10–25% report band; >25% not clean physical validation'}
    (BASE/'m1_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not unchanged or not all(result['passed'] for result in validations.values()):
        raise RuntimeError('Frozen M1 hash or package validation failed')


if __name__ == '__main__':
    run()
