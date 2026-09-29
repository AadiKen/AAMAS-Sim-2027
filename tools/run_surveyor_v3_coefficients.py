"""Generate uncalibrated Surveyor v3 packages through the ordinary pipeline."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'


def run() -> None:
    outputs = {}
    for name in ('nominal', 'fuller', 'finer'):
        geometry = BASE / 'meshes' / f'surveyor_{name}_v3_aft_repair.stl'
        output = BASE / 'coefficients' / name
        result = generate_simple_vessel(geometry=geometry, output=output,
            mass_kg=52.3, cg_frd_m=(0., 0., 0.), units='m',
            water_density_kg_m3=1000.)
        provenance = json.loads((result / 'provenance.json').read_text())
        validation = json.loads((result / 'validation.json').read_text())
        outputs[name] = {'geometry_sha256': sha256(geometry.read_bytes()).hexdigest(),
                         'package_sha256': sha256((result / 'coefficient_package.yaml').read_bytes()).hexdigest(),
                         'generation_request_sha256': provenance['generation_request_sha256'],
                         'validation': validation}
        print(name, 'generated', validation, flush=True)
    manifest = {'schema': 'surveyor-v3-production-coefficients-1',
                'source_label': 'SeaRobotics Surveyor-derived CAD — geometry-repair fallback',
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                      text=True).strip(),
                'mass_kg': 52.3, 'cg_frd_m': [0, 0, 0], 'water_density_kg_m3': 1000.,
                'pipeline': 'bcod_sim.vessel_generation.simple_pipeline.generate_simple_vessel',
                'disable_bem': False, 'calibration': 'none', 'variants': outputs}
    (BASE / 'coefficient_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    run()
