"""Frequency-match the frozen KCS package to HMRI's 0.065 Hz pure-sway PMM."""
from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import capytaine as cpt
import trimesh

from bcod_sim.vessel_generation.simple_models import _wetted_capytaine_panels


ROOT = Path("stage3_results/simple_hydrodynamics/hmri_coefficient_campaign")
PACKAGE = ROOT / "canonical"
FREQUENCY_HZ = .065  # Sung & Park (2015), Fig. 5(a), KCS pure sway.


def main() -> None:
    mesh = trimesh.load(PACKAGE / "hydrodynamic_mesh.stl", force="mesh")
    provenance = json.loads((PACKAGE / "provenance.json").read_text())
    waterline = json.loads((PACKAGE / "hydrostatics.json").read_text())["waterline_z_frd_m"]
    vertices, faces = _wetted_capytaine_panels(mesh, waterline)
    hull = cpt.Mesh(vertices=vertices, faces=faces)
    lid = hull.generate_lid(z=0.)
    body = cpt.FloatingBody(mesh=hull, lid_mesh=lid,
        dofs=cpt.rigid_body_dofs(rotation_center=(0., 0., 0.)))
    solver = cpt.BEMSolver()
    target = 2*math.pi*FREQUENCY_HZ
    rows = []
    for omega in (0., .01, .5*target, target, 2*target):
        start = perf_counter()
        result = solver.solve(cpt.RadiationProblem(body=body,
            radiating_dof="Sway", omega=omega, rho=1025.))
        rows.append({"omega_rad_s": omega, "frequency_hz": omega/(2*math.pi),
            "A22_kg": float(result.added_masses["Sway"]),
            "B22_rad_N_s_m": float(result.radiation_dampings["Sway"]),
            "time_s": perf_counter()-start})
    output = {"source_frequency_hz": FREQUENCY_HZ,
        "source": "Sung and Park (2015), Fig. 5(a), KCS pure-sway test",
        "frozen_geometry_sha256": provenance["original_geometry_sha256"],
        "frozen_hydrodynamic_mesh_sha256": provenance["hydrodynamic_mesh_sha256"],
        "wetted_panels": len(faces), "lid_panels": lid.nb_faces,
        "frequency_sweep": rows,
        "maneuvering_damping": "Radiation B22 recorded only; excluded from BCOD damping"}
    (ROOT / "frequency_sweep.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
