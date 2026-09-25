"""Prepare one fixed-hull HMRI +6-degree VOF/SST case from the checked mesh."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.frame_contract import body_velocity_to_fixed_hull_inlet
from bcod_sim.vessel_generation.identification import TurbulenceModel

root = Path("stage3_results/kcs-validation/track_b_hmri")
mesh = root / "mesh_smoke"
case_root = root / "static_drift_plus6"
if case_root.exists():
    raise SystemExit(f"case already exists; preserving prior results: {case_root}")

speed = 1.953
beta = math.radians(6)
body_velocity = (speed * math.cos(beta), -speed * math.sin(beta), 0.0)
inlet = body_velocity_to_fixed_hull_inlet(body_velocity)
nu = speed * 5.75 / 9.851e6
k = 1.5 * (0.01 * speed)**2
omega = math.sqrt(k) / (0.09**0.25 * 0.07 * 5.75)
case = SimpleNamespace(
    domain_settings=SimpleNamespace(minimum_frd_m=(-15., -8., -5.), maximum_frd_m=(20., 8., 4.)),
    mesh_settings=SimpleNamespace(base_cell_size_m=0.5),
    water_properties=SimpleNamespace(density_kg_m3=1000., kinematic_viscosity_m2_s=nu, gravity_mps2=9.81),
    turbulence_settings=SimpleNamespace(model=TurbulenceModel.K_OMEGA_SST,
                                        inlet_k_m2_s2=k, inlet_omega_s_inv=omega,
                                        inlet_nut_m2_s=1e-5),
    solver_settings=SimpleNamespace(end_time_s=3.0, initial_timestep_s=0.001,
                                    timestep_s=0.005, max_courant=0.4,
                                    max_alpha_courant=0.25, write_interval_s=0.5),
    reference_point_frd_m=(-0.0851, 0., 0.), waterline_z_m=0.,
)
files = OpenFOAMAdapter._free_surface_files(case, inlet, include_hull=True)
for source in ("system/blockMeshDict", "system/snappyHexMeshDict"):
    target = case_root / source
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mesh / source, target)
for source in ("constant/polyMesh", "constant/triSurface"):
    shutil.copytree(mesh / source, case_root / source)
for name, content in files.items():
    if name == "system/blockMeshDict":
        continue
    target = case_root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
(case_root / "case_metadata.json").write_text(json.dumps({
    "configuration_id": "KCS-HMRI-T579-2015-bare-hull-1to40-Fn0.26",
    "case": "static_drift_plus6", "beta_deg": 6,
    "speed_mps": speed, "Re": 9.851e6, "nu_m2_s": nu,
    "body_velocity_frd_mps": body_velocity, "inlet_velocity_foam_mps": inlet,
    "solver_frame": "X forward, Y port, Z up",
    "body_frame": "bcod-openfoam-frd-v1",
    "moment_reference_frd_m": case.reference_point_frd_m,
    "moment_reference_note": "-1.48% Lpp, inferred from KCS buoyancy/LCG, not independently reported for HMRI T579",
    "turbulence": "kOmegaSST, 1% inlet intensity, 0.07 Lpp length scale; HMRI paper used Realizable k-epsilon",
    "water_density_kg_m3": 1000,
    "status": "prepared_not_run",
}, indent=2) + "\n")
print(case_root)
