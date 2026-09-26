"""Prepare the sole ramped HMRI +6 degree case on the qualified layer mesh."""
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
mesh = root / "mesh_wall_layers_v2"
out = root / "static_drift_plus6_wall_ramped"
if out.exists():
    raise SystemExit(f"preserving existing case: {out}")
if "Mesh OK." not in (mesh / "checkMesh.log").read_text():
    raise SystemExit("wall-layer mesh has not passed checkMesh")

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
    solver_settings=SimpleNamespace(end_time_s=6.5, initial_timestep_s=0.001,
                                    timestep_s=0.005, max_courant=0.25,
                                    max_alpha_courant=0.15, write_interval_s=0.5),
    reference_point_frd_m=(-0.0851, 0., 0.), waterline_z_m=0.,
)
files = OpenFOAMAdapter._free_surface_files(case, inlet, include_hull=True)
for name in ("system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict"):
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mesh / name, path)
for name in ("constant/polyMesh", "constant/triSurface"):
    shutil.copytree(mesh / name, out / name)
for name, value in files.items():
    if name != "system/blockMeshDict":
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

def vector(value):
    return "(" + " ".join(f"{x:.12g}" for x in value) + ")"

# Accelerate the entire water/air domain using a density-weighted body force.
# Ramping only the remote inlet leaves the hull unforced for several seconds.
ramp_end = 1.0
samples = []
for i in range(21):
    t = ramp_end * i / 20
    x = t / ramp_end
    factor = 3*x*x - 2*x*x*x
    samples.append(f"({t:.3f} {vector([factor*u for u in inlet])})")
samples.append(f"(1000 {vector(inlet)})")
table = "table (" + " ".join(samples) + ")"
velocity_path = out / "0/U"
velocity_text = velocity_path.read_text()
full = f"({inlet[0]} {inlet[1]} {inlet[2]})"
for patch in ("inletWater", "inletAir"):
    old = f"{patch} {{type fixedValue; value uniform {full};}}"
    assert old in velocity_text
    velocity_text = velocity_text.replace(
        old, f"{patch} {{type uniformFixedValue; uniformValue {table}; value uniform (0 0 0);}}",
    )
velocity_text = velocity_text.replace(f"internalField uniform {full};",
                                      "internalField uniform (0 0 0);")
velocity_path.write_text(velocity_text)
(out / "constant/fvModels").write_text(
    'FoamFile { version 2.0; format ascii; class dictionary; location "constant"; object fvModels; }\n'
    "momentumRamp\n{\n"
    "    type coded;\n    select all;\n    field U;\n"
    "    codeAddRhoSup\n    #{\n"
    "        const scalar t = mesh().time().value();\n"
    f"        const scalar T = {ramp_end};\n"
    "        if (t > 0 && t < T)\n        {\n"
    "            const scalar x = t/T;\n"
    "            const scalar rate = 6*x*(1-x)/T;\n"
    f"            const vector target({inlet[0]:.12g}, {inlet[1]:.12g}, {inlet[2]:.12g});\n"
    "            vectorField& source = eqn.source();\n"
    "            const scalarField& volume = mesh().V();\n"
    "            const scalarField& density = rho.primitiveField();\n"
    "            forAll(source, celli)\n"
    "                source[celli] -= volume[celli]*density[celli]*rate*target;\n"
    "        }\n"
    "    #};\n}\n"
)
solution_path = out / "system/fvSolution"
solution = solution_path.read_text()
assert "momentumPredictor no;" in solution
solution = solution.replace("momentumPredictor no;", "momentumPredictor yes;")
solution = solution.replace(
    "U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.1;}",
    "U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.1;} "
    "UFinal {$U; relTol 0;}",
)
solution_path.write_text(solution)

control_path = out / "system/controlDict"
control = control_path.read_text().replace("runTimeModifiable false;", "runTimeModifiable true;")
control_path.write_text(control)
(out / "case_metadata.json").write_text(json.dumps({
    "configuration_id": "KCS-HMRI-T579-2015-bare-hull-1to40-Fn0.26",
    "case": "static_drift_plus6_wall_ramped", "beta_deg": 6,
    "body_velocity_frd_mps": body_velocity, "target_inlet_foam_mps": inlet,
    "startup": {"type": "smoothstep inlet and density-weighted uniform momentum acceleration", "ramp_end_s": ramp_end, "source_after_ramp": 0},
    "solver_frame": "X forward, Y port, Z up",
    "frame_contract": "bcod-openfoam-frd-v1",
    "moment_reference_frd_m": case.reference_point_frd_m,
    "turbulence": "kOmegaSST with kqR/omega/nutk wall functions",
    "wall_mesh": "../mesh_wall_layers_v2", "water_density_kg_m3": 1000,
    "max_simulation_time_guard_s": 6.5,
    "comparison_wrench": "physical_fluid_on_hull",
    "bcod_fitting_wrench": "resisting, opposite sign",
    "status": "prepared_not_run",
}, indent=2) + "\n")
(out / "system/setFieldsDict").write_text(
    'FoamFile { format ascii; class dictionary; location "system"; object setFieldsDict; }\n'
    'defaultFieldValues (volScalarFieldValue alpha.water 0);\n'
    'regions (boxToCell { box (-15.1 -8.1 -5.1) (20.1 8.1 0); '
    'fieldValues (volScalarFieldValue alpha.water 1); });\n'
)
print(out)
