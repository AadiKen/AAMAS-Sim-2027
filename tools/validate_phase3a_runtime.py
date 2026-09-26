"""Exercise a generated package through the production Plant6 runtime."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import RestoringLUT
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.web.runtime_factory import VesselRuntime


def plant_from_package(path: Path) -> tuple[Plant6, torch.Tensor]:
    spec = VesselRuntime.model_validate_json((path / "runtime_payload.json").read_text())
    t = lambda value: torch.as_tensor(value, dtype=torch.float64)
    h = spec.hydrostatics
    hydro = RestoringLUT(t(h["axes"]["heave_m"]), t(h["axes"]["roll_rad"]),
                         t(h["axes"]["pitch_rad"]), t(h["wrench_frd"]))
    cf = SectionalCrossflow.from_stations(spec.crossflow["stations"],
                                           density=spec.crossflow["water_density_kg_m3"])
    curve = spec.surge_resistance
    damping = Damping(t(spec.linear_damping), t(spec.quadratic_damping),
                      t(spec.linear_damping_matrix),
                      surge_resistance_curve=(t(curve["speed_mps"]), t(curve["force_x_n"])))
    plant = Plant6(MassProperties(spec.mass_kg, t(spec.cg_frd_m), t(spec.inertia_cg_kg_m2),
                                 t(spec.added_mass_kg)), damping, hydro,
                   OperatingEnvelope(t(spec.max_abs_nu), spec.min_substep_s, spec.max_substep_s),
                   crossflow=cf)
    return plant, t(spec.added_mass_kg)


def run(path: Path, duration_s: float = 1.) -> dict:
    plant, added = plant_from_package(path)
    payload = json.loads((path / "runtime_payload.json").read_text())
    eq_heave, eq_roll, eq_pitch = payload["equilibrium_heave_roll_pitch"]
    t = lambda value: torch.as_tensor(value, dtype=torch.float64)
    zero = {name: t([0.]*6) for name in EXTERNAL_TERMS}
    cases = {
        "rest": ((0., 0., 0.), (0., 0., 0.), (0., 0., 0., 0., 0., 0.)),
        "surge_coast_down": ((0., 0., 0.), (0., 0., 0.), (.5, 0., 0., 0., 0., 0.)),
        "sway_decay": ((0., 0., 0.), (0., 0., 0.), (0., .2, 0., 0., 0., 0.)),
        "yaw_decay": ((0., 0., 0.), (0., 0., 0.), (0., 0., 0., 0., 0., .1)),
        "combined_horizontal": ((0., 0., 0.), (0., 0., 0.), (.5, .2, 0., 0., 0., .1)),
        "heave_decay": ((0., 0., .01), (0., 0., 0.), (0., 0., 0., 0., 0., 0.)),
        "roll_decay": ((0., 0., 0.), (.03, 0., 0.), (0., 0., 0., 0., 0., 0.)),
        "pitch_decay": ((0., 0., 0.), (0., .03, 0.), (0., 0., 0., 0., 0., 0.)),
    }
    results = {}
    dt = .01
    steps = round(duration_s/dt)
    for name, (position, angles, velocity) in cases.items():
        position = (position[0], position[1], eq_heave+position[2])
        roll, pitch, yaw = eq_roll+angles[0], eq_pitch+angles[1], angles[2]
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        q = t((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
               cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))
        state = VesselState(t(position), q, t(velocity))
        initial_speed = float(torch.linalg.vector_norm(state.nu_body))
        max_speed = initial_speed
        for _ in range(steps):
            state = plant.step(state, zero, dt).state
            speed = float(torch.linalg.vector_norm(state.nu_body))
            max_speed = max(max_speed, speed)
            if not torch.isfinite(state.nu_body).all() or not torch.isfinite(state.position_ned).all():
                raise ValueError(f"Non-finite Plant6 state in {name}")
        final_speed = float(torch.linalg.vector_norm(state.nu_body))
        if name == "rest" and max_speed > .01:
            raise ValueError(f"Plant6 accelerated materially from rest: {max_speed}")
        if max_speed > max(.5, initial_speed*3):
            raise ValueError(f"Plant6 response grew excessively in {name}: {max_speed}")
        results[name] = {"initial_speed_norm": initial_speed, "final_speed_norm": final_speed,
                         "maximum_speed_norm": max_speed,
                         "final_position_ned_m": state.position_ned.tolist(),
                         "final_velocity_body": state.nu_body.tolist(), "passed": True}
    return {"package": str(path), "duration_s": duration_s, "time_step_s": dt,
            "equilibrium_heave_roll_pitch": [eq_heave, eq_roll, eq_pitch],
            "cases": results, "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({name: {"final_speed_norm": row["final_speed_norm"], "passed": row["passed"]}
                      for name, row in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
