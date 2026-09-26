"""Export sectional and component loads from a generated passive vessel package.

This is a geometry/model diagnostic. It never reads benchmark measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from bcod_sim.dynamics.coriolis import coriolis_wrench
from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.dynamics.damping import Damping
from bcod_sim.state.vessel_state import VesselState


def diagnose(package: Path, states: list[tuple[str, float, float, float]]) -> tuple[list[dict], list[dict]]:
    stations = json.loads((package / "maneuvering.json").read_text())["crossflow"]["stations"]
    payload = json.loads((package / "runtime_payload.json").read_text())
    coefficients = json.loads((package / "coefficients.json").read_text())
    t = lambda value: torch.as_tensor(value, dtype=torch.float64)
    cf = SectionalCrossflow.from_stations(stations, density=payload["crossflow"]["water_density_kg_m3"])
    curve = payload["surge_resistance"]
    damping = Damping(t(payload["linear_damping"]), t(payload["quadratic_damping"]),
                      t(payload["linear_damping_matrix"]),
                      surge_resistance_curve=(t(curve["speed_mps"]), t(curve["force_x_n"])))
    added = t(coefficients["M_A"])
    detail, totals = [], []
    for name, u, v, r in states:
        nu = t([u, v, 0., 0., 0., r])
        state = VesselState(t([0., 0., 0.]), t([1., 0., 0., 0.]), nu)
        result = cf.evaluate(state)
        diagnostic = result.diagnostics
        linear, nonlinear = damping.components(nu)
        added_coriolis = -coriolis_wrench(added, nu)
        cumulative_y = cumulative_n = 0.
        for i, station in enumerate(stations):
            value = lambda key: float(diagnostic[key][i])
            row = {"state": name, "hull_id": station.get("hull_id", 0),
                   "x_m": station["x_m"], "y_m": station["y_m"],
                   "section_area_m2": station.get("submerged_section_area_m2"),
                   "beam_m": station["beam_m"], "depth_m": station["draft_m"],
                   "fullness": station.get("section_fullness"),
                   "u_local_mps": value("station_local_surge_mps"),
                   "v_local_mps": value("station_local_sway_mps"),
                   "beta_local_rad": value("station_local_incidence_rad"),
                   "cd": value("drag_coefficient"),
                   "transition_weight": value("station_crossflow_weight"),
                   "dY_linear_lift_n": value("section_lift_force_n"),
                   "dY_nonlinear_drag_n": value("section_drag_force_n"),
                   "dY_total_n": value("section_force_n"),
                   "dN_total_nm": value("section_yaw_moment_nm")}
            cumulative_y += row["dY_total_n"]
            cumulative_n += row["dN_total_nm"]
            row["cumulative_Y_n"] = cumulative_y
            row["cumulative_N_nm"] = cumulative_n
            detail.append(row)
        totals.append({"state": name, "u_mps": u, "v_mps": v, "r_rad_s": r,
                       "section_Y_n": float(result.tau_body[1]),
                       "section_N_nm": float(result.tau_body[5]),
                       "linear_viscous_wrench": linear.tolist(),
                       "nonlinear_resistance_wrench": nonlinear.tolist(),
                       "added_mass_coriolis_wrench": added_coriolis.tolist(),
                       "dissipative_power_w": float(nu @ (result.tau_body + linear + nonlinear))})
    return detail, totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state", nargs=4, action="append", metavar=("NAME", "U", "V", "R"), required=True)
    args = parser.parse_args()
    states = [(name, float(u), float(v), float(r)) for name, u, v, r in args.state]
    detail, totals = diagnose(args.package, states)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "sections.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    (args.output / "totals.json").write_text(json.dumps(totals, indent=2) + "\n")


if __name__ == "__main__":
    main()
