"""Report the sole +6 degree HMRI run without comparing transient loads."""
from __future__ import annotations

import json
import math
import re

from bcod_sim.vessel_generation.hmri_wrench import PhysicalFluidWrench, hmri_y_n_error

from monitor_kcs_hmri_plus6 import LOG, ROOT, force_history, snapshot


def main() -> None:
    time, physical, resisting = force_history()
    if not len(time):
        raise SystemExit("no force samples")
    observed, qualified = snapshot(time, physical, resisting, LOG.read_text())
    start = observed["window_start_s"]
    active = time >= start
    physical_mean = physical[active].mean(axis=0)
    resisting_mean = resisting[active].mean(axis=0)

    coefficients = {
        row["name"]: row["estimate"]
        for row in json.loads((ROOT.parent / "experimental_coefficients.json").read_text())["coefficients"]
    }
    v_prime = -math.sin(math.radians(6))
    force_scale = 0.5 * 1000 * 5.75 * 0.270 * 1.953**2
    moment_scale = force_scale * 5.75
    target_y = force_scale * (
        coefficients["Y_v_prime"] * v_prime
        + coefficients["Y_vvv_prime"] * v_prime**3
    )
    target_n = moment_scale * (
        coefficients["N_v_prime"] * v_prime
        + coefficients["N_vvv_prime"] * v_prime**3
    )

    comparison = {"status": "withheld_not_stationary", "hmri_Y_N": [target_y, target_n]}
    if qualified:
        load = PhysicalFluidWrench(tuple(physical_mean[:3]), tuple(physical_mean[3:]))
        error = hmri_y_n_error(load, target_y_n=target_y, target_n_nm=target_n)
        comparison = {"status": "qualified", "hmri_Y_N": [target_y, target_n],
                      "cfd_physical_Y_N": [load.force_frd_n[1], load.moment_frd_nm[2]],
                      "error_Y_N": error}

    clock = re.findall(r"ClockTime = (\d+) s", LOG.read_text())
    result = {
        "status": "PASS" if qualified else "FAIL_no_stationary_X_Y_N_window",
        "beta_deg": 6,
        "simulation_end_s": float(time[-1]),
        "wall_time_s": int(clock[-1]) if clock else None,
        "cells_old_new": [68064, 168808],
        "wall_layers": {"hull_face_coverage": 0.937076, "actual_layer_cells": 54886,
                        "requested_layer_cells": 73326, "first_layer_m": 0.002},
        "mesh_check": "PASS",
        "ramp": {"start_s": 0, "end_s": 1, "type": "smoothstep inlet plus density-weighted bulk acceleration"},
        "stationarity": observed,
        "physical_wrench": {"representation": "fluid_on_hull_FRD", "qualified": qualified,
                            "diagnostic_tail_mean": physical_mean.tolist()},
        "bcod_resisting_wrench": {"representation": "resisting_FRD", "qualified": qualified,
                                  "diagnostic_tail_mean": resisting_mean.tolist()},
        "hmri_comparison": comparison,
    }
    (ROOT / "single_plus6_result.json").write_text(json.dumps(result, indent=2) + "\n")
    meta_path = ROOT / "case_metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata["status"] = result["status"]
    metadata["actual_end_time_s"] = result["simulation_end_s"]
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(ROOT / "single_plus6_result.json")


if __name__ == "__main__":
    main()
