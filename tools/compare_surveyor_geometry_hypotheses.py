"""Compare frozen M1 outputs from three Surveyor envelope hypotheses."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from bcod_sim.vessel_generation.coefficient_package import (load_coefficient_package,
                                                              reference_wrench)


STATES = {
    "surge_0p75_mps_X_N": [0.75, 0, 0, 0, 0, 0],
    "surge_1p5_mps_X_N": [1.5, 0, 0, 0, 0, 0],
    "surge_3_mps_X_N": [3., 0, 0, 0, 0, 0],
    "pure_sway_0p3_mps_Y_N": [0, .3, 0, 0, 0, 0],
    "pure_sway_0p3_mps_N_Nm": [0, .3, 0, 0, 0, 0],
    "pure_yaw_0p3_radps_Y_N": [0, 0, 0, 0, 0, .3],
    "pure_yaw_0p3_radps_N_Nm": [0, 0, 0, 0, 0, .3],
    "turn_u1_v0p3_r0p1_Y_N": [1, .3, 0, 0, 0, .1],
    "turn_u1_v0p3_r0p1_N_Nm": [1, .3, 0, 0, 0, .1],
}


def run(root: Path) -> dict:
    packages = {name: load_coefficient_package(root/name/"coefficient_package.yaml")
                for name in ("smooth", "inset", "outset")}
    values = {}
    for name, package in packages.items():
        matrix = np.asarray(package["added_mass"]["matrix_6x6"], dtype=float)
        row = {f"M_A_{axis}_diagonal": float(matrix[i, i]) for axis, i in
               (("sway", 1), ("heave", 2), ("pitch", 4), ("yaw", 5))}
        for key, state in STATES.items():
            wrench = reference_wrench(package, np.asarray(state, dtype=float))
            component = 0 if "_X_N" in key else 1 if "_Y_N" in key else 5
            row[key] = float(wrench[component])
        row["draft_m"] = float(package["reference"]["draft_m"])
        values[name] = row
    rows = []
    for metric in values["smooth"]:
        low, nominal, high = (values[name][metric] for name in ("inset", "smooth", "outset"))
        spread = (max(low, nominal, high)-min(low, nominal, high))/max(abs(nominal), 1e-12)*100
        rows.append({"metric": metric, "inset": low, "smooth": nominal, "outset": high,
                     "spread_percent_of_smooth_magnitude": spread})
    report = {"schema": "surveyor-geometry-hypothesis-coefficient-comparison-v1",
              "status": "DIAGNOSTIC_ONLY_SOURCE_FIDELITY_NOT_ACCEPTED",
              "mass_kg": 52.3, "cg_frd_m": [0, 0, 0],
              "geometry_hypotheses": list(packages),
              "package_sha256": {name: package["canonical_sha256"] for name, package in packages.items()},
              "added_mass_method": {name: package["added_mass"]["method"] for name, package in packages.items()},
              "rows": rows,
              "interpretation": "conditional bridge-shape spread only; does not bound mixed-group classification or starboard-source discrepancy"}
    target = root.parent
    (target/"coefficient_spread.json").write_text(json.dumps(report, indent=2) + "\n")
    with (target/"coefficient_spread.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/compare_surveyor_geometry_hypotheses.py COEFFICIENT_DIR")
    report = run(Path(sys.argv[1]))
    print({row["metric"]: round(row["spread_percent_of_smooth_magnitude"], 3)
           for row in report["rows"]})
