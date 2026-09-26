"""Pre-HMRI physics and fleet gate for experimental 2D+t Level A."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from time import perf_counter

import numpy as np

from bcod_sim.vessel_generation.twodt.level_a import SectionalTwoDt, sectional_v5_arrays

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "stage3_results/simple_hydrodynamics/phase3a/final_fleet_equilibrium"
OUT = ROOT / "stage3_results/twodt"


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(FLEET.glob("*/maneuvering.json")):
        stations = json.loads(path.read_text())["crossflow"]["stations"]
        model = SectionalTwoDt(stations)
        start = perf_counter()
        cases = []
        for name, u, v, r in (("sway", 1., .2, 0.), ("yaw", 1., 0., .1),
                              ("combined", 1., .1, .1)):
            a = model.steady_captive(u, v, r)
            b = sectional_v5_arrays(stations, u, v, r)
            negative = model.steady_captive(u, -v, -r)
            sign = np.allclose((a["Y_n"], a["N_nm"]),
                               (-negative["Y_n"], -negative["N_nm"]), atol=1e-9)
            dissipation = v*a["Y_n"] + r*a["N_nm"]
            cases.append({"name": name, "u_mps": u, "v_mps": v, "r_rad_s": r,
                          "Y_2dt_N": a["Y_n"], "N_2dt_Nm": a["N_nm"],
                          "Y_V5_N": b["Y_n"], "N_V5_Nm": b["N_nm"],
                          "max_t_star": float(max(a["t_star"])),
                          "sign_odd": bool(sign), "lateral_power_W": float(dissipation),
                          "finite": bool(np.isfinite([a["Y_n"], a["N_nm"]]).all())})
        results.append({"hull": path.parent.name, "station_count": len(stations),
                        "hull_components": len({s["hull_id"] for s in stations}),
                        "wall_time_s": perf_counter()-start, "cases": cases})
    checks = [case for row in results for case in row["cases"]]
    status = all(c["finite"] and c["sign_odd"] and c["lateral_power_W"] <= 1e-9
                 for c in checks)
    (OUT / "level_a_section_validation.json").write_text(json.dumps({
        "status": "PASS_GENERIC_TESTS" if status else "FAIL_GENERIC_TESTS",
        "source": "Rabliås and Kristiansen (2021), Eqs. 19-21 / Table 1",
        "family": "scaled cylinder startup history with geometry-derived sectional Cd_inf",
        "canonical_ratio": {str(t): float(__import__("bcod_sim.vessel_generation.twodt.level_a", fromlist=["cylinder_startup_ratio"]).cylinder_startup_ratio(np.array([t]))[0])
                            for t in (0, .5, 1, 2, 5, 9, 20, 25)},
        "fleet": results,
        "scope": "steady forward captive generic fixture states; no KCS/HMRI numbers used",
    }, indent=2, default=_serial) + "\n")
    hashes = {}
    for path in sorted((ROOT / "src/bcod_sim/vessel_generation/twodt").glob("*.py")):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "level_a_freeze.json").write_text(json.dumps({
        "identifier": "bcod-twodt-level-a-experiment-1", "generic_pass": status,
        "code_sha256": hashes, "source": "level_a_section_validation.json",
        "HMRI_access_before_freeze": False,
    }, indent=2) + "\n")
    print("generic gate:", status, "fleet:", len(results))


if __name__ == "__main__":
    main()
