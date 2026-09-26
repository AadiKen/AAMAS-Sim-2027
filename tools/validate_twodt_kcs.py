"""Run the frozen HMRI projection with only the sectional Y/N provider replaced.

Invoke only after tools/validate_twodt_generic.py freezes the model. The
benchmark runner, conditions, reference values and fitting equations are
imported unchanged from validate_kcs_hmri_passive.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ["BCOD_HMRI_OUTPUT"] = "stage3_results/twodt/level_a_campaign"
os.environ["BCOD_HMRI_PACKAGE"] = "stage3_results/simple_hydrodynamics/phase3a/final_kcs/canonical"

from tools import validate_kcs_hmri_passive as campaign  # noqa: E402
from bcod_sim.vessel_generation.twodt.level_a import SectionalTwoDt, sectional_v5_arrays  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage3_results/twodt"
PACKAGE = ROOT / os.environ["BCOD_HMRI_PACKAGE"]


def main():
    freeze = json.loads((OUT / "level_a_freeze.json").read_text())
    if not freeze["generic_pass"]:
        raise RuntimeError("Generic 2D+t gate must pass before KCS")
    for rel, expected in freeze["code_sha256"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError("2D+t code changed after pre-KCS freeze: " + rel)
    original_loader = campaign.load_model
    _, original_evaluate = original_loader()
    stations = json.loads((PACKAGE / "maneuvering.json").read_text())["crossflow"]["stations"]
    model = SectionalTwoDt(stations, density=campaign.RHO)

    def load_model():
        payload, _ = original_loader()

        def evaluate(vprime: float, rprime: float) -> dict:
            old = original_evaluate(vprime, rprime)
            v = campaign.SPEED * vprime
            u = math.sqrt(max(campaign.SPEED**2 - v**2, 0.))
            r = campaign.SPEED / campaign.LENGTH * rprime
            result = model.steady_captive(u, v, r)
            added = np.zeros(6)
            added[1] = result["Y_n"] - old["crossflow"][1]
            added[5] = result["N_nm"] - old["crossflow"][5]
            modified = {name: value.copy() for name, value in old.items()}
            modified["physical"] += added
            modified["rhs"] += added
            modified["crossflow"] += added
            return modified
        return payload, evaluate

    campaign.load_model = load_model
    campaign.main()
    validation = json.loads((OUT / "level_a_campaign/validation.json").read_text())
    validation["experimental_provider"] = "2D+t Level A, frozen before HMRI"
    validation["scope_warning"] = (
        "Frozen campaign samples sinusoidal PMM phases quasi-statically; "
        "true history-dependent PMM prediction needs test frequency and explicit time history. "
        "Static-drift held-out forces are physically interpretable; yaw/mixed fits are diagnostic projections."
    )
    (OUT / "level_a_kcs_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    static = next(row for row in validation["held_out"] if row["family"] == "static_drift" and row["beta_deg"] == 6.)
    drift = model.steady_captive(campaign.SPEED*math.cos(math.radians(6)),
                                 -campaign.SPEED*math.sin(math.radians(6)), 0.)
    old = sectional_v5_arrays(stations, campaign.SPEED*math.cos(math.radians(6)),
                              -campaign.SPEED*math.sin(math.radians(6)), 0.)
    order = np.argsort(drift["x_m"])
    (OUT / "kcs_plus6_section_distribution.json").write_text(json.dumps({
        "condition": "HMRI beta=+6 deg; geometry only, no reference values in model",
        "HMRI_static_Y_prime": static["HMRI_Y_prime"],
        "HMRI_static_N_prime": static["HMRI_N_prime"],
        "stations": [{"x_m": float(drift["x_m"][i]), "q_mps": float(drift["q_mps"][i]),
                      "t_star": float(drift["t_star"][i]), "cd_steady": float(drift["cd_steady"][i]),
                      "cd_effective": float(drift["cd_effective"][i]),
                      "V5_dY_n": float(old["dY_n"][i]), "twodt_dY_n": float(drift["dY_n"][i]),
                      "V5_dN_nm": float(old["dN_nm"][i]), "twodt_dN_nm": float(drift["dN_nm"][i])}
                     for i in order],
        "V5_cumulative_Y_n": old["cumulative_Y_n"].tolist(),
        "V5_cumulative_N_nm": old["cumulative_N_nm"].tolist(),
        "twodt_cumulative_Y_n": drift["cumulative_Y_n"].tolist(),
        "twodt_cumulative_N_nm": drift["cumulative_N_nm"].tolist(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
