"""Summarize the experimental SQCM qualification without opening HMRI inputs.

Only measured component/probe outputs and the frozen Phase 3A comparison are
used.  Unqualified downstream results are represented as null, never estimated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage3_results/sqcm_hybrid"
BASE = ROOT / "stage3_results/simple_hydrodynamics/phase3a/final_comparison.json"


def read(name: str):
    return json.loads((OUT / name).read_text())


def write(name: str, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    probes = read("wake_segmentation_probe.json")
    fixed = read("published_scale_fixed_wake_probe.json")
    wake = []
    for run in probes:
        tail = run["history"][-20:]
        wake.append({
            "model": 2, "wake_elements": run["wake_elements"],
            "iterations": run["iterations"], "converged": run["converged"],
            "wall_time_s": run["wall_time_s"],
            "last_Y_N": tail[-1]["Y_n"], "last_N_Nm": tail[-1]["N_nm"],
            "last_20_Y_range_N": [min(x["Y_n"] for x in tail), max(x["Y_n"] for x in tail)],
            "last_20_N_range_Nm": [min(x["N_nm"] for x in tail), max(x["N_nm"] for x in tail)],
            "last_20_max_relative_force_change": max(x["relative_force_change"] for x in tail),
        })
    write("wake_validation.json", {
        "status": "FAILED_QUALIFICATION", "condition": "Wigley L=2.5 m, U=0.5 m/s, beta=+10 deg; coarse 8x2 source grid; Model 2",
        "criterion": "Four consecutive coupled steps with <1% relative Y and N change",
        "runs": wake, "published_scale_fixed_wake": fixed,
        "interpretation": "Low linear-system residual does not establish wake or force convergence. Fixed-wake loads are diagnostic only.",
        "source": "wake_segmentation_probe.json",
    })
    write("wigley_reproduction.json", {
        "status": "NOT_REPRODUCED", "source_only_straight_ahead": "Symmetric Y and N near numerical zero at 8x3, 16x4, 30x5; see tests.",
        "oblique_fixed_wake": fixed,
        "oblique_deformed_wake": "Model 2 did not meet the force-convergence criterion at 25 or 50 elements in 100 iterations on a coarse grid.",
        "published_experiment_numeric_Y_N": None,
        "published_SQCM_numeric_Y_N": None,
        "comparison": None,
        "reason": "The accessible thesis presents the target trends graphically; numeric data were not extracted, and this solver's required wake state did not converge.",
    })
    write("published_ship_reproduction.json", {
        "status": "NOT_RUN_GATE", "reason": "Published Wigley reproduction and wake convergence must pass first.",
        "KCS": None, "KVLCC2": None,
    })
    code = {}
    for path in sorted((ROOT / "src/bcod_sim/vessel_generation/sqcm").glob("*.py")):
        code[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write("sqcm_freeze.json", {
        "status": "NO_QUALIFIED_CANDIDATE_FREEZE", "candidate_identifier": None,
        "experimental_code_hashes": code,
        "unmet_gates": ["canonical vortex lift/Kutta", "convergent Model 2 wake", "quantitative Wigley reproduction", "published ship reproduction"],
        "baseline_release_unchanged": True,
    })
    gated = {
        "kcs_hmri_validation.json": "SQCM candidate was not frozen; locked HMRI campaign was not opened for this model.",
        "applicability_map.json": "No qualified forward-speed SQCM force law exists to sweep.",
        "large_drift_validation.json": "Published small/moderate-drift validity does not justify an unqualified large-drift sweep.",
        "low_speed_validation.json": "Forward-speed SQCM lacks a qualified force solution; V5 remains the zero/reverse-speed path.",
        "force_map_validation.json": "A force map would interpolate unqualified SQCM loads.",
        "hybrid_validation.json": "No validated SQCM domain or transition weight can be selected without tuning.",
        "plant6_validation.json": "No qualified SQCM or hybrid load provider was connected to Plant6.",
    }
    for name, reason in gated.items():
        write(name, {"status": "NOT_RUN_GATE", "reason": reason, "result": None})
    write("runtime.json", {
        "status": "EXPERIMENTAL_COMPONENT_PROFILE", "published_scale_fixed_wake_state_s": {str(x["model"]): x["fixed_wake_state_time_s"] for x in fixed},
        "coarse_deforming_wake_100_iteration_s": {str(x["wake_elements"]): x["wall_time_s"] for x in probes},
        "source_sphere_384_panel_s": read("analytic_source_validation.json")["sphere"][-1]["solve_wall_time_s"],
        "qualified_full_state_s": None, "force_map_generation_s": None,
        "interpretation": "The standalone kernel lower bound does not include repeated panel influence assembly, pressure integration, and full wake convection.",
    })
    base = json.loads(BASE.read_text())["summary"]
    comparison = {"status": "SQCM_NOT_COMPARABLE", "decision": "KEEP V5",
                  "reason": "Required wake and literature-reproduction gates failed; SQCM and Hybrid have no defensible HMRI or Plant6 metrics.",
                  "metrics": {key: {**base[source], "runtime_per_state_s": None,
                                     "full_generation_s": 2.19 if key == "V5" else None}
                              for key, source in (("V1", "V1"), ("V2", "V2"), ("V5", "final"))}}
    comparison["metrics"]["SQCM"] = None
    comparison["metrics"]["Hybrid"] = None
    write("final_comparison.json", comparison)


if __name__ == "__main__":
    main()
