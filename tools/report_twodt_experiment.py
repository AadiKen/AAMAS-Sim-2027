"""Assemble measured 2D+t experiment evidence without rerunning CFD."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage3_results/twodt"


def read(path):
    return json.loads((ROOT / path).read_text())


def put(name, content):
    (OUT / name).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def forces(case):
    folder = OUT / "level_b_cases" / case
    rows = {}
    for path in (folder / "postProcessing/sectionForces").glob("*/forces.dat"):
        for line in path.read_text().splitlines():
            values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if line.startswith("#") or len(values) != 13:
                continue
            z = list(map(float, values))
            rows[z[0]] = (z[1] + z[4], z[2] + z[5])
    history = np.array([(t, *f) for t, f in sorted(rows.items())])
    if history.size == 0:
        raise RuntimeError(f"No forces for {case}")
    config = json.loads((folder / "case_config.json").read_text())
    source_hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted([folder / "system/blockMeshDict", folder / "0/U",
                                      folder / "0/p", folder / "system/controlDict",
                                      folder / "constant/physicalProperties"])}
    tail = history[history[:, 0] >= history[-1, 0] - 20.]
    return {"case": case, "settings": config, "source_sha256": source_hashes,
            "sample_count": len(history), "last_time_s": float(history[-1, 0]),
            "last_20_s_mean_cd": float(np.mean(tail[:, 1])),
            "last_20_s_cd_range": [float(np.min(tail[:, 1])), float(np.max(tail[:, 1]))],
            "last_20_s_lift_range": [float(np.min(tail[:, 2])), float(np.max(tail[:, 2]))],
            "mesh_check_pass": "Mesh OK." in (folder / "log.checkMesh").read_text()
            if (folder / "log.checkMesh").exists() else "previously_observed_pass_log_not_saved"}


def main():
    a = read("stage3_results/twodt/level_a_kcs_validation.json")
    independent = read("stage3_results/twodt/level_a_independent_validation.json")
    prior = read("stage3_results/simple_hydrodynamics/phase3a/final_comparison.json")
    cases = [forces(name) for name in ("cylinder_re100", "cylinder_re100_fine",
                                     "rectangle_re100", "kcs_midship_re100")]
    cylinder_coarse, cylinder_fine, rectangle, kcs_section = cases
    solver = {"status": "NOT_QUALIFIED_FOR_VESSEL_LEVEL_B",
              "solver": "OpenFOAM 11 incompressibleFluid, laminar Re=100, 2D extruded O-grid",
              "cases": cases,
              "canonical_cylinder_reference": {"Cd": 1.338, "St": 0.164,
                  "url": "https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D68EA98129B8C4A8F80F5F69584FE35C/S0022112011001340a.pdf/vortex_suppression_and_drag_reduction_in_the_wake_of_counterrotating_cylinders.pdf"},
              "fine_cylinder_cd_error_percent": 100*(cylinder_fine["last_20_s_mean_cd"] / 1.338 - 1),
              "coarse_fine_cd_difference_percent": 100*(cylinder_fine["last_20_s_mean_cd"] /
                    cylinder_coarse["last_20_s_mean_cd"] - 1),
              "fine_cylinder_strouhal_approx": 0.150,
              "fine_cylinder_strouhal_note": "Estimated from late lift oscillation (f about 0.075, D=2); not a formal uncertainty estimate",
              "missing_gates": ["spatial convergence", "time-step sensitivity", "domain-size sensitivity",
                                 "wall-treatment and high-Re turbulence qualification",
                                 "published rectangle mean-drag benchmark and settled wake",
                                 "representative vessel-section stationary startup Cd(t*) curves"],
              "conclusion": "2D case generation and force extraction work, but no sectional hydrodynamic response is qualified for vessel inference"}
    put("level_b_solver_validation.json", solver)
    put("level_b_section_database.json", {"status": "EMPTY_QUALIFIED_LIBRARY",
         "qualified_entry_count": 0, "candidate_case_count": len(cases),
         "cache_key_schema": "normalized_contour_sha256 + Reynolds + solver_version + mesh_and_BC_hash",
         "family_reduction": "deterministic per-hull medoid selection tested in section_geometry_validation.json",
         "reason": "Canonical and representative section convergence gates have not passed"})
    for name in ("level_b_kcs_validation.json", "level_b_independent_validation.json"):
        put(name, {"status": "NOT_RUN_QUALIFICATION_GATE", "reference_used_for_tuning": False,
                   "reason": "No qualified Level B section Cd(t*) database; using the Re=100 smoke outputs would misrepresent vessel Reynolds number"})
    put("dynamic_validation.json", {"status": "NOT_RUN_CANDIDATE_REJECTED",
         "plant6_v5_baseline": "preserved in stage3_results/simple_hydrodynamics/phase3a/final_kcs/canonical/plant6_runtime_validation.json",
         "reason": "Published captive 2D+t age is a spatial-convection mapping, not a validated time-domain section-state update; the candidate failed KCS/independent gates and was not wired into Plant6",
         "requirement_for_future": "separate deterministic section memory states integrated with Plant6 timestep acceptance and reversal logic"})
    generic = read("stage3_results/twodt/level_a_section_validation.json")
    multi = [h for h in generic["fleet"] if h["hull"] == "catamaran"]
    put("multihull_validation.json", {"status": "STRUCTURAL_PASS_QUANTITATIVE_UNVALIDATED",
         "catamaran_generic": multi, "independent_hull_sectioning": True,
         "limitation": "No 2D section model here captures inter-hull wake/pressure interference"})
    a_summary = dict(a["summary"])
    rows = a["coefficient_table"]
    a_summary.update(Y_family_median_error_percent=median(x["error_percent"] for x in rows if x["name"].startswith("Y_")),
                     N_family_median_error_percent=median(x["error_percent"] for x in rows if x["name"].startswith("N_")),
                     coupling_median_error_percent=median(x["error_percent"] for x in rows if "vvr" in x["name"] or "vrr" in x["name"]))
    table = {k: prior["summary"][k] for k in ("V1", "V2", "final")}
    table["2D+t A"] = a_summary
    table["2D+t B"] = None
    put("final_comparison.json", {"outcome": "ESCALATE TO VIRTUAL PMM", "summary": table,
         "level_b_absence": "No qualified response library; comparison would be invalid",
         "independent_kvlcc2": independent["summary"],
         "level_a_kcs_coefficient_table": rows,
         "held_out_conditions": a["held_out"],
         "distribution_source": "kcs_plus6_section_distribution.json"})
    put("runtime.json", {"level_a_locked_kcs_campaign_wall_s": a["campaign_wall_time_s"],
         "level_a_section_evaluation_typical_wall_s": "~0.001 per captive state from generic fleet tests",
         "level_b_case_simulated_times": {c["case"]: c["last_time_s"] for c in cases},
         "level_b_solver_wall_s": "Not reliably recorded across Docker restart/split runs; no total-time claim",
         "level_b_efficiency_result": "Sparse sections still require canonical convergence and Reynolds-matched turbulence; minutes-per-vessel target unproven"})


if __name__ == "__main__":
    main()
