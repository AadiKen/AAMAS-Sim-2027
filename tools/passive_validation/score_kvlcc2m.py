"""Score the already frozen KVLCC2M package against staged NMRI references."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from bcod_sim.vessel_generation.coefficient_package import (
    _reference_crossflow, _reference_damping, _runtime_plant,
    load_coefficient_package, validate_coefficients,
)
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.state.vessel_state import VesselState

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs/passive_hull_validation/benchmarks/kvlcc2m"
PACKAGE = BENCH / "package/coefficient_package.yaml"
MANIFEST = BENCH / "package/package_manifest.json"
REFERENCE = BENCH / "reference"
RESULTS = BENCH / "results"
PLOTS = BENCH / "plots"
RHO, U, LPP, DRAFT, AREA = 1025., .994, 4.97, .323, 6.58919


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen():
    manifest = json.loads(MANIFEST.read_text())
    if digest(PACKAGE) != manifest["package_file_sha256"]:
        raise RuntimeError("Frozen package file hash changed after pre-score freeze")
    package = load_coefficient_package(PACKAGE)
    if package["canonical_sha256"] != manifest["canonical_package_sha256"]:
        raise RuntimeError("Canonical frozen package hash mismatch")
    return package, manifest


def plant_wrench(package, nu):
    plant = _runtime_plant(package)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    state = VesselState(t([0., 0., 0.]), t([1., 0., 0., 0.]), t(nu))
    zeros = {name: t([0.] * 6) for name in EXTERNAL_TERMS}
    ledger = plant.diagnostics(state, zeros)
    return {name: value.detach().cpu().numpy() for name, value in ledger.terms.items()}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    package, manifest = load_frozen()
    roundtrip = validate_coefficients(PACKAGE, grid_size=7)
    if not roundtrip["passed"]:
        raise RuntimeError("Frozen coefficient package failed runtime roundtrip")
    (RESULTS / "runtime_roundtrip.json").write_text(json.dumps(roundtrip, indent=2)+"\n")

    # Physical comparison begins only after load_frozen has verified the seal.
    resistance_ref = json.loads((REFERENCE / "resistance.json").read_text())
    efd_ct = float(resistance_ref["experimental"]["CT"])
    p = package["runtime_payload"]
    speed_curve = p["surge_resistance"]
    manta_rt = -float(np.interp(U, speed_curve["speed_mps"], speed_curve["force_x_n"]))
    manta_ct = manta_rt / (.5 * RHO * U**2 * AREA)
    ct_error = manta_ct - efd_ct
    ct_relative = abs(ct_error) / abs(efd_ct) * 100.
    ct_uncertainty_pct = float(resistance_ref["official_condition"]["experimental_uncertainty_percent_of_CT"])
    ct_result = {
        "benchmark": resistance_ref["benchmark"], "speed_mps": U,
        "Re": resistance_ref["official_condition"]["Re"], "Fn": 0.,
        "reference_area_m2": AREA, "density_kg_m3": RHO,
        "experimental_CT": efd_ct, "manta_total_resistance_N": manta_rt,
        "manta_CT": manta_ct, "absolute_error": abs(ct_error),
        "relative_error_percent": ct_relative,
        "experimental_uncertainty_percent": ct_uncertainty_pct,
        "error_in_experimental_uncertainties": ct_relative / ct_uncertainty_pct,
        "experimental_source_url": resistance_ref.get("source_url"),
        "condition_source_url": resistance_ref.get("official_condition_url"),
        "source_quality": resistance_ref.get("source_quality"),
        "package_canonical_sha256": manifest["canonical_package_sha256"],
        "score_status": "SCORED",
    }
    (RESULTS / "resistance.json").write_text(json.dumps(ct_result, indent=2)+"\n")

    efd = json.loads((REFERENCE / "admitted_reference.json").read_text())
    rows = efd["rows"]
    force_scale = .5 * RHO * U**2 * LPP * DRAFT
    moment_scale = force_scale * LPP
    scored = []
    for row in rows:
        beta = math.radians(float(row["beta_deg"]))
        nu = np.array((U*math.cos(beta), -U*math.sin(beta), 0., 0., 0., 0.))
        ledger = plant_wrench(package, nu)
        total = sum((ledger[k] for k in ("added_mass_coriolis", "linear_damping",
                                         "nonlinear_damping", "crossflow")), np.zeros(6))
        manta = {"CX": float(total[0] / force_scale),
                 "CY": float(total[1] / force_scale),
                 "CN": float(total[5] / moment_scale)}
        result = {"beta_deg": row["beta_deg"], "u_mps": nu[0], "v_mps": nu[1],
                  "r_rad_s": 0., "experimental_CX": row["CX"], "manta_CX": manta["CX"],
                  "experimental_CY": row["CY"], "manta_CY": manta["CY"],
                  "experimental_CN": row["CN"], "manta_CN": manta["CN"],
                  "CX_abs_error": abs(manta["CX"]-row["CX"]),
                  "CY_abs_error": abs(manta["CY"]-row["CY"]),
                  "CN_abs_error": abs(manta["CN"]-row["CN"]),
                  "CY_uncertainty": row.get("CY_uncertainty"),
                  "CN_uncertainty": row.get("CN_uncertainty"),
                  "representative_force_scale_N": force_scale,
                  "representative_moment_scale_Nm": moment_scale}
        # Avoid percentage error for near-zero experimental values.
        for key in ("CX", "CY", "CN"):
            ev, mv = result[f"experimental_{key}"], result[f"manta_{key}"]
            result[f"{key}_relative_error_percent"] = (None if abs(ev) < 1e-3 else
                abs(mv-ev)/abs(ev)*100.)
        result["zero_drift_residual_over_force_scale"] = (
            max(result["CX_abs_error"], result["CY_abs_error"]) if row["beta_deg"] == 0 else None)
        result.update({"added_mass_coriolis_X": float(ledger["added_mass_coriolis"][0]),
                       "added_mass_coriolis_Y": float(ledger["added_mass_coriolis"][1]),
                       "added_mass_coriolis_N": float(ledger["added_mass_coriolis"][5]),
                       "linear_X": float(ledger["linear_damping"][0]),
                       "linear_Y": float(ledger["linear_damping"][1]),
                       "linear_N": float(ledger["linear_damping"][5]),
                       "crossflow_X": float(ledger["crossflow"][0]),
                       "crossflow_Y": float(ledger["crossflow"][1]),
                       "crossflow_N": float(ledger["crossflow"][5]),
                       "surge_resistance_X": float(ledger["nonlinear_damping"][0])})
        scored.append(result)
    fieldnames = list(scored[0])
    with (RESULTS / "static_drift.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(scored)

    # Use the package's full steady force evaluator to inspect component loads.
    beta_grid = np.array([x["beta_deg"] for x in scored])
    for coeff, exp_key, manta_key, title, ylabel, filename in (
        ("CX", "experimental_CX", "manta_CX", "Surge coefficient", "$C_X$", "CX_vs_beta.png"),
        ("CY", "experimental_CY", "manta_CY", "Lateral force coefficient", "$C_Y$", "CY_vs_beta.png"),
        ("CN", "experimental_CN", "manta_CN", "Yaw moment coefficient", "$C_N$", "CN_vs_beta.png"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(beta_grid, [r[exp_key] for r in scored], "o", label="NMRI EFD")
        ax.plot(beta_grid, [r[manta_key] for r in scored], "-", label="Frozen MANTA")
        ax.set(xlabel="Drift angle beta (deg)", ylabel=ylabel, title=title)
        ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(PLOTS / filename, dpi=160); plt.close(fig)

    ref_resistance_force = .5*RHO*U**2*AREA*efd_ct
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["NMRI EFD", "MANTA"], [ref_resistance_force, manta_rt])
    ax.set(ylabel="Total resistance (N)", title="KVLCC2M fixed towing")
    fig.tight_layout(); fig.savefig(PLOTS / "resistance.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, key, efd_key, manta_key in zip(axes, ("X", "Y", "N"),
            ("experimental_CX", "experimental_CY", "experimental_CN"),
            ("manta_CX", "manta_CY", "manta_CN")):
        ax.plot(beta_grid, [r[efd_key] for r in scored], "o", label="EFD")
        ax.plot(beta_grid, [r[manta_key] for r in scored], "-", label="MANTA total")
        if key != "N":
            scale = force_scale
            cv = lambda r, k: r[f"crossflow_{k}"]/scale
            lv = lambda r, k: r[f"linear_{k}"]/scale
            rv = lambda r, k: r[f"surge_resistance_{k}"]/scale
            av = lambda r, k: r[f"added_mass_coriolis_{k}"]/scale
        else:
            scale = moment_scale
            cv = lambda r, k: r[f"crossflow_{k}"]/scale
            lv = lambda r, k: r[f"linear_{k}"]/scale
            rv = lambda r, k: r[f"surge_resistance_X"]/scale
            av = lambda r, k: r[f"added_mass_coriolis_N"]/scale
        ax.plot(beta_grid, [cv(r,key) for r in scored], "--", label="crossflow")
        ax.plot(beta_grid, [lv(r,key) for r in scored], ":", label="linear")
        ax.plot(beta_grid, [av(r,key) for r in scored], "-.", label="added-mass/Coriolis")
        if key == "X": ax.plot(beta_grid, [rv(r,key) for r in scored], "-.", label="surge")
        ax.set_ylabel(f"C_{key}"); ax.grid(True, alpha=.3); ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Drift angle beta (deg)")
    fig.suptitle("Frozen MANTA force decomposition")
    fig.tight_layout(); fig.savefig(PLOTS / "component_decomposition.png", dpi=160); plt.close(fig)

    # Centered derivatives at zero sway/yaw, with the same fixed tow speed.
    eps_v, eps_r = U*1e-3, .01
    def physical(nu):
        load = plant_wrench(package, nu)
        total = sum((load[k] for k in ("added_mass_coriolis", "linear_damping",
                                       "nonlinear_damping", "crossflow")), np.zeros(6))
        return total
    def derivative(axis, epsilon):
        plus, minus = np.array((U,0,0,0,0,0),float), np.array((U,0,0,0,0,0),float)
        plus[axis], minus[axis] = epsilon, -epsilon
        return (physical(plus)-physical(minus))/(2*epsilon)
    dv, dr = derivative(1,eps_v), derivative(5,eps_r)
    derivatives = {
        "method": "centered finite difference on frozen Plant6 force evaluator",
        "epsilon_v_mps": eps_v, "epsilon_r_rad_s": eps_r,
        "Y_v_N_per_mps": float(dv[1]), "N_v_Nm_per_mps": float(dv[5]),
        "Y_r_N_per_rad_s": float(dr[1]), "N_r_Nm_per_rad_s": float(dr[5]),
        "Y_v_prime": float(dv[1]/(.5*RHO*U*LPP*DRAFT)),
        "N_v_prime": float(dv[5]/(.5*RHO*U*LPP**2*DRAFT)),
        "Y_r_prime": float(dr[1]/(.5*RHO*U*LPP**2*DRAFT)),
        "N_r_prime": float(dr[5]/(.5*RHO*U*LPP**3*DRAFT)),
        "literature_comparison": "KVLCC2 references are contextual only; KVLCC2 != KVLCC2M and derivative convention is flagged ambiguous.",
    }
    (RESULTS / "derivatives.json").write_text(json.dumps(derivatives, indent=2)+"\n")
    with (RESULTS / "derivatives.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(derivatives));writer.writeheader();writer.writerow(derivatives)

    # Vary only the frozen added-mass matrix after the primary score.
    base_mass = np.asarray(package["added_mass"]["matrix_6x6"], float)
    sensitivity = []
    for scale in (.5, 1., 1.5):
        variant = copy.deepcopy(package)
        variant["runtime_payload"]["added_mass_kg"] = (base_mass*scale).tolist()
        item = {"scale": scale}
        for beta in (0., 6., 12.):
            rad = math.radians(beta)
            nu = np.array((U*math.cos(rad), -U*math.sin(rad),0.,0.,0.,0.))
            load = plant_wrench(variant,nu)
            total=sum((load[k] for k in ("added_mass_coriolis","linear_damping","nonlinear_damping","crossflow")),np.zeros(6))
            item[f"CX_beta_{int(beta)}"] = float(total[0]/force_scale)
            item[f"CY_beta_{int(beta)}"] = float(total[1]/force_scale)
            item[f"CN_beta_{int(beta)}"] = float(total[5]/moment_scale)
        sensitivity.append(item)
    with (RESULTS/"added_mass_sensitivity.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(sensitivity[0]));writer.writeheader();writer.writerows(sensitivity)

    summary={"package_canonical_sha256":manifest["canonical_package_sha256"],
             "roundtrip_passed":roundtrip["passed"],"resistance":ct_result,
             "drift_rows":len(scored),"scored_required_betas_deg":[0,6,12],
             "required_point_scores":{
                 str(int(row["beta_deg"])):{
                     "experimental":{"CX":row["experimental_CX"],"CY":row["experimental_CY"],"CN":row["experimental_CN"]},
                     "manta":{"CX":row["manta_CX"],"CY":row["manta_CY"],"CN":row["manta_CN"]},
                     "absolute_error":{"CX":row["CX_abs_error"],"CY":row["CY_abs_error"],"CN":row["CN_abs_error"]},
                     "relative_error_percent":{"CX":row["CX_relative_error_percent"],"CY":row["CY_relative_error_percent"],"CN":row["CN_relative_error_percent"]}}
                 for row in scored if row["beta_deg"] in (0,6,12)},
             "added_mass_sensitivity_spread":{
                 key:max(r[key] for r in sensitivity)-min(r[key] for r in sensitivity)
                 for key in sensitivity[0] if key.startswith(("CX_","CY_","CN_"))}}
    (RESULTS/"score_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
