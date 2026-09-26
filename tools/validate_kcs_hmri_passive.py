"""Uncalibrated HMRI-equivalent fits from one frozen production vessel package."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from bcod_sim.dynamics.coriolis import coriolis_wrench
from bcod_sim.dynamics.crossflow import SectionalCrossflow
from bcod_sim.dynamics.damping import Damping
from bcod_sim.state.vessel_state import VesselState


BASELINE_ROOT = Path("stage3_results/simple_hydrodynamics/hmri_coefficient_campaign")
ROOT = Path(os.environ.get("BCOD_HMRI_OUTPUT", str(BASELINE_ROOT)))
PACKAGE = Path(os.environ.get("BCOD_HMRI_PACKAGE", str(BASELINE_ROOT / "canonical")))
FREQUENCY_FILE = Path(os.environ.get("BCOD_HMRI_FREQUENCY_FILE", str(BASELINE_ROOT / "frequency_sweep.json")))
RHO, LENGTH, DRAFT, SPEED = 1025., 5.75, .270, 1.953
FY = .5 * RHO * LENGTH * DRAFT * SPEED**2
NM = FY * LENGTH
EXPERIMENT = Path("stage3_results/kcs-validation/track_b_hmri/experimental_coefficients.json")


def load_model() -> tuple[dict, callable]:
    payload = json.loads((PACKAGE / "runtime_payload.json").read_text())
    maneuvering = json.loads((PACKAGE / "maneuvering.json").read_text())
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    crossflow = SectionalCrossflow.from_stations(maneuvering["crossflow"]["stations"], density=RHO)
    curve = payload["surge_resistance"]
    damping = Damping(t(payload["linear_damping"]), t(payload["quadratic_damping"]),
        t(payload["linear_damping_matrix"]),
        surge_resistance_curve=(t(curve["speed_mps"]), t(curve["force_x_n"])))
    rigid = t(json.loads((PACKAGE / "coefficients.json").read_text())["M_RB"])
    added = t(payload["added_mass_kg"])
    zero = t([0.] * 6)
    identity = t([1., 0., 0., 0.])

    def evaluate(vprime: float, rprime: float) -> dict:
        v = SPEED * vprime
        u = math.sqrt(max(SPEED**2 - v**2, 0.))
        r = SPEED / LENGTH * rprime
        nu = t([u, v, 0., 0., 0., r])
        state = VesselState(zero[:3], identity, nu)
        lin, nonlinear = damping.components(nu)
        cf = crossflow.evaluate(state).tau_body
        ca = -coriolis_wrench(added, nu)
        crb = -coriolis_wrench(rigid, nu)
        physical = lin + nonlinear + cf + ca
        rhs = physical + crb
        return {"physical": physical.numpy().copy(), "rhs": rhs.numpy().copy(),
                "linear": lin.numpy().copy(), "nonlinear": nonlinear.numpy().copy(),
                "crossflow": cf.numpy().copy(), "added_coriolis": ca.numpy().copy(),
                "rigid_coriolis": crb.numpy().copy()}
    return payload, evaluate


def fit_odd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack((x, x**3))
    return np.linalg.lstsq(design, y, rcond=None)[0]


def harmonics(evaluate, vprime: float, r_amplitude: float, phases: np.ndarray) -> dict:
    values = [evaluate(vprime, r_amplitude * math.sin(float(phase))) for phase in phases]
    out = {}
    for representation in ("physical", "rhs", "linear", "nonlinear", "crossflow",
                           "added_coriolis", "rigid_coriolis"):
        matrix = np.array([item[representation] for item in values])
        out[representation] = {"mean": np.mean(matrix, axis=0),
                               "sine_first": 2 * np.mean(matrix * np.sin(phases)[:, None], axis=0)}
    return out


def category(value: float, reference: dict) -> str:
    lo, hi, est = reference["low95"], reference["high95"], reference["estimate"]
    if lo <= value <= hi:
        return "inside HMRI 95% interval"
    error = abs(value - est) / abs(est) if est else math.inf
    if error <= .1:
        return "outside interval but within 10%"
    if error <= .25:
        return "outside interval by 10–25%"
    return ">25% discrepancy"


def main() -> None:
    start = perf_counter()
    payload, evaluate = load_model()
    references = {row["name"]: row for row in json.loads(EXPERIMENT.read_text())["coefficients"]}
    records = {}
    families = {}
    component_names = ("linear", "nonlinear", "crossflow", "added_coriolis", "rigid_coriolis")
    component_fits = {name: {} for name in component_names}

    # HMRI beta = atan(-v/u). Use every 2 degrees through 24 and the 30-degree endpoint.
    drift = np.array([x for x in range(-24, 25, 2) if x] + [-30, 30], dtype=float)
    held_drift = {-6., 6.}
    x = -np.sin(np.deg2rad(drift))
    static_results = [evaluate(float(v), 0.) for v in x]
    y = np.array([result["physical"] for result in static_results])
    train = np.array([beta not in held_drift for beta in drift])
    cy = fit_odd(x[train], y[train, 1] / FY)
    cn = fit_odd(x[train], y[train, 5] / NM)
    for name, value in zip(("Y_v_prime", "Y_vvv_prime", "N_v_prime", "N_vvv_prime"), (*cy, *cn)):
        records[name] = float(value)
    for component in component_names:
        values = np.array([result[component] for result in static_results])
        c_y = fit_odd(x[train], values[train, 1]/FY)
        c_n = fit_odd(x[train], values[train, 5]/NM)
        for name, value in zip(("Y_v_prime", "Y_vvv_prime", "N_v_prime", "N_vvv_prime"), (*c_y, *c_n)):
            component_fits[component][name] = float(value)
    families["static_drift"] = {"beta_deg": drift.tolist(), "held_out_beta_deg": sorted(held_drift),
        "training_count": int(train.sum()), "fit_design": "v_prime, v_prime^3",
        "force": "physical fluid-on-hull, including added-mass Coriolis at zero yaw",
        "fit_rank": int(np.linalg.matrix_rank(np.column_stack((x[train], x[train]**3))))}

    # HMRI pure yaw: a sinusoidal PMM run at each r' maximum. HMRI used a
    # Fourier-filtered force; identify odd terms from the first sine harmonic.
    yaw_amp = np.arange(1, 7, dtype=float) / 10
    phases = np.linspace(0., 2*math.pi, 96, endpoint=False)
    held_yaw = {.3}
    yaw_harmonics = [harmonics(evaluate, 0., float(a), phases) for a in yaw_amp]
    yy = np.array([h["rhs"]["sine_first"][1] / FY for h in yaw_harmonics])
    nn = np.array([h["physical"]["sine_first"][5] / NM for h in yaw_harmonics])
    train = np.array([round(float(r), 2) not in held_yaw for r in yaw_amp])
    harmonic_design = np.column_stack((yaw_amp[train], .75*yaw_amp[train]**3))
    cyaw = np.linalg.lstsq(harmonic_design, yy[train], rcond=None)[0]
    nyaw = np.linalg.lstsq(harmonic_design, nn[train], rcond=None)[0]
    for name, value in zip(("Y_r-(m+m_x)_prime", "Y_rrr_prime", "N_r_prime", "N_rrr_prime"), (*cyaw, *nyaw)):
        records[name] = float(value)
    for component in component_names:
        yh = np.array([h[component]["sine_first"][1]/FY for h in yaw_harmonics])
        nh = np.array([h[component]["sine_first"][5]/NM for h in yaw_harmonics])
        c_y = np.linalg.lstsq(harmonic_design, yh[train], rcond=None)[0]
        c_n = np.linalg.lstsq(harmonic_design, nh[train], rcond=None)[0]
        for name, value in zip(("Y_r-(m+m_x)_prime", "Y_rrr_prime", "N_r_prime", "N_rrr_prime"), (*c_y, *c_n)):
            component_fits[component][name] = float(value)
    families["pure_yaw"] = {"r_prime_amplitudes": yaw_amp.tolist(), "held_out_r_prime_amplitudes": sorted(held_yaw),
        "fit_design": "first sine harmonic: R, (3/4)R^3", "Y_definition": "physical Y minus rigid-body Coriolis",
        "N_definition": "physical N", "training_count": int(train.sum())}

    # HMRI yaw+drift ranges: beta 0,3,6,9,12 deg; r' amplitude 0.1..0.6.
    # The published mixed model has separate v^2 r and v r^2 terms.
    mixed_states = [(float(beta), float(r)) for beta in (0, 3, 6, 9, 12)
                    for r in yaw_amp]
    held_mixed = {(6., .3)}
    rows = []
    for beta, rprime in mixed_states:
        vprime = -math.sin(math.radians(beta))
        result = harmonics(evaluate, vprime, rprime, phases)
        rows.append((beta, vprime, rprime, result))
    train_rows = [(beta, v, r, result) for beta, v, r, result in rows if (beta, round(r, 2)) not in held_mixed]
    odd_design = np.array([v*v*r for _, v, r, _ in train_rows])
    even_design = np.array([.5*v*r*r for _, v, r, _ in train_rows])
    ybase = lambda v, r: records["Y_v_prime"]*v + records["Y_vvv_prime"]*v**3 + records["Y_r-(m+m_x)_prime"]*r + records["Y_rrr_prime"]*r**3
    nbase = lambda v, r: records["N_v_prime"]*v + records["N_vvv_prime"]*v**3 + records["N_r_prime"]*r + records["N_rrr_prime"]*r**3
    ymix_odd = np.array([result["rhs"]["sine_first"][1]/FY -
        (records["Y_r-(m+m_x)_prime"]*r + .75*records["Y_rrr_prime"]*r**3)
        for _, v, r, result in train_rows])
    nmix_odd = np.array([result["physical"]["sine_first"][5]/NM -
        (records["N_r_prime"]*r + .75*records["N_rrr_prime"]*r**3)
        for _, v, r, result in train_rows])
    ymix_even = np.array([result["rhs"]["mean"][1]/FY - ybase(v, 0.)
        for _, v, r, result in train_rows])
    nmix_even = np.array([result["physical"]["mean"][5]/NM - nbase(v, 0.)
        for _, v, r, result in train_rows])
    yc = np.array([odd_design @ ymix_odd / (odd_design @ odd_design),
                   even_design @ ymix_even / (even_design @ even_design)])
    nc = np.array([odd_design @ nmix_odd / (odd_design @ odd_design),
                   even_design @ nmix_even / (even_design @ even_design)])
    for name, value in zip(("Y_vvr_prime", "Y_vrr_prime", "N_vvr_prime", "N_vrr_prime"), (*yc, *nc)):
        records[name] = float(value)
    for component in component_names:
        c = component_fits[component]
        yo = np.array([result[component]["sine_first"][1]/FY -
            (c["Y_r-(m+m_x)_prime"]*r + .75*c["Y_rrr_prime"]*r**3)
            for _, v, r, result in train_rows])
        no = np.array([result[component]["sine_first"][5]/NM -
            (c["N_r_prime"]*r + .75*c["N_rrr_prime"]*r**3)
            for _, v, r, result in train_rows])
        ye = np.array([result[component]["mean"][1]/FY -
            (c["Y_v_prime"]*v + c["Y_vvv_prime"]*v**3)
            for _, v, r, result in train_rows])
        ne = np.array([result[component]["mean"][5]/NM -
            (c["N_v_prime"]*v + c["N_vvv_prime"]*v**3)
            for _, v, r, result in train_rows])
        for name, value in (("Y_vvr_prime", odd_design @ yo / (odd_design @ odd_design)),
                            ("Y_vrr_prime", even_design @ ye / (even_design @ even_design)),
                            ("N_vvr_prime", odd_design @ no / (odd_design @ odd_design)),
                            ("N_vrr_prime", even_design @ ne / (even_design @ even_design))):
            c[name] = float(value)
    # Surge's mixed v r term is identified with a first-harmonic parity
    # contrast, cancelling speed-only resistance and even-in-v terms.
    surge_rows = []
    for beta in (3., 6., 9., 12.):
        v = -math.sin(math.radians(beta))
        for r in (.1, .2, .4, .5, .6):
            pos = harmonics(evaluate, v, r, phases)["rhs"]["sine_first"][0] / FY
            neg = harmonics(evaluate, -v, r, phases)["rhs"]["sine_first"][0] / FY
            surge_rows.append((v*r, (pos-neg)/2))
    vx = np.array([a for a, _ in surge_rows]); xf = np.array([b for _, b in surge_rows])
    records["X_vr+(m+m_y)_prime"] = float(vx @ xf / (vx @ vx))
    for component in component_names:
        samples = []
        for beta in (3., 6., 9., 12.):
            v = -math.sin(math.radians(beta))
            for r in (.1, .2, .4, .5, .6):
                pos = harmonics(evaluate, v, r, phases)[component]["sine_first"][0]/FY
                neg = harmonics(evaluate, -v, r, phases)[component]["sine_first"][0]/FY
                samples.append((pos-neg)/2)
        component_fits[component]["X_vr+(m+m_y)_prime"] = float(vx @ np.array(samples) / (vx @ vx))
    families["yaw_and_drift"] = {"beta_deg": [0,3,6,9,12], "r_prime_amplitudes": yaw_amp.tolist(),
        "held_out": [[b,r] for b,r in sorted(held_mixed)],
        "fit_design": "fixed static/pure-yaw fits; first sine harmonic v_prime^2*R and cycle mean (1/2)v_prime*R^2",
        "training_count": len(train_rows), "mixed_design_rank": 2,
        "surge_fit": "opposite-drift first-harmonic v*R parity contrast including rigid-body term"}

    # Published HMRI coefficients define the held-out expected force model.
    ref = {name: row["estimate"] for name, row in references.items()}
    held = []
    def append_held(family: str, beta: float, r: float):
        v = -math.sin(math.radians(beta))
        result = evaluate(v, r)
        terms_y = ref["Y_v_prime"]*v + ref["Y_vvv_prime"]*v**3 + ref["Y_r-(m+m_x)_prime"]*r + ref["Y_rrr_prime"]*r**3 + ref["Y_vvr_prime"]*v*v*r + ref["Y_vrr_prime"]*v*r*r
        terms_n = ref["N_v_prime"]*v + ref["N_vvv_prime"]*v**3 + ref["N_r_prime"]*r + ref["N_rrr_prime"]*r**3 + ref["N_vvr_prime"]*v*v*r + ref["N_vrr_prime"]*v*r*r
        # At r!=0, HMRI's Y coefficient is the RHS combination containing
        # rigid Coriolis, while N is the physical fluid yaw moment.
        actual_y = float(result["rhs"][1] / FY if r else result["physical"][1] / FY)
        actual_n = float(result["physical"][5] / NM)
        held.append({"family": family, "beta_deg": beta, "r_prime": r,
            "HMRI_Y_prime": terms_y, "BCOD_Y_prime": actual_y,
            "Y_absolute_error_prime": abs(actual_y-terms_y),
            "Y_error_percent": 100*abs(actual_y-terms_y)/max(abs(terms_y), 1e-12),
            "HMRI_N_prime": terms_n, "BCOD_N_prime": actual_n,
            "N_absolute_error_prime": abs(actual_n-terms_n),
            "N_error_percent": 100*abs(actual_n-terms_n)/max(abs(terms_n), 1e-12),
            "comparison_definition": "RHS Y includes rigid Coriolis when r is nonzero; N is physical hull moment"})
    for beta in (-6., 6.): append_held("static_drift", beta, 0.)
    for r in (-.3, .3): append_held("pure_yaw", 0., r)
    for r in (-.3, .3): append_held("yaw_and_drift", 6., r)

    table = []
    for name, row in references.items():
        if name == "(m+m_y)_prime":
            continue
        value = records[name]
        table.append({"name": name, "HMRI_estimate": row["estimate"],
            "HMRI_low95": row["low95"], "HMRI_high95": row["high95"],
            "BCOD_fit": value,
            "error_percent": 100*abs(value-row["estimate"])/abs(row["estimate"]),
            "status": category(value, row)})
    freq = json.loads(FREQUENCY_FILE.read_text())["frequency_sweep"]
    mass = payload["mass_kg"]
    denominator = .5*RHO*LENGTH**2*DRAFT
    selected = np.asarray(payload["added_mass_kg"])[1,1]
    strip = json.loads((PACKAGE / "added_mass.json").read_text())["strip"]["matrix_6x6"][1][1]
    mass_values = {"normalizer_kg": denominator,
        "strip": (mass+strip)/denominator,
        "BEM_zero": (mass+selected)/denominator,
        "BEM_HMRI_frequency": (mass+next(x["A22_kg"] for x in freq if abs(x["frequency_hz"]-.065)<1e-9))/denominator,
        "HMRI": ref["(m+m_y)_prime"]}
    provenance = json.loads((PACKAGE / "provenance.json").read_text())
    out = {"schema": "bcod-hmri-passive-validation-v1", "package": str(PACKAGE),
        "frozen_generation_request_sha256": provenance["generation_request_sha256"],
        "original_geometry_sha256": provenance["original_geometry_sha256"],
        "hydrodynamic_mesh_sha256": provenance["hydrodynamic_mesh_sha256"],
        "source_provenance": provenance,
        "normalization": {"rho_kg_m3": RHO, "L_m": LENGTH, "T_m": DRAFT,
            "U_m_s": SPEED, "force_scale_N": FY, "moment_scale_Nm": NM,
            "v_prime": "v/U=-sin(beta)", "r_prime": "r*L/U"},
        "family_conditions": families, "coefficient_table": table,
        "coefficient_component_fits": component_fits,
        "held_out": held, "frequency_sweep": freq, "added_mass_comparison": mass_values,
        "summary": {"comparable_coefficients": len(table),
            "within_interval": sum(x["status"] == "inside HMRI 95% interval" for x in table),
            "median_error_percent": float(np.median([x["error_percent"] for x in table])),
            "worst_error_percent": float(max(x["error_percent"] for x in table)),
            "held_out_median_Y_error_percent": float(np.median([x["Y_error_percent"] for x in held])),
            "held_out_median_N_error_percent": float(np.median([x["N_error_percent"] for x in held]))},
        "campaign_wall_time_s": perf_counter()-start}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "validation.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
