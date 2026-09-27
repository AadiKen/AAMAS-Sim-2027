"""Reprocess saved KCS pure-yaw cases; never invokes OpenFOAM."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from bcod_sim.vessel_generation.spec_a.fit import surface_from_payload
from bcod_sim.vessel_generation.spec_a.harmonic_postprocess import (
    body_kinematics, harmonic, interpolate_motion, read_forces, read_motion,
    world_to_body_cg,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "stage3_results/spec_a/kcs/coarse_first"
OUT = BASE / "harmonic_yaw_postprocess_audit"
L, T, U, RHO, FREQ = 5.75, .270, 1.953, 1025., .08
PERIOD = 1/FREQ
NSCALE = .5*RHO*L**2*T*U**2
EFD_NR, EFD_NRRR = -.0462, -.0313


def analyse_case(case: Path, windows: list[tuple[float, float]], surface):
    raw = read_forces(case)
    t = raw[:, 0]
    motion = read_motion(case)
    translation, theta, vy, theta_dot = interpolate_motion(motion, t)
    u, v, r = body_kinematics(theta, vy, theta_dot, (-U, 0., 0.))
    transformed = np.array([world_to_body_cg(row[1:4], row[4:7],
        origin_world=(0., 0., 0.), cg_world=trans, yaw_foam_rad=angle)
        for row, trans, angle in zip(raw, translation, theta)])
    y = transformed[:, 0, 1]
    n = transformed[:, 1, 2]
    mrf = np.asarray([surface.evaluate(float(uu), float(vv), float(rr))[1:]
                      for uu, vv, rr in zip(u, v, r)])
    # Keep each cycle and the actual samples for the two-amplitude reconstruction.
    cycles, cache = [], []
    for start, end in windows:
        mask = (t >= start-1e-8) & (t <= end+1e-8)
        ts = t[mask]
        if len(ts) < 100 or ts[0] > start+.015 or ts[-1] < end-.015 or np.max(np.diff(ts)) > .015:
            raise ValueError(f"Incomplete cycle {case}: {start}-{end}")
        rec = {"start_s": start, "end_s": end, "samples": len(ts),
               "dt_min_s": float(np.diff(ts).min()), "dt_median_s": float(np.median(np.diff(ts))),
               "dt_max_s": float(np.diff(ts).max()),
               "u_min_mps": float(u[mask].min()), "u_max_mps": float(u[mask].max()),
               "v_abs_max_mps": float(np.abs(v[mask]).max()),
               "r_abs_max_rad_s": float(np.abs(r[mask]).max()),
               "rprime_paper_abs_max": float(np.abs(r[mask]*L/U).max()),
               "rprime_local_abs_max": float(np.abs(r[mask]*L/np.hypot(u[mask], v[mask])).max())}
        for name, arr in (("Y", y), ("N", n), ("Y_fixed_origin_world_converted_sign", -raw[:, 2]),
                          ("N_fixed_origin_world_converted_sign", -raw[:, 6]),
                          ("MRF_Y", mrf[:, 0]), ("MRF_N", mrf[:, 1])):
            rec[name] = harmonic(ts, arr[mask], FREQ)
        cycles.append(rec)
        cache.append({"t": ts, "N": n[mask], "Y": y[mask], "rprime": r[mask]*L/U,
                      "r": r[mask], "u": u[mask], "v": v[mask]})
    return cycles, cache


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load((BASE / "coefficients.yaml").read_text())
    surface = surface_from_payload(payload)
    cases = {
        "r02_baseline": BASE / "diagnostic_pmm_yaw_0_+0.2",
        "r02_refined": BASE / "diagnostic_pmm_yaw_0_+0.2/timestep_replay",
        "r04": BASE / "diagnostic_pmm_yaw_0_+0.4",
    }
    windows = {"r02_baseline": [(6.25, 18.75), (18.75, 31.25)],
               "r02_refined": [(6.25, 18.75), (18.75, 31.25)],
               "r04": [(6.25, 18.75), (18.75, 31.25)]}
    result = {"status": "BENCHMARK_MAPPING_BLOCKED", "frequency_hz": FREQ,
              "frequency_status": "inferred_not_published", "force_origin_world_m": [0, 0, 0],
              "case_cg_body_frd_m": [0, 0, 0], "output_frame": "foam_world_xyz",
              "converted_frame": "body_frd_at_moving_case_cg",
              "load_kind": "physical_fluid_on_hull", "normalization": "paper_initial_U_1.953_mps",
              "N_scale_Nm": NSCALE, "reference_draft_m": T,
              "paper_table_printed_draft_m": .207,
              "unresolved": ["experimental_CG_coordinates_and_moment_origin",
                             "experimental_load_inertial_correction_mapping",
                             "experimental_harmonic_phase_sign",
                             "pure_yaw_frequency_exact_value", "printed_model_draft_0.207_vs_fullscale_divided_40_0.270"],
              "earlier_4_35_percent_EFD_agreement": "SUPERSEDED", "cases": {}}
    caches = {}
    for key, case in cases.items():
        cycles, cache = analyse_case(case, windows[key], surface)
        result["cases"][key] = {"source_case": str(case), "cycles": cycles}
        caches[key] = cache
    b = result["cases"]["r02_baseline"]["cycles"][-1]
    refined = result["cases"]["r02_refined"]["cycles"][-1]
    for field in ("N", "Y"):
        bb, rr = b[field], refined[field]
        result.setdefault("timestep_sensitivity", {})[field] = {
            "rate_change_pct": 100*abs(rr["rate_1"]-bb["rate_1"])/abs(bb["rate_1"]),
            "amplitude_change_pct": 100*abs(rr["amplitude_1"]-bb["amplitude_1"])/bb["amplitude_1"],
            "phase_change_deg": (rr["phase_1_deg"]-bb["phase_1_deg"]+180)%360-180,
            "refined_cycle_repeatability_pct": 100*abs(rr["amplitude_1"]-
                result["cases"]["r02_refined"]["cycles"][0][field]["amplitude_1"])/
                result["cases"]["r02_refined"]["cycles"][0][field]["amplitude_1"]}
    # Compare EFD and MRF as first harmonics on the same actual cycle. EFD is conditional.
    for key, amplitude in (("r02_refined", .2), ("r04", .4)):
        rec = result["cases"][key]["cycles"][-1]
        motion = caches[key][-1]
        rp = motion["rprime"]
        efd_series = NSCALE*(EFD_NR*rp+EFD_NRRR*rp**3)
        efd = harmonic(motion["t"], efd_series, FREQ)
        rec["EFD_conditional_N"] = efd
        rec["EFD_rate_absolute_error_Nm"] = abs(rec["N"]["rate_1"]-efd["rate_1"])
        rec["EFD_rate_relative_error_pct"] = 100*rec["EFD_rate_absolute_error_Nm"]/abs(efd["rate_1"])
        rec["MRF_vs_EFD_rate_absolute_error_Nm"] = abs(rec["MRF_N"]["rate_1"]-efd["rate_1"])
    # Identify c1/c3 using actual r'(t), with the same extractor used for loads.
    targets, matrix = [], []
    for key in ("r02_refined", "r04"):
        cycle = caches[key][-1]
        targets.append(result["cases"][key]["cycles"][-1]["N"]["rate_1"])
        matrix.append([NSCALE*harmonic(cycle["t"], cycle["rprime"], FREQ)["rate_1"],
                       NSCALE*harmonic(cycle["t"], cycle["rprime"]**3, FREQ)["rate_1"]])
    c1, c3 = np.linalg.solve(matrix, targets)
    result["provisional_coefficients"] = {"N_r_prime": float(c1), "N_rrr_prime": float(c3),
        "identification": "two_amplitudes_only_no_independent_validation"}
    for key in ("r02_refined", "r04"):
        cycle = caches[key][-1]
        measured = result["cases"][key]["cycles"][-1]["N"]
        predicted_rate = NSCALE*(c1*cycle["rprime"]+c3*cycle["rprime"]**3)
        # Measured acceleration phase is kept separate from the rate polynomial.
        phase = 2*math.pi*FREQ*cycle["t"]
        predicted = predicted_rate-measured["accel_1"]*np.sin(phase)
        pred_h = harmonic(cycle["t"], predicted, FREQ)
        rate_h = harmonic(cycle["t"], predicted_rate, FREQ)
        residual = cycle["N"]-predicted
        result["cases"][key]["cycles"][-1]["reconstruction"] = {
            "rate_polynomial_harmonics": rate_h, "predicted_waveform_harmonics": pred_h,
            "third_rate_absolute_error_Nm": abs(measured["rate_3"]-pred_h["rate_3"]),
            "third_accel_absolute_error_Nm": abs(measured["accel_3"]-pred_h["accel_3"]),
            "waveform_rms_error_Nm": float(np.sqrt(np.mean(residual**2))),
            "first_rate_absolute_error_Nm": abs(measured["rate_1"]-pred_h["rate_1"])}
    (OUT / "corrected_harmonics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"artifact": str(OUT / "corrected_harmonics.json"),
                      "timestep": result["timestep_sensitivity"],
                      "coefficients": result["provisional_coefficients"],
                      "last_cycles": {k: v["cycles"][-1] for k, v in result["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
