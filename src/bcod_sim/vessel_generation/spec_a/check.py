"""Plain-text sanity report for a fitted Spec A surface."""
from __future__ import annotations

from pathlib import Path
import math
from .empirical import clarke_linear
from .fit import CoefficientSurface


def sanity_checks(surface: CoefficientSurface, fit_diagnostics: dict, *,
                  beam_m: float, block_coefficient: float,
                  mass_kg: float, cg_x_m: float = 0., cases: list[dict] | None = None) -> dict:
    reference = clarke_linear(surface.length_m, beam_m, surface.draft_m, block_coefficient)
    linear = {"Y_v": surface.y[0], "Y_r": surface.y[1],
              "N_v": surface.n[0], "N_r": surface.n[1]}
    checks = {name: {"sign_match": linear[name]*reference[name] > 0,
                     "within_factor_two": .5 <= abs(linear[name]/reference[name]) <= 2.,
                     "model": linear[name], "clarke": reference[name]}
              for name in linear}
    m_prime = mass_kg/(.5*surface.density_kg_m3*surface.length_m**3)
    x_prime = (cg_x_m-surface.moment_reference_frd_m[0])/surface.length_m
    stability = (linear["Y_v"]*(linear["N_r"]-m_prime*x_prime) -
                 linear["N_v"]*(linear["Y_r"]-m_prime))
    loo = {k: fit_diagnostics[k]["loo_rmse"] for k in ("Y", "N")}
    deficient = {k: fit_diagnostics[k].get("rank_deficient_loo_folds", []) for k in ("Y", "N")}
    force_checks = {}
    if cases:
        straight = [row for row in cases if abs(float(row['v_mps'])) < 1e-10 and abs(float(row['r_rad_s'])) < 1e-10]
        anchor = [row for row in cases if abs(float(row['r_rad_s'])) < 1e-10 and
                  abs(math.degrees(math.atan2(-float(row['v_mps']),float(row['u_mps'])))-8) < 1e-6]
        for channel, key in (("Y", "Y_n"), ("N", "N_nm")):
            values = [float(row[key]) for row in cases]
            span = max(values)-min(values)
            max_loo = max(abs(value) for value in fit_diagnostics[channel]["loo_errors"])
            # LOO errors are nondimensional; normalize the measured span too.
            scale = (.5*surface.density_kg_m3*surface.length_m**2*
                     surface.reference_speed_mps**2*(surface.length_m if channel == "N" else 1.))
            force_checks[channel] = {"loo_max_below_15_percent_span":
                                     max_loo < .15*span/scale if span else False,
                                     "max_loo_prime": max_loo,
                                     "straight_below_1_percent_anchor": (
                                         abs(float(straight[0][key])) < .01*abs(float(anchor[0][key]))
                                         if straight and anchor else None)}
    review = (not all(z["sign_match"] and z["within_factor_two"] for z in checks.values()) or
              any(deficient.values()) or
              any(not row["loo_max_below_15_percent_span"] or
                  row["straight_below_1_percent_anchor"] is not True for row in force_checks.values()))
    return {"linear": checks, "stability_index": stability,
            "stability_sign": "positive" if stability > 0 else "negative" if stability < 0 else "zero",
            "loo_rmse": loo, "rank_deficient_loo_folds": deficient,
            "force_checks": force_checks, "requires_review": review}


def write_fit_report(path: Path, checks: dict, *, measured_core_hours: float | None = None) -> None:
    lines = ["# Spec A fit report", "", "Physical fluid-on-hull FRD force surface.", "",
             f"Review required: **{checks['requires_review']}**", "",
             f"Straight-line stability index: {checks['stability_index']:.8g} ({checks['stability_sign']})", "",
             "| Linear term | Model | Clarke | Sign | Within 2× |",
             "|---|---:|---:|---|---|"]
    for name, row in checks["linear"].items():
        lines.append(f"| {name} | {row['model']:.6g} | {row['clarke']:.6g} | "
                     f"{row['sign_match']} | {row['within_factor_two']} |")
    lines += ["", f"Y leave-one-out RMSE: {checks['loo_rmse']['Y']:.6g}",
              f"N leave-one-out RMSE: {checks['loo_rmse']['N']:.6g}", "",
              f"Rank-deficient leave-one-out folds: {checks['rank_deficient_loo_folds']}",
              f"Force-range checks: {checks['force_checks']}", "",
              "Original Clarke (1983) paper was not directly verified; this comparator follows Fossen's published implementation.",
              f"Measured compute: {measured_core_hours if measured_core_hours is not None else 'not measured'} core-hours.", ""]
    Path(path).write_text("\n".join(lines))


def plot_fit(path: Path, surface: CoefficientSurface, cases: list[dict], *, v5_surface=None) -> None:
    """Plot physical held-case loads against the selected fit and optional V5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    predicted = [surface.evaluate(*[float(row[key]) for key in ('u_mps','v_mps','r_rad_s')]) for row in cases]
    figure, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,index,key,label in zip(axes,(1,2),('Y_n','N_nm'),('Y (N)','N (N m)')):
        ax.plot(range(len(cases)),[float(row[key]) for row in cases],'o',label='CFD observations')
        ax.plot(range(len(cases)),[row[index] for row in predicted],'-',label='Selected fit')
        if v5_surface is not None:
            baseline=[v5_surface(*[float(row[key]) for key in ('u_mps','v_mps','r_rad_s')]) for row in cases]
            ax.plot(range(len(cases)),[row[index] for row in baseline],'--',label='V5')
        ax.set(xlabel='Case index',ylabel=label);ax.grid(alpha=.25);ax.legend()
    figure.savefig(path,dpi=150)
    plt.close(figure)
