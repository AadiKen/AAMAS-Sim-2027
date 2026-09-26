"""Symmetric captive Y/N and resistance-preserving surge increment fit."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import yaml


BASES = {
    "cubic": ("v", "r", "vvv", "vvr", "vrr", "rrr"),
    "absolute": ("v", "r", "v_abs_v", "v_abs_r", "r_abs_v", "r_abs_r"),
}


def terms(v: float, r: float, basis: str) -> np.ndarray:
    if basis == "cubic":
        return np.array([v, r, v**3, v*v*r, v*r*r, r**3], float)
    if basis == "absolute":
        return np.array([v, r, v*abs(v), v*abs(r), r*abs(v), r*abs(r)], float)
    raise ValueError("Unknown maneuvering basis")


def _fit_channel(states: np.ndarray, values: np.ndarray) -> tuple[str, np.ndarray, dict]:
    # Signed mirror states impose odd symmetry without adding new CFD cases.
    mirrored = np.vstack((states, -states))
    observed = np.concatenate((values, -values))
    fits = {}
    for basis in BASES:
        design = np.array([terms(v, r, basis) for v, r in mirrored])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        coeff = np.linalg.lstsq(design, observed, rcond=None)[0]
        errors = []
        deficient_folds = []
        for i in range(len(states)):
            keep = np.ones(len(states), dtype=bool)
            keep[i] = False
            trial = np.vstack((states[keep], -states[keep]))
            target = np.concatenate((values[keep], -values[keep]))
            matrix = np.array([terms(v, r, basis) for v, r in trial])
            if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
                deficient_folds.append(i)
            estimate = np.linalg.lstsq(matrix, target, rcond=None)[0]
            errors.append(float(terms(*states[i], basis) @ estimate - values[i]))
        if errors:
            fits[basis] = (float(np.sqrt(np.mean(np.square(errors)))), coeff,
                           {"loo_errors": errors, "rank_deficient_loo_folds": deficient_folds,
                            "residuals": (design @ coeff - observed).tolist()})
    if not fits:
        raise ValueError("Captive matrix is rank deficient under leave-one-out validation")
    choice = min(fits, key=lambda key: (fits[key][0], key))
    score, coeff, diagnostics = fits[choice]
    return choice, coeff, {"loo_rmse": score, **diagnostics,
                          "candidate_loo_rmse": {name: value[0] for name, value in fits.items()}}


@dataclass(frozen=True)
class CoefficientSurface:
    length_m: float
    draft_m: float
    reference_speed_mps: float
    density_kg_m3: float
    y_basis: str
    n_basis: str
    y: tuple[float, ...]
    n: tuple[float, ...]
    dx: tuple[float, float, float]
    moment_reference_frd_m: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if (not all(math.isfinite(z) and z > 0 for z in (self.length_m, self.draft_m,
                self.reference_speed_mps, self.density_kg_m3)) or
            self.y_basis not in BASES or self.n_basis not in BASES):
            raise ValueError("Invalid surface reference quantities or basis")
        for name, count in (("y", 6), ("n", 6), ("dx", 3), ("moment_reference_frd_m", 3)):
            values = tuple(float(z) for z in getattr(self, name))
            if len(values) != count or not all(math.isfinite(z) for z in values):
                raise ValueError("Invalid surface coefficients")
            object.__setattr__(self, name, values)

    def evaluate(self, u: float, v: float, r: float) -> tuple[float, float, float]:
        """Physical FRD load (ΔX,Y,N); baseline R(u) is external."""
        if not all(math.isfinite(z) for z in (u, v, r)):
            raise ValueError("Finite velocity required")
        speed = math.hypot(u, v)
        if speed <= 0:
            return 0., 0., 0.
        vp, rp = v/speed, r*self.length_m/speed
        xprime = self.dx[0]*vp*vp + self.dx[1]*rp*rp + self.dx[2]*vp*rp
        yprime = float(terms(vp, rp, self.y_basis) @ np.asarray(self.y))
        nprime = float(terms(vp, rp, self.n_basis) @ np.asarray(self.n))
        scale = .5*self.density_kg_m3*self.length_m**2*speed**2
        return scale*xprime, scale*yprime, scale*self.length_m*nprime


def fit_cases(rows: Sequence[Mapping[str, float]], *, length_m: float, draft_m: float,
              reference_speed_mps: float, density_kg_m3: float = 1025.,
              moment_reference_frd_m: tuple[float, float, float] = (0.,0.,0.)) -> tuple[CoefficientSurface, dict]:
    """Fit physical fluid-on-hull FRD observations; never resisting loads."""
    if not all(math.isfinite(z) and z > 0 for z in
               (length_m, draft_m, reference_speed_mps, density_kg_m3)):
        raise ValueError("Positive reference quantities required")
    if len({row.get("mesh_profile") for row in rows}) > 1:
        raise ValueError("Cannot fit mixed mesh profiles")
    states, ys, ns, xs = [], [], [], []
    straight = {}
    for row in rows:
        if row.get("convergence_status", "converged") not in ("converged", "oscillatory"):
            raise ValueError("Failed CFD observation cannot enter the fit")
        if not all(math.isfinite(float(row[key])) for key in
                   ("u_mps", "v_mps", "r_rad_s", "X_n", "Y_n", "N_nm")):
            raise ValueError("Nonfinite CFD observation")
        u, v, r = (float(row[k]) for k in ("u_mps", "v_mps", "r_rad_s"))
        speed = math.hypot(u, v)
        if speed <= 0 or u <= 0:
            raise ValueError("Only forward captive states can be fitted")
        scale = .5*density_kg_m3*length_m**2*speed**2
        vp, rp = v/speed, r*length_m/speed
        if abs(vp) < 1e-12 and abs(rp) < 1e-12:
            straight[round(speed, 9)] = float(row["X_n"])
        states.append((vp, rp))
        ys.append(float(row["Y_n"])/scale)
        ns.append(float(row["N_nm"])/(scale*length_m))
        xs.append(float(row["X_n"]))
    state = np.asarray(states, float)
    if len(state) < 7 or not straight:
        raise ValueError("Identifiable matrix and at least one straight CFD case required")
    y_basis, y, ydiag = _fit_channel(state, np.asarray(ys))
    n_basis, n, ndiag = _fit_channel(state, np.asarray(ns))
    # ΔX is referenced to a straight case at the same speed. A single-speed
    # matrix is required; fitting across speeds needs matched straight cases.
    speeds = np.array([math.hypot(float(z["u_mps"]), float(z["v_mps"])) for z in rows])
    if max(speeds)-min(speeds) > 1e-6*reference_speed_mps:
        raise ValueError("Spec A fit requires one speed with a matched straight case")
    if not np.allclose(speeds, reference_speed_mps, rtol=1e-6, atol=0):
        raise ValueError("Reference speed does not match the CFD matrix")
    baseline = straight.get(round(float(speeds[0]), 9))
    if baseline is None:
        raise ValueError("Missing straight CFD force at matrix speed")
    x_design = np.array([[v*v, r*r, v*r] for v, r in state])
    x_values = (np.asarray(xs)-baseline)/(.5*density_kg_m3*length_m**2*speeds**2)
    if np.linalg.matrix_rank(x_design) < 3:
        raise ValueError("Surge increment matrix is rank deficient")
    dx = np.linalg.lstsq(x_design, x_values, rcond=None)[0]
    surface = CoefficientSurface(length_m, draft_m, reference_speed_mps, density_kg_m3,
                                 y_basis, n_basis, tuple(y), tuple(n), tuple(dx), moment_reference_frd_m)
    return surface, {"Y": ydiag, "N": ndiag,
                     "surge_residuals": (x_design @ dx - x_values).tolist(),
                     "case_count": len(rows), "force_convention": "physical_fluid_on_hull_FRD"}


def write_coefficients_yaml(path, surface: CoefficientSurface, *, froude: dict,
                            source: str = "cfd", mesh_selection: dict | None = None) -> None:
    from dataclasses import asdict
    if source not in ("v5", "empirical", "cfd", "sysid"):
        raise ValueError("Invalid coefficient source")
    payload = {"schema": "bcod-maneuvering-spec-a-v1",
               "force_convention": "physical_fluid_on_hull_FRD",
               "load_contract": "M_A acceleration only; C_A excluded while surface active",
               "coefficients": asdict(surface), "froude_gate": froude,
               "term_sources": {name: source for name in (
                   *(f"Y_{name}" for name in BASES[surface.y_basis]),
                   *(f"N_{name}" for name in BASES[surface.n_basis]),
                   "X_vv", "X_rr", "X_vr")}}
    if mesh_selection is not None:
        payload["mesh_selection"] = mesh_selection
    from pathlib import Path
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))


def surface_from_payload(payload: dict) -> CoefficientSurface:
    """Reject force-sign ambiguity at the runtime package boundary."""
    if (payload.get("schema") != "bcod-maneuvering-spec-a-v1" or
        payload.get("force_convention") != "physical_fluid_on_hull_FRD"):
        raise ValueError("Maneuvering surface requires explicit physical fluid-on-hull FRD contract")
    if payload.get("froude_gate", {}).get("status") == "out_of_envelope":
        raise ValueError("Out-of-envelope CFD surface cannot be installed")
    return CoefficientSurface(**payload["coefficients"])
