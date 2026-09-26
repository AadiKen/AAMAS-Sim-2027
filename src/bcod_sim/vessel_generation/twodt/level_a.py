"""Literature-driven sectional transverse-drag age for captive states.

Rabliås & Kristiansen (2021), Eqs. 16--21, Table 1. This candidate replaces
V5's sectional Y/N provider with full 2D+t separated drag plus the existing
geometry-derived linear lift. Added mass/Coriolis and other plant terms stay
with their separate providers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

_POLYNOMIAL = np.array([2.481e-7, -3.647e-5, 1.906e-3, -4.417e-2,
                        4.315e-1, 7.339e-2], dtype=float)
_FIT_LIMIT = 25.0


def cylinder_startup_ratio(t_star: np.ndarray) -> np.ndarray:
    """Eq. 21 / steady 1.2, with a bounded numerical evaluation domain.

    Outside the observed startup interval, hold the last fitted value. The
    resulting 1.05% excess over unity is disclosed, not hidden by a jump.
    """
    value = np.asarray(t_star, dtype=float)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Nonnegative finite nondimensional age required")
    return np.polyval(_POLYNOMIAL, np.minimum(value, _FIT_LIMIT)) / 1.2


def _arrays(stations: list[dict]) -> dict[str, np.ndarray]:
    if len(stations) < 3:
        raise ValueError("At least three sections required")
    names = {"x": "x_m", "y": "y_m", "draft": "draft_m", "dx": "dx_m",
             "cd": "cd", "lift": "lift_base_kg_per_m"}
    data = {name: np.array([row[key] for row in stations], dtype=float)
            for name, key in names.items()}
    if (not all(np.isfinite(v).all() for v in data.values()) or
        any((data[k] <= 0).any() for k in ("draft", "dx", "cd")) or
        (data["lift"] < 0).any()):
        raise ValueError("Invalid sectional data")
    data["hull"] = np.array([row["hull_id"] for row in stations], dtype=int)
    return data


def sectional_v5_arrays(stations: list[dict], u: float, v: float, r: float,
                        *, density: float = 1025., reference_x: float = 0.) -> dict:
    """Independent NumPy reproduction of frozen translation/shear V5 Y/N."""
    if not all(math.isfinite(z) for z in (u, v, r, density, reference_x)) or density <= 0:
        raise ValueError("Invalid velocity, density, or reference")
    a = _arrays(stations)
    q = v + (a["x"] - reference_x) * r
    local_u = u - a["y"] * r
    projected = a["draft"] * a["dx"]
    mean_q = float(np.sum(projected * q) / np.sum(projected))
    mean_u = float(np.sum(projected * np.abs(local_u)) / np.sum(projected))
    denom = mean_u**2 + mean_q**2
    w = mean_q**2 / denom if denom else 0.
    shear = q - mean_q
    drag = -.5 * density * a["cd"] * projected * q * (w * np.abs(q) + (1-w) * np.abs(shear))
    lift = -(1-w) * a["lift"] * np.abs(local_u) * q
    force = drag + lift
    arm = a["x"] - reference_x
    return {"x_m": a["x"], "q_mps": q, "u_local_mps": local_u,
            "cd_steady": a["cd"], "draft_m": a["draft"], "dx_m": a["dx"],
            "hull_id": a["hull"], "drag_n": drag, "lift_n": lift,
            "dY_n": force, "dN_nm": arm * force, "Y_n": float(force.sum()),
            "N_nm": float((arm*force).sum()), "translation_weight": w,
            "cumulative_Y_n": np.cumsum(force[np.argsort(a["x"])]),
            "cumulative_N_nm": np.cumsum((arm*force)[np.argsort(a["x"])])}


@dataclass(frozen=True)
class SectionalTwoDt:
    stations: list[dict]
    density: float = 1025.
    reference_x: float = 0.

    def steady_captive(self, u: float, v: float, r: float) -> dict:
        """Eq. 16--21 steady captive load with additive linear lift."""
        base = sectional_v5_arrays(self.stations, u, v, r, density=self.density,
                                   reference_x=self.reference_x)
        x, local_u, q = base["x_m"], base["u_local_mps"], base["q_mps"]
        age = np.zeros_like(x)
        valid = np.ones_like(x, dtype=bool)
        hull_ids = base["hull_id"]
        for hull_id in np.unique(hull_ids):
            members = hull_ids == hull_id
            bow = float(np.max(x[members] + base["dx_m"][members]/2))
            distance = bow - x[members]
            numerator = v*distance + .5*r*((bow-self.reference_x)**2 -
                                           (x[members]-self.reference_x)**2)
            valid[members] = local_u[members] > 0.
            good = valid[members]
            local_age = np.zeros(np.sum(members), dtype=float)
            local_age[good] = np.abs(numerator[good]/local_u[members][good]) / base["draft_m"][members][good]
            age[members] = local_age
        if not valid.all():
            raise ValueError("2D+t forward-plane mapping invalid at zero/reverse local axial speed")
        ratio = cylinder_startup_ratio(age)
        geometry = _arrays(self.stations)
        drag = (-.5 * self.density * geometry["cd"] * ratio *
                geometry["draft"] * geometry["dx"] * q * np.abs(q))
        # The paper's cross-flow term is additive to the non-viscous terms
        # in its modular model. Keep the existing geometry-derived linear
        # lift separate; do not also add the V5 cross-flow/blend result.
        lift = -geometry["lift"] * np.abs(local_u) * q
        force = drag + lift
        arm = x - self.reference_x
        order = np.argsort(x)
        return {**base, "method": "rablias_kristiansen_2021_eq19_21_level_a",
                "t_star": age, "cd_ratio": ratio, "cd_effective": base["cd_steady"]*ratio,
                "drag_n": drag, "lift_n": lift, "dY_n": force, "dN_nm": arm*force,
                "Y_n": float(force.sum()), "N_nm": float((arm*force).sum()),
                "cumulative_Y_n": np.cumsum(force[order]),
                "cumulative_N_nm": np.cumsum((arm*force)[order])}
