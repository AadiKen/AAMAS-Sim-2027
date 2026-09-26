"""Seeded per-episode coefficient multipliers with stability-sign rejection."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .fit import CoefficientSurface


def stability_index(surface: CoefficientSurface, mass_kg: float, cg_x_m: float = 0.) -> float:
    m = mass_kg/(.5*surface.density_kg_m3*surface.length_m**3)
    x = (cg_x_m-surface.moment_reference_frd_m[0])/surface.length_m
    return surface.y[0]*(surface.n[1]-m*x)-surface.n[0]*(surface.y[1]-m)


def sample_episode(surface: CoefficientSurface, *, mass_kg: float,
                   seed: int, cg_x_m: float = 0., max_draws: int = 1000) -> tuple[CoefficientSurface, dict]:
    if mass_kg <= 0 or not np.isfinite([mass_kg, cg_x_m]).all() or max_draws <= 0:
        raise ValueError("Invalid episode mass, CG, or draw limit")
    rng = np.random.default_rng(seed)
    nominal = stability_index(surface, mass_kg, cg_x_m)
    for draw in range(max_draws):
        linear = rng.uniform(.8, 1.2)
        nonlinear = rng.uniform(.7, 1.3)
        added_mass = rng.uniform(.9, 1.1)
        y = tuple(z*(linear if i < 2 else nonlinear) for i, z in enumerate(surface.y))
        n = tuple(z*(linear if i < 2 else nonlinear) for i, z in enumerate(surface.n))
        candidate = replace(surface, y=y, n=n,
                            dx=tuple(z*nonlinear for z in surface.dx))
        sign = stability_index(candidate, mass_kg, cg_x_m)
        if np.sign(sign) == np.sign(nominal):
            return candidate, {"seed": seed, "draw": draw, "linear_multiplier": linear,
                               "nonlinear_multiplier": nonlinear,
                               "added_mass_multiplier": added_mass,
                               "stability_index": sign}
    raise ValueError("No stability-preserving episode sample")
