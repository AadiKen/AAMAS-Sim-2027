"""Geometry-only mesh budgets and qualified sentinel selection (no EFD input)."""
from __future__ import annotations
import math

PROFILES = {
    "fast": {"background_cells": 150_000, "target_final_cells": [300_000, 700_000]},
    "standard": {"background_cells": 800_000, "target_final_cells": [1_000_000, 1_500_000]},
    "reference": {"background_cells": 1_800_000, "target_final_cells": [2_000_000, 3_000_000]},
}


def background_grid(extents, budget):
    if budget < 1 or any(not math.isfinite(x) or x <= 0 for x in extents):
        raise ValueError("Positive finite domain and cell budget required")
    spacing = (math.prod(extents)/budget)**(1/3)
    counts = tuple(max(1, math.floor(x/spacing+1e-10)) for x in extents)
    if math.prod(counts) > budget:
        raise ValueError("Budget too small for domain aspect ratio")
    return counts, spacing


def normalized_difference(fast, standard, floor):
    if not all(math.isfinite(x) for x in (fast, standard, floor)) or floor <= 0:
        raise ValueError("Finite loads and positive scale floor required")
    return abs(fast-standard)/max(abs(standard), floor)


def select_profile(fast, standard, *, force_floor, moment_floor):
    for result in (fast, standard):
        if result["status"] not in ("converged", "oscillatory"):
            raise ValueError("Both sentinels must pass existing force qualification")
    yf, nf = fast["qualification"]["mean_foam"][1::4]
    ys, ns = standard["qualification"]["mean_foam"][1::4]
    ey = normalized_difference(yf, ys, force_floor)
    en = normalized_difference(nf, ns, moment_floor)
    return {"mesh_profile_selected": "fast" if max(ey, en) <= .10+1e-12 else "standard",
            "fast_cells": fast["cell_count"], "standard_cells": standard["cell_count"],
            "sentinel_Y_difference": ey, "sentinel_N_difference": en,
            "sentinel_threshold": .10, "force_scale_floor_n": force_floor,
            "moment_scale_floor_nm": moment_floor}
