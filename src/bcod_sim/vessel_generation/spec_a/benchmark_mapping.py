"""Strict admission checks for saved KCS benchmark comparisons.

This module makes no experimental or CFD prediction. It only rejects mappings
whose physical load and reference definitions are incomplete or incompatible.
"""
from __future__ import annotations

import math


REQUIRED = ("reference_point_m", "frame", "sign_convention", "normalization",
            "motion_definition", "load_definition", "vessel_configuration")


def _unresolved(value: object) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def draft_normalization_ratio(draft_a_m: float, draft_b_m: float) -> float:
    """Ratio of N' evaluated with draft a to N' evaluated with draft b."""
    if not all(math.isfinite(x) and x > 0 for x in (draft_a_m, draft_b_m)):
        raise ValueError("Positive draft required")
    return draft_b_m / draft_a_m


def assert_comparable(cfd: dict, reference: dict, *, kind: str) -> None:
    """Require complete matching metadata before comparing any loads."""
    if kind not in ("steady_cmt", "harmonic_pmm"):
        raise ValueError("Unknown comparison family")
    for name, record in (("CFD", cfd), ("reference", reference)):
        missing = [field for field in REQUIRED if _unresolved(record.get(field))]
        if kind == "harmonic_pmm":
            missing += [field for field in ("frequency_hz", "phase_convention", "amplitude_r_prime")
                        if _unresolved(record.get(field))]
        if missing:
            raise ValueError(f"{name} missing: {', '.join(missing)}")
        if record.get("comparison_family") != kind:
            raise ValueError(f"{name} is not {kind}")
    for field in REQUIRED[1:]:
        if cfd[field] != reference[field]:
            raise ValueError(f"Incompatible {field}")
    if len(cfd["reference_point_m"]) != 3 or len(reference["reference_point_m"]) != 3:
        raise ValueError("Reference point must have three coordinates")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x)
           for point in (cfd["reference_point_m"], reference["reference_point_m"])
           for x in point):
        raise ValueError("Reference point must be finite and numeric")
    if any(abs(a-b) > 1e-9 for a, b in zip(cfd["reference_point_m"], reference["reference_point_m"])):
        raise ValueError("Incompatible moment reference point")
    if kind == "harmonic_pmm":
        if any(not isinstance(record["amplitude_r_prime"], (int, float))
               or not math.isfinite(record["amplitude_r_prime"])
               or record["amplitude_r_prime"] <= 0 for record in (cfd, reference)):
            raise ValueError("Harmonic amplitude must be positive")
        if not math.isclose(cfd["amplitude_r_prime"], reference["amplitude_r_prime"], rel_tol=1e-6):
            raise ValueError("Incompatible harmonic amplitude")
        if any(not isinstance(record["frequency_hz"], (int, float))
               or record["frequency_hz"] <= 0 for record in (cfd, reference)):
            raise ValueError("Harmonic frequency must be positive")
        if not math.isclose(cfd["frequency_hz"], reference["frequency_hz"], rel_tol=1e-6):
            raise ValueError("Incompatible harmonic frequency")
        if cfd["phase_convention"] != reference["phase_convention"]:
            raise ValueError("Incompatible harmonic phase convention")
