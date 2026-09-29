"""Strict admission checks for saved KCS benchmark comparisons.

This module makes no experimental or CFD prediction. It only rejects mappings
whose physical load and reference definitions are incomplete or incompatible.
"""
from __future__ import annotations

import math


REQUIRED = ("reference_point_m", "frame", "force_frame", "sign_convention", "normalization",
            "motion_definition", "load_definition", "vessel_configuration")

FORCE_FRAMES = ("body_frd", "tow_track")


def body_to_tow_force(x_body: float, y_body: float, beta_deg: float) -> tuple[float, float]:
    """Rotate body FRD force into the towing-track frame at positive drift.

    Positive beta means body sway velocity is negative. Tow-track +X follows
    vessel travel and +Y points starboard of that track.
    """
    if not all(math.isfinite(v) for v in (x_body, y_body, beta_deg)):
        raise ValueError("Finite force and drift required")
    beta = math.radians(beta_deg)
    c, s = math.cos(beta), math.sin(beta)
    return c*x_body-s*y_body, s*x_body+c*y_body


def normalize_static_drift(y_newton: float, n_newton_m: float, *, rho: float,
                           speed: float, length: float, draft: float) -> tuple[float, float]:
    """NMRI CY/CN convention, with the supplied force already in its declared frame."""
    if not all(math.isfinite(v) for v in (y_newton, n_newton_m, rho, speed, length, draft)):
        raise ValueError("Finite load and reference properties required")
    if min(rho, speed, length, draft) <= 0:
        raise ValueError("Positive reference properties required")
    y_denom = .5*rho*speed**2*length*draft
    return y_newton/y_denom, n_newton_m/(y_denom*length)


def shift_yaw_moment(n_about_a: float, x_force: float, y_force: float,
                     a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
    """Shift physical yaw moment A→B with (A−B)×F in one common frame."""
    values = (n_about_a, x_force, y_force, *a_xy, *b_xy)
    if len(a_xy) != 2 or len(b_xy) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Finite planar force and origins required")
    return n_about_a + (a_xy[0]-b_xy[0])*y_force - (a_xy[1]-b_xy[1])*x_force


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
        if record["force_frame"] not in FORCE_FRAMES:
            raise ValueError(f"{name} has unsupported force frame")
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
