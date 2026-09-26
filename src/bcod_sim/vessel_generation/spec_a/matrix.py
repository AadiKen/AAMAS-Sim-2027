"""Deterministic ten-state captive matrix and Froude eligibility."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CaseState:
    beta_deg: float
    r_prime: float
    family: str

    def body_velocity(self, speed_mps: float, length_m: float) -> tuple[float, float, float]:
        if (speed_mps <= 0 or length_m <= 0 or not all(math.isfinite(z) for z in
            (speed_mps, length_m, self.beta_deg, self.r_prime))):
            raise ValueError("Positive speed and length required")
        beta = math.radians(self.beta_deg)
        return (speed_mps * math.cos(beta), -speed_mps * math.sin(beta),
                self.r_prime * speed_mps / length_m)


def case_matrix(*, max_beta_deg: float = 16., max_abs_r_prime: float = .6) -> tuple[CaseState, ...]:
    if (not all(math.isfinite(z) for z in (max_beta_deg, max_abs_r_prime)) or
        max_beta_deg <= 0 or max_abs_r_prime <= 0):
        raise ValueError("Positive intended operating envelope required")
    raw = [(0., 0., "straight")]
    # Scale the fixed design as a whole when the envelope is smaller. Clipping
    # each state independently would duplicate rows and destroy fit rank.
    bscale, rscale = min(1., max_beta_deg/16), min(1., max_abs_r_prime/.6)
    raw += [(b*bscale, 0., "drift") for b in (4., 8., 12., 16.)]
    raw += [(0., r*rscale, "yaw") for r in (.2, .4, .6)]
    raw += [(8.*bscale, .3*rscale, "combined"), (8.*bscale, -.3*rscale, "combined")]
    return tuple(CaseState(*row) for row in raw)


def froude_gate(speed_mps: float, length_m: float, gravity_mps2: float = 9.80665) -> dict:
    if speed_mps < 0 or length_m <= 0 or gravity_mps2 <= 0 or not all(
        math.isfinite(x) for x in (speed_mps, length_m, gravity_mps2)):
        raise ValueError("Invalid speed, length, or gravity")
    fr = speed_mps / math.sqrt(gravity_mps2 * length_m)
    if fr < .30:
        status, effective = "eligible", fr
    elif fr <= .45:
        status, effective = "extrapolated_fr", .30
    else:
        status, effective = "out_of_envelope", None
    return {"fr_l": fr, "status": status, "effective_cfd_fr": effective}
