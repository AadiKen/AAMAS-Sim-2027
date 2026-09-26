"""Small symmetric state design for a future 3D viscous Y/N response map."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True, order=True)
class CaptiveState:
    """FRD state; v/U and rL/U are signed nondimensional coordinates."""

    v_over_u: float
    r_l_over_u: float

    def __post_init__(self) -> None:
        if not all(isfinite(z) for z in (self.v_over_u, self.r_l_over_u)):
            raise ValueError("Finite captive coordinates required")


def initial_states() -> tuple[CaptiveState, ...]:
    """Minimal symmetric drift, yaw, and interaction probes.

    Angles/rates are generic experiment-design choices, not benchmark fits.
    Opposite signs permit odd-symmetry checks; diagonals isolate coupling.
    """
    drift = 0.10
    yaw = 0.10
    return tuple(CaptiveState(v, r) for v, r in (
        (0., 0.), (drift, 0.), (-drift, 0.),
        (0., yaw), (0., -yaw),
        (drift, yaw), (-drift, -yaw),
        (drift, -yaw), (-drift, yaw),
    ))


def select_next_state(
    candidate_uncertainty: Mapping[CaptiveState, float],
    completed: set[CaptiveState],
) -> CaptiveState | None:
    """Choose the highest uncertainty unevaluated state, deterministically.

    The caller supplies uncertainty from a response interpolator or leave-one-
    out validation. No measured benchmark coefficient enters this selection.
    """
    available = []
    for state, uncertainty in candidate_uncertainty.items():
        if not isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("Finite nonnegative uncertainty required")
        if state not in completed:
            available.append((uncertainty, state))
    if not available:
        return None
    return sorted(available, key=lambda item: (-item[0], item[1]))[0][1]
