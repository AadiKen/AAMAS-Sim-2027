"""Explicit external dimensional quantities to canonical SI."""

import math
from typing import Literal

from bcod_sim.core.errors import FrameConversionError

Dimension = Literal["length", "speed", "angle", "force", "moment"]

_TO_SI: dict[Dimension, dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0, "ft": 0.3048, "nmi": 1852.0},
    "speed": {"m/s": 1.0, "km/h": 1 / 3.6, "kn": 1852 / 3600},
    "angle": {"rad": 1.0, "deg": math.pi / 180},
    "force": {"N": 1.0, "kN": 1000.0},
    "moment": {"N*m": 1.0, "kN*m": 1000.0},
}


def to_si(value: float, *, dimension: Dimension, unit: str) -> float:
    if not math.isfinite(value):
        raise FrameConversionError("Quantity must be finite")
    try:
        factor = _TO_SI[dimension][unit]
    except KeyError as exc:
        raise FrameConversionError(f"Unsupported or missing {dimension} unit: {unit}") from exc
    return value * factor
