"""Stage 3B-ready passive parameter hooks; no optimizer or actuator fit."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

SCALES = ("surge_resistance_scale", "crossflow_cd_scale", "linear_sway_scale",
          "linear_yaw_scale", "added_mass_scale", "roll_damping_scale")


def apply_passive_scales(runtime_payload: dict, scales: dict[str, float]) -> dict:
    unknown = set(scales) - set(SCALES)
    if unknown:
        raise ValueError(f"Unknown passive calibration scales: {sorted(unknown)}")
    if any(not math.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("Passive calibration scales must be finite and positive")
    payload = deepcopy(runtime_payload)
    if "surge_resistance_scale" in scales:
        payload["surge_resistance"]["force_x_n"] = [
            x * scales["surge_resistance_scale"] for x in payload["surge_resistance"]["force_x_n"]]
    if "crossflow_cd_scale" in scales:
        flow = payload["crossflow"]
        flow["cd_scale"] = flow.get("cd_scale", 1.) * scales["crossflow_cd_scale"]
    if "added_mass_scale" in scales:
        matrix = np.asarray(payload["added_mass_kg"], dtype=float) * scales["added_mass_scale"]
        payload["added_mass_kg"] = matrix.tolist()
    if any(key in scales for key in ("linear_sway_scale", "linear_yaw_scale", "roll_damping_scale")):
        matrix = np.asarray(payload["linear_damping_matrix"], dtype=float)
        congruence = np.ones(6)
        congruence[1] = math.sqrt(scales.get("linear_sway_scale", 1.))
        congruence[5] = math.sqrt(scales.get("linear_yaw_scale", 1.))
        congruence[3] = math.sqrt(scales.get("roll_damping_scale", 1.))
        payload["linear_damping_matrix"] = (congruence[:, None] * matrix * congruence[None, :]).tolist()
    return payload
