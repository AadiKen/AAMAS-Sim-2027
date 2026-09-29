"""Versioned policy actions shared by Gymnasium and future PettingZoo wrappers."""
import math
import numpy as np
from gymnasium import spaces

from .high_level import HeadingController, LowLevelCommand, wrap_angle

ACTION_SCHEMAS = {"low_level": "desired-speed-yawrate-v2",
                  "high_level": "desired-speed-heading-v1"}


def action_space(mode: str):
    if mode not in ACTION_SCHEMAS:
        raise ValueError(f"Unsupported policy action mode: {mode}")
    return spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                      high=np.array([1., 1.], dtype=np.float32))


def decode_action(action, config, *, heading_rad: float = 0., yaw_rate_radps: float = 0.) -> LowLevelCommand:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < -1) or np.any(values > 1):
        raise ValueError("Action must be finite Box(-1, 1, shape=(2,))")
    speed = float((values[0] + 1.) * config.max_surge_mps / 2.)
    if config.action_mode == "low_level":
        return LowLevelCommand(speed, float(values[1] * config.max_yaw_rps))
    if config.action_mode != "high_level":
        raise ValueError(f"Unsupported policy action mode: {config.action_mode}")
    desired_heading = wrap_angle(heading_rad + float(values[1]) * config.max_heading_offset_rad)
    return HeadingController(config.max_yaw_rps, config.heading_k_p).command(
        speed, desired_heading, heading_rad, yaw_rate_radps, config.dt_s)
