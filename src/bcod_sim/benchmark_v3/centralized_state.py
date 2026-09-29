"""Training-only, simulator-independent global state for a centralized critic."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from gymnasium import spaces


CENTRALIZED_STATE_VERSION = "fleet-global-state-v1"
MAX_AGENTS = 4
MAX_OBSTACLES = 10
AGENT_FIELDS = ("x", "y", "sin_heading", "cos_heading", "surge", "yaw_rate",
                "velocity_x", "velocity_y", "goal_x", "goal_y", "reached", "present")
OBSTACLE_FIELDS = ("x", "y", "radius", "present")
FIELDS = ("remaining_time_fraction",
          *(f"agent_{i}_{field}" for i in range(MAX_AGENTS) for field in AGENT_FIELDS),
          *(f"obstacle_{i}_{field}" for i in range(MAX_OBSTACLES) for field in OBSTACLE_FIELDS))
STATE_SCHEMA_HASH = hashlib.sha256(json.dumps(
    {"version": CENTRALIZED_STATE_VERSION, "fields": FIELDS},
    separators=(",", ":")).encode()).hexdigest()
STATE_SPACE = spaces.Box(-1., 1., shape=(len(FIELDS),), dtype=np.float32)


def encode_centralized_state(scenario, frame, *, previous_readings=None,
                             reached=None, steps: int, config) -> np.ndarray:
    """Encode global training state; call only from state(), never actor observations.

    Positions and velocities come from backend readings. Velocity is estimated from
    successive measured positions; at reset, where there is no history, it is zero.
    The scenario supplies goals and static obstacle geometry. Padding and presence
    flags keep the schema fixed across two- and four-vessel tasks.
    """
    if not 1 <= len(scenario.agent_ids) <= MAX_AGENTS or len(scenario.obstacles) > MAX_OBSTACLES:
        raise ValueError("Centralized state exceeds the declared fleet/obstacle capacity")
    if set(frame.readings) != set(scenario.agent_ids) or not 0 <= steps <= config.deadline_steps:
        raise ValueError("Centralized state frame or time disagrees with the task")
    reached = reached or {}
    values = [(config.deadline_steps - steps) / config.deadline_steps]
    for i in range(MAX_AGENTS):
        if i >= len(scenario.agent_ids):
            values.extend((0.,) * len(AGENT_FIELDS))
            continue
        name = scenario.agent_ids[i]
        reading = frame.readings[name]
        if previous_readings is not None and name in previous_readings:
            old = previous_readings[name]
            vx = (reading.x_m - old.x_m) / config.dt_s
            vy = (reading.y_m - old.y_m) / config.dt_s
        else:
            vx = vy = 0.
        gx, gy = scenario.goals[i]
        values.extend((reading.x_m / config.half_width_m,
                       reading.y_m / config.half_width_m,
                       math.sin(reading.heading_rad), math.cos(reading.heading_rad),
                       reading.surge_mps / config.max_surge_mps,
                       reading.yaw_rps / config.max_yaw_rps,
                       vx / config.max_surge_mps, vy / config.max_surge_mps,
                       gx / config.half_width_m, gy / config.half_width_m,
                       float(bool(reached.get(name, False))), 1.))
    for i in range(MAX_OBSTACLES):
        if i >= len(scenario.obstacles):
            values.extend((0.,) * len(OBSTACLE_FIELDS))
            continue
        obstacle = scenario.obstacles[i]
        values.extend((obstacle.x_m / config.half_width_m,
                       obstacle.y_m / config.half_width_m,
                       obstacle.radius_m / config.half_width_m, 1.))
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (len(FIELDS),) or not np.isfinite(state).all():
        raise ValueError("Invalid centralized state")
    return np.clip(state, -1., 1.)
