"""Measured, ego-relative 78-field V3 policy observation.

Relative velocity is (neighbor world velocity - ego world velocity), rotated into
the ego heading frame. Forward is positive ahead; lateral is positive left.
Velocities come from successive *measured* positions at dt, never scoring truth.
At reset there is no position history, so relative velocities are zero.
"""
import hashlib
import json
import math
import numpy as np

from . import OBSERVATION_VERSION

FIELDS = (
    "x_over_halfwidth", "y_over_halfwidth", "surge_over_max", "yaw_over_max",
    "sin_heading", "cos_heading", "goal_range_over_arena_diagonal",
    "sin_goal_bearing", "cos_goal_bearing",
    *(f"neighbor_{i}_{field}" for i in range(3) for field in
      ("range_over_visibility", "sin_bearing", "cos_bearing",
       "relative_forward_over_twice_maxspeed", "relative_lateral_over_twice_maxspeed", "present")),
    *(f"obstacle_{i}_{field}" for i in range(10) for field in
      ("range_over_visibility", "sin_bearing", "cos_bearing", "radius_over_visibility", "present")),
    "remaining_time_fraction",
)
assert len(FIELDS) == 78
SCHEMA_HASH = hashlib.sha256(json.dumps({"version": OBSERVATION_VERSION, "fields": FIELDS},
                                       separators=(",", ":")).encode()).hexdigest()


def _bearing(dx, dy, heading):
    angle = math.atan2(dy, dx) - heading
    return math.sin(angle), math.cos(angle)


def build_observation(name, reading, goal, readings, previous, steps, config):
    x, y, heading = reading.x_m, reading.y_m, reading.heading_rad
    dx, dy = goal[0] - x, goal[1] - y
    goal_sin, goal_cos = _bearing(dx, dy, heading)
    values = [x / config.half_width_m, y / config.half_width_m,
              reading.surge_mps / config.max_surge_mps, reading.yaw_rps / config.max_yaw_rps,
              math.sin(heading), math.cos(heading),
              math.hypot(dx, dy) / (2 * math.sqrt(2) * config.half_width_m), goal_sin, goal_cos]
    def measured_velocity(agent):
        if previous is None or agent not in previous:
            return (0.0, 0.0)
        current, old = readings[agent], previous[agent]
        return ((current.x_m - old.x_m) / config.dt_s,
                (current.y_m - old.y_m) / config.dt_s)
    own_vx, own_vy = measured_velocity(name)
    neighbors = sorted(((other, item) for other, item in readings.items() if other != name
                        and math.dist((x, y), (item.x_m, item.y_m)) <= config.visibility_m),
                       key=lambda pair: math.dist((x, y), (pair[1].x_m, pair[1].y_m)))[:3]
    for other, item in list(neighbors) + [(None, None)] * (3 - len(neighbors)):
        if item is None:
            values.extend((0.,) * 6)
            continue
        rx, ry = item.x_m - x, item.y_m - y
        sine, cosine = _bearing(rx, ry, heading)
        vx, vy = measured_velocity(other)
        rvx, rvy = vx - own_vx, vy - own_vy
        forward = rvx * math.cos(heading) + rvy * math.sin(heading)
        lateral = -rvx * math.sin(heading) + rvy * math.cos(heading)
        values.extend((math.hypot(rx, ry) / config.visibility_m, sine, cosine,
                       forward / (2 * config.max_surge_mps),
                       lateral / (2 * config.max_surge_mps), 1.))
    obstacles = sorted(reading.nearby_obstacles,
                       key=lambda item: math.dist((x, y), item[:2]))[:10]
    for item in list(obstacles) + [None] * (10 - len(obstacles)):
        if item is None or math.dist((x, y), item[:2]) > config.visibility_m:
            values.extend((0.,) * 5)
            continue
        rx, ry = item[0] - x, item[1] - y
        sine, cosine = _bearing(rx, ry, heading)
        values.extend((math.hypot(rx, ry) / config.visibility_m, sine, cosine,
                       item[2] / config.visibility_m, 1.))
    values.append(max(0, config.deadline_steps - steps) / config.deadline_steps)
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (78,) or not np.isfinite(vector).all():
        raise ValueError("Invalid V3 observation")
    return np.clip(vector, -1., 1.)
