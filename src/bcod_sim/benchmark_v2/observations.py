"""48-element V2 observation: legacy 47 measured fields plus remaining horizon."""
import math
import numpy as np
from .backend import VesselReading
from .config import TaskConfig

OBSERVATION_VERSION = "navigation-v2-48"
OBSERVATION_FIELDS = (
    "x/half_width", "y/half_width", "sin_heading", "cos_heading",
    "surge/max_surge", "yaw_rate/max_yaw", "goal_dx/(2half_width)", "goal_dy/(2half_width)",
    *(f"neighbor_{i}_{component}" for i in range(3) for component in ("range/visibility", "sin_bearing", "cos_bearing")),
    *(f"obstacle_{i}_{component}" for i in range(10) for component in ("range/visibility", "sin_bearing", "cos_bearing")),
    "remaining_time_fraction",
)
assert len(OBSERVATION_FIELDS) == 48


def observation(reading: VesselReading, goal: tuple[float, float], steps: int, config: TaskConfig) -> np.ndarray:
    scale = config.half_width_m
    vector = [reading.x_m / scale, reading.y_m / scale,
              math.sin(reading.heading_rad), math.cos(reading.heading_rad),
              reading.surge_mps / config.max_surge_mps, reading.yaw_rps / config.max_yaw_rps,
              (goal[0] - reading.x_m) / (2 * scale), (goal[1] - reading.y_m) / (2 * scale)]
    for count, detected in ((3, reading.nearby_agents), (10, reading.nearby_obstacles)):
        ordered = sorted(detected, key=lambda p: math.dist(p[:2], (reading.x_m, reading.y_m)))[:count]
        for item in list(ordered) + [None] * (count - len(ordered)):
            if item is None or math.dist(item[:2], (reading.x_m, reading.y_m)) > config.visibility_m + 1e-8:
                vector.extend((0., 0., 0.))
            else:
                dx, dy = item[0] - reading.x_m, item[1] - reading.y_m
                bearing = math.atan2(dy, dx) - reading.heading_rad
                vector.extend((math.hypot(dx, dy) / config.visibility_m, math.sin(bearing), math.cos(bearing)))
    vector.append(max(0, config.deadline_steps - steps) / config.deadline_steps)
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (48,) or not np.isfinite(result).all():
        raise ValueError("Invalid V2 observation")
    return np.clip(result, -1., 1.)
