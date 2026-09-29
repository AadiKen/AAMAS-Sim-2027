"""V3 wrapper around the qualified BCOD physical backend."""
from dataclasses import replace
import math

from bcod_sim.benchmark_v2.backend import BackendFrame
from bcod_sim.benchmark_v2.adapters.bcod import BCODBackend


class BCODV3Backend:
    def __init__(self, config):
        self.config = config
        self._physical = BCODBackend(config)
        self._physical.reduced_fidelity = config.dynamics == "bcod-reduced"
        self.scenario = None

    @property
    def engine(self):
        return self._physical.engine

    def reset(self, scenario, seed):
        self.scenario = scenario
        return self._with_radii(self._physical.reset(scenario, seed))

    def step(self, physical_commands, dt_s):
        return self._with_radii(self._physical.step(physical_commands, dt_s))

    def _with_radii(self, frame):
        readings = {}
        six_dof = {}
        for name, reading in frame.readings.items():
            obstacles = []
            for x, y, _ in reading.nearby_obstacles:
                nearest = min(self.scenario.obstacles,
                              key=lambda obstacle: (obstacle.x_m - x) ** 2 + (obstacle.y_m - y) ** 2,
                              default=None)
                radius = nearest.radius_m if nearest is not None else 0.0
                obstacles.append((x, y, radius))
            readings[name] = replace(reading, nearby_obstacles=tuple(obstacles))
            state = self._physical.engine.states[name]
            w, x, y, z = (float(component) for component in state.q_body_to_ned)
            roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = math.asin(max(-1., min(1., 2 * (w * y - z * x))))
            six_dof[name] = {
                "roll_rad": roll, "pitch_rad": pitch,
                "surge_frd_mps": float(state.nu_body[0]),
                "sway_frd_mps": float(state.nu_body[1]),
                "yaw_rate_frd_radps": float(state.nu_body[5]),
            }
        diagnostics = dict(frame.diagnostics)
        diagnostics["physical_6dof"] = six_dof
        return BackendFrame(readings, frame.truth, diagnostics)

    def close(self):
        self._physical.close()
