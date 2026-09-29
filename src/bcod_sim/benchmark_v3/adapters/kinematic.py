"""Deterministic planar reference backend, intentionally not a paper simulator."""
import math

from bcod_sim.benchmark_v2.backend import BackendFrame, Truth, VesselReading
from ..config import TaskConfig


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class KinematicBackend:
    def __init__(self, config: TaskConfig):
        self.config = config
        self.scenario = None
        self.states = {}

    def reset(self, scenario, seed: int) -> BackendFrame:
        if seed != scenario.seed or len(scenario.agent_ids) != self.config.agent_count:
            raise ValueError("Scenario seed or agent count mismatch")
        self.scenario = scenario
        self.states = {name: [pose.x_m, pose.y_m, pose.heading_rad, 0.0, 0.0]
                       for name, pose in zip(scenario.agent_ids, scenario.starts)}
        return self._frame()

    def _frame(self) -> BackendFrame:
        readings = {}
        truth = {}
        for name, (x, y, heading, speed, yaw) in self.states.items():
            neighbors = tuple((other[0], other[1]) for key, other in self.states.items() if key != name
                              and math.dist((x, y), other[:2]) <= self.config.visibility_m)
            obstacles = tuple((o.x_m, o.y_m, o.radius_m) for o in self.scenario.obstacles
                              if math.dist((x, y), (o.x_m, o.y_m)) <= self.config.visibility_m)
            readings[name] = VesselReading(x, y, heading, speed, yaw, neighbors, obstacles)
            truth[name] = Truth(x, y, heading)
        return BackendFrame(readings, truth, {"backend": "kinematic", "native_contacts": ()})

    def step(self, physical_commands, dt_s: float) -> BackendFrame:
        if self.scenario is None or set(physical_commands) != set(self.states):
            raise ValueError("Reset first and command exactly the scenario agents")
        if abs(dt_s - self.config.dt_s) > 1e-12:
            raise ValueError("Time step mismatch")
        next_states = {}
        for name, target in physical_commands.items():
            speed, yaw = target
            if not (math.isfinite(speed) and math.isfinite(yaw)
                    and 0 <= speed <= self.config.max_surge_mps
                    and abs(yaw) <= self.config.max_yaw_rps):
                raise ValueError("Physical command outside V3 limits")
            x, y, heading, _, _ = self.states[name]
            heading = wrap(heading + yaw * dt_s)
            x += speed * math.cos(heading) * dt_s
            y += speed * math.sin(heading) * dt_s
            next_states[name] = [x, y, heading, speed, yaw]
        self.states = next_states
        return self._frame()

    def close(self) -> None:
        self.scenario = None
        self.states = {}
