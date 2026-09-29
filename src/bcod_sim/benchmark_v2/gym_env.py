"""Thin Gymnasium wrapper for the single-agent V2 task."""
import math
import numpy as np
import gymnasium as gym

from .adapters.bcod import BCODBackend
from .config import TaskConfig
from .scenarios import Scenario, case_a_reference
from .task import NavigationTask

ACTION_VERSION = "symmetric-speed-yaw-v2"


def action_to_physical(action, config: TaskConfig) -> tuple[float, float]:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all() or np.any(array < -1.) or np.any(array > 1.):
        raise ValueError("V2 action must be finite Box(-1, 1, shape=(2,))")
    return (float((array[0] + 1.) / 2. * config.max_surge_mps),
            float(array[1] * config.max_yaw_rps))


class NavigationGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: TaskConfig = TaskConfig(), backend=None, scenario: Scenario | None = None):
        super().__init__()
        if config.agent_count != 1:
            raise ValueError("Gymnasium V2 interface is single-agent; use the task/backend API for fleets")
        self.config = config
        self.backend = backend if backend is not None else BCODBackend(config)
        self.task = NavigationTask(config)
        self.default_scenario = scenario
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(48,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scenario = options.get("scenario", self.default_scenario)
        if scenario is None:
            scenario_seed = int(self.np_random.integers(0, 2**31))
            scenario = case_a_reference(scenario_seed)
        frame = self.backend.reset(scenario, scenario.seed)
        observations = self.task.reset(scenario, frame)
        self.closed = False
        return observations["vessel_0"], {"scenario": scenario, "scenario_hash": scenario.geometry_hash(),
                                           "backend_diagnostics": frame.diagnostics,
                                           "truth": frame.truth["vessel_0"],
                                           "reading": frame.readings["vessel_0"]}

    def step(self, action):
        speed, yaw = action_to_physical(action, self.config)
        frame = self.backend.step({"vessel_0": (speed, yaw)}, self.config.dt_s)
        observations, rewards, terminated, truncated, info = self.task.step(frame)
        info.update({"physical_command": {"speed_mps": speed, "yaw_rate_radps": yaw},
                     "backend_diagnostics": frame.diagnostics,
                     "truth": frame.truth["vessel_0"], "reading": frame.readings["vessel_0"]})
        return observations["vessel_0"], float(rewards["vessel_0"]), terminated, truncated, info

    def close(self):
        if not self.closed:
            self.backend.close()
            self.closed = True
