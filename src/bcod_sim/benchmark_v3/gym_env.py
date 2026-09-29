"""Thin single-agent Gymnasium binding for the versioned V3 task."""
import numpy as np
import gymnasium as gym

from . import ACTION_VERSION, OBSERVATION_VERSION, TASK_VERSION
from .adapters.kinematic import KinematicBackend
from .config import TaskConfig
from .control.actions import ACTION_SCHEMAS, action_space, decode_action
from .scenarios import sample_s0, sample_s1_train
from .task import NavigationTask


def action_to_physical(action, config: TaskConfig) -> tuple[float, float]:
    command = decode_action(action, config)
    return command.desired_speed_mps, command.desired_yaw_rate_radps


def make_backend(config: TaskConfig):
    if config.dynamics == "kinematic":
        return KinematicBackend(config)
    from .adapters.bcod import BCODV3Backend
    return BCODV3Backend(config)


class NavigationGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: TaskConfig = TaskConfig(), backend=None, scenario=None,
                 scenario_stage: str = "S0"):
        super().__init__()
        if config.agent_count != 1:
            raise ValueError("Single-agent Gymnasium interface requires one vessel")
        self.config = config
        self.backend = backend if backend is not None else make_backend(config)
        self.task = NavigationTask(config)
        self.default_scenario = scenario
        if scenario_stage not in {"S0", "S1"}:
            raise ValueError("Unsupported scenario stage")
        self.scenario_stage = scenario_stage
        self.action_space = action_space(config.action_mode)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(78,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scenario = options.get("scenario", self.default_scenario)
        if scenario is None:
            scenario = (sample_s0 if self.scenario_stage == "S0" else sample_s1_train)(self.np_random)
        frame = self.backend.reset(scenario, scenario.seed)
        observations = self.task.reset(scenario, frame)
        self.reading = frame.readings["vessel_0"]
        self.closed = False
        return observations["vessel_0"], {
            "scenario": scenario, "scenario_hash": scenario.geometry_hash(),
            "task_version": TASK_VERSION, "observation_version": OBSERVATION_VERSION,
            "action_version": ACTION_SCHEMAS[self.config.action_mode],
            "action_mode": self.config.action_mode, "backend": self.config.dynamics,
            "truth": frame.truth["vessel_0"], "reading": frame.readings["vessel_0"],
            "backend_diagnostics": frame.diagnostics,
        }

    def step(self, action):
        command = decode_action(action, self.config, heading_rad=self.reading.heading_rad,
                                yaw_rate_radps=self.reading.yaw_rps)
        speed, yaw = command.desired_speed_mps, command.desired_yaw_rate_radps
        commands = {"vessel_0": (speed, yaw)}
        frame = self.backend.step(commands, self.config.dt_s)
        self.reading = frame.readings["vessel_0"]
        observations, rewards, terminated, truncated, info = self.task.step(frame, commands)
        info.update({"physical_command": {"speed_mps": speed, "yaw_rate_radps": yaw},
                     "heading_control": {"heading_error_rad": command.heading_error_rad,
                         "requested_yaw_rate_radps": command.requested_yaw_rate_radps,
                         "saturated": command.saturated},
                     "scenario_hash": self.task.scenario.geometry_hash(),
                     "truth": frame.truth["vessel_0"], "reading": frame.readings["vessel_0"],
                     "backend_diagnostics": frame.diagnostics})
        return observations["vessel_0"], float(rewards["vessel_0"]), terminated, truncated, info

    def close(self):
        if not self.closed:
            self.backend.close()
            self.closed = True
