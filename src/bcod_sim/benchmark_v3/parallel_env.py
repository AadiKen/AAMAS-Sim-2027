"""Native simultaneous-vessel V3 PettingZoo environment.

Actor observations and training-only centralized state are separate outputs.
No optimizer or simulator-specific control logic lives here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from . import OBSERVATION_VERSION
from .centralized_state import (CENTRALIZED_STATE_VERSION, STATE_SCHEMA_HASH,
                                STATE_SPACE, encode_centralized_state)
from .config import TaskConfig
from .control.actions import ACTION_SCHEMAS, action_space, decode_action
from .gym_env import make_backend
from .observations import SCHEMA_HASH as ACTOR_SCHEMA_HASH
from .task import NavigationTask


class NavigationParallelEnv(ParallelEnv):
    metadata = {"name": "bcod_v3_fleet_parallel_v1", "is_parallelizable": True,
                "render_modes": []}

    def __init__(self, config: TaskConfig, *, scenario=None, scenario_sampler=None,
                 sampler_seed: int | None = None, backend=None):
        if config.agent_count < 2:
            raise ValueError("Fleet ParallelEnv requires at least two agents")
        if scenario is None and scenario_sampler is None:
            raise ValueError("Provide a fixed scenario or a scenario sampler")
        if scenario is not None and len(scenario.agent_ids) != config.agent_count:
            raise ValueError("Scenario and task agent counts disagree")
        self.config = config
        self.scenario = scenario
        self.scenario_sampler = scenario_sampler
        self._rng = np.random.default_rng(sampler_seed)
        self.backend = backend if backend is not None else make_backend(config)
        self.task = NavigationTask(config)
        self.possible_agents = [f"vessel_{i}" for i in range(config.agent_count)]
        self.agents = []
        self._frame = None
        self._closed = False
        self.actor_observation_schema = {"version": OBSERVATION_VERSION,
                                         "sha256": ACTOR_SCHEMA_HASH}
        self.centralized_state_schema = {"version": CENTRALIZED_STATE_VERSION,
                                          "sha256": STATE_SCHEMA_HASH}
        self.state_space = STATE_SPACE
        self.action_schema = ACTION_SCHEMAS[config.action_mode]
        self.render_mode = None

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return spaces.Box(-1., 1., shape=(78,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return action_space(self.config.action_mode)

    def state(self):
        if self._frame is None:
            raise RuntimeError("Reset before requesting centralized training state")
        return encode_centralized_state(
            self.scenario, self._frame, previous_readings=self.task._previous_readings,
            reached=self.task.reached, steps=self.task.steps, config=self.config)

    def reset(self, seed=None, options=None):
        options = options or {}
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        scenario = options.get("scenario")
        if scenario is None:
            scenario = (self.scenario_sampler(self._rng) if self.scenario_sampler is not None
                        else self.scenario)
        if len(scenario.agent_ids) != self.config.agent_count:
            raise ValueError("Scenario and task agent counts disagree")
        self.scenario = scenario
        self._frame = self.backend.reset(scenario, scenario.seed)
        observations = self.task.reset(scenario, self._frame)
        self.agents = list(scenario.agent_ids)
        self._closed = False
        infos = {name: {"scenario_hash": scenario.geometry_hash(),
                        "actor_observation_schema": self.actor_observation_schema,
                        "action_schema": self.action_schema}
                 for name in self.agents}
        return observations, infos

    def step(self, actions: Mapping[str, np.ndarray]):
        acting = list(self.agents)
        if not acting or set(actions) != set(acting):
            raise ValueError("Provide exactly one simultaneous action per live vessel")
        commands = {}
        controls = {}
        for name in acting:
            reading = self._frame.readings[name]
            if self.task.reached.get(name, False):
                # A finished vessel remains physical, but its policy action no
                # longer moves it while the rest of the fleet navigates.
                commands[name] = (0.0, 0.0)
                controls[name] = None
            else:
                control = decode_action(actions[name], self.config,
                                        heading_rad=reading.heading_rad,
                                        yaw_rate_radps=reading.yaw_rps)
                commands[name] = (control.desired_speed_mps,
                                  control.desired_yaw_rate_radps)
                controls[name] = control
        self._frame = self.backend.step(commands, self.config.dt_s)
        observations, rewards, done, _, common = self.task.step(self._frame, commands)
        reason = common["terminal_reason"]
        terminations = {name: bool(done and reason != "deadline") for name in acting}
        truncations = {name: bool(done and reason == "deadline") for name in acting}
        infos = {name: {"terminal_reason": reason,
                        "physical_colliders": common["physical_colliders"],
                        "fleet_success": common["fleet_success"],
                        "reward_components": common["reward_components"][name],
                        "physical_command": commands[name],
                        "heading_controller_saturated": (False if controls[name] is None
                                                         else controls[name].saturated),
                        "policy_action_ignored_after_goal": controls[name] is None}
                 for name in acting}
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def close(self):
        if not self._closed:
            self.backend.close()
            self._closed = True
