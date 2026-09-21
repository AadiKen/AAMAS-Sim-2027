"""Thin PettingZoo Parallel API compatible adapter over EpisodeEngine."""

from types import MappingProxyType
from typing import Mapping

from bcod_sim.core.engine import EpisodeEngine, EpisodeFrame
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.core.lifecycle import TerminationReason
from bcod_sim.rl.centralized_state import CentralizedStateContract
from bcod_sim.rl.vector_env import Action


class PettingZooParallelEnv:
    metadata = {"name": "bcod_sim_parallel_v0", "is_parallelizable": True}

    def __init__(self, engine: EpisodeEngine, *, observation_spaces: Mapping[str, object],
                 action_spaces: Mapping[str, object],
                 centralized_state: CentralizedStateContract | None = None) -> None:
        expected = {name for name, vessel in engine.config_vessels.items()
                    if vessel.controller.mode != "scripted"}
        if set(observation_spaces) != expected or set(action_spaces) != expected:
            raise PhysicalValidationError("Each RL agent requires its own declared observation and action space")
        self.engine = engine
        self.possible_agents = sorted(expected)
        self.agents: list[str] = []
        self._observation_spaces = dict(observation_spaces)
        self._action_spaces = dict(action_spaces)
        self.centralized_state_contract = centralized_state

    def observation_space(self, agent: str) -> object:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> object:
        return self._action_spaces[agent]

    def _active(self) -> list[str]:
        return sorted(name for name, status in self.engine.statuses.items() if status.rl_active)

    def reset(self, seed: int | None = None, options: Mapping | None = None):
        options = options or {}
        unknown = set(options) - {"episode_index"}
        if unknown:
            raise PhysicalValidationError("Unknown reset option")
        frame = self.engine.reset(seed=seed, episode_index=int(options.get("episode_index", 0)))
        self.agents = self._active()
        observations = {name: frame.observations.get(name, MappingProxyType({})) for name in self.agents}
        infos = {name: {"pending_observation": name in frame.pending_observations} for name in self.agents}
        return observations, infos

    def step(self, actions: Mapping[str, Action]):
        acting = tuple(self.agents)
        if set(actions) != set(acting):
            raise PhysicalValidationError("Parallel step requires exactly one action per active agent")
        interval = self.engine.resolved.config.simulation.policy_every_n_master_steps
        totals = {name: 0.0 for name in acting}
        frame = self.engine.step(actions)
        for name, value in (frame.reward.per_agent_total if frame.reward else {}).items():
            totals[name] += value
        for _ in range(1, interval):
            if frame.terminated:
                break
            frame = self.engine.step({})
            for name, value in (frame.reward.per_agent_total if frame.reward else {}).items():
                totals[name] += value
        observations = {name: frame.observations.get(name, MappingProxyType({}))
                        for name in acting if self.engine.statuses[name].rl_active and not frame.terminated}
        is_truncation = frame.termination_reason == TerminationReason.TIME_LIMIT
        terminations = {name: frame.terminated and not is_truncation for name in acting}
        truncations = {name: frame.terminated and is_truncation for name in acting}
        infos = {name: {"termination_reason": frame.termination_reason.value if frame.termination_reason else None,
                        "master_step": frame.master_step,
                        "pending_observation": name in frame.pending_observations} for name in acting}
        self.agents = [] if frame.terminated else self._active()
        return observations, totals, terminations, truncations, infos

    def state(self):
        if self.centralized_state_contract is None:
            raise PhysicalValidationError("Centralized critic state was not declared")
        return self.centralized_state_contract.assemble(self.engine)

    def close(self) -> None:
        pass

