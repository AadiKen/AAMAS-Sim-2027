"""Thin PettingZoo Parallel API compatible adapter over EpisodeEngine."""

from types import MappingProxyType
from typing import Callable, Mapping

try:
    from pettingzoo import ParallelEnv
except ImportError:  # Native typed adapter remains usable without the validation extra.
    class ParallelEnv:
        @property
        def unwrapped(self):
            return self

from bcod_sim.core.engine import EpisodeEngine, EpisodeFrame
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.core.lifecycle import TerminationReason
from bcod_sim.rl.centralized_state import CentralizedStateContract
from bcod_sim.rl.vector_env import Action


class PettingZooParallelEnv(ParallelEnv):
    metadata = {"name": "bcod_sim_parallel_v0", "is_parallelizable": True}

    def __init__(self, engine: EpisodeEngine, *, observation_spaces: Mapping[str, object],
                 action_spaces: Mapping[str, object],
                 centralized_state: CentralizedStateContract | None = None,
                 action_decoders: Mapping[str, Callable[[object], Action]] | None = None,
                 observation_encoders: Mapping[str, Callable[[Mapping[str, object]], object]] | None = None) -> None:
        expected = {name for name, vessel in engine.config_vessels.items()
                    if vessel.controller.mode != "scripted"}
        if set(observation_spaces) != expected or set(action_spaces) != expected:
            raise PhysicalValidationError("Each RL agent requires its own declared observation and action space")
        self.engine = engine
        self.possible_agents = sorted(expected)
        self.agents: list[str] = []
        self._observation_spaces = dict(observation_spaces)
        self._action_spaces = dict(action_spaces)
        self.observation_spaces = self._observation_spaces
        self.action_spaces = self._action_spaces
        self._action_decoders = dict(action_decoders or {})
        self._observation_encoders = dict(observation_encoders or {})
        if set(self._action_decoders) - expected or set(self._observation_encoders) - expected:
            raise PhysicalValidationError("Codec declared for unknown agent")
        self._pending_disables: dict[str, str] = {}
        self._last_observations: dict[str, object] = {}
        self.render_mode = None
        self.centralized_state_contract = centralized_state

    def observation_space(self, agent: str) -> object:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> object:
        return self._action_spaces[agent]

    def _active(self) -> list[str]:
        return sorted(name for name, status in self.engine.statuses.items() if status.rl_active)

    def _observation(self, name: str, raw: Mapping[str, object]) -> object:
        value = self._observation_encoders[name](raw) if name in self._observation_encoders else raw
        space = self._observation_spaces[name]
        if callable(getattr(space, "contains", None)) and not space.contains(value):
            raise PhysicalValidationError(f"Observation outside declared space: {name}")
        return value

    def disable_agent(self, name: str, *, reason: str) -> None:
        if name not in self.agents or name in self._pending_disables or not reason:
            raise PhysicalValidationError("Can only disable a live agent with reason")
        self._pending_disables[name] = reason

    def reset(self, seed: int | None = None, options: Mapping | None = None):
        options = options or {}
        if not isinstance(options, Mapping):
            raise PhysicalValidationError("Reset options must be a mapping")
        frame = self.engine.reset(seed=seed, episode_index=int(options.get("episode_index", 0)))
        self.agents = self._active()
        self._pending_disables.clear()
        observations = {name: self._observation(name, frame.observations.get(name, MappingProxyType({})))
                        for name in self.agents}
        self._last_observations = dict(observations)
        infos = {name: {"pending_observation": name in frame.pending_observations} for name in self.agents}
        return observations, infos

    def step(self, actions: Mapping[str, Action]):
        acting = tuple(self.agents)
        if not acting or self.engine.terminated:
            raise PhysicalValidationError("Episode must be active before parallel step")
        if set(actions) != set(acting):
            raise PhysicalValidationError("Parallel step requires exactly one action per active agent")
        decoded = {}
        for name, value in actions.items():
            space = self._action_spaces[name]
            if callable(getattr(space, "contains", None)) and not space.contains(value):
                raise PhysicalValidationError(f"Action outside declared space: {name}")
            decoded[name] = self._action_decoders[name](value) if name in self._action_decoders else value
        for name, reason in self._pending_disables.items():
            self.engine.disable_agent(name, reason=reason)
            decoded.pop(name)
        self._pending_disables.clear()
        interval = self.engine.resolved.config.simulation.policy_every_n_master_steps
        totals = {name: 0.0 for name in acting}
        frame = self.engine.step(decoded)
        for name, value in (frame.reward.per_agent_total if frame.reward else {}).items():
            totals[name] += value
        for _ in range(1, interval):
            if frame.terminated:
                break
            frame = self.engine.step({})
            for name, value in (frame.reward.per_agent_total if frame.reward else {}).items():
                totals[name] += value
        # Parallel API returns the final observation on the transition that ends an episode.
        observations = {name: self._observation(name, frame.observations.get(name, MappingProxyType({})))
                        for name in acting if self.engine.statuses[name].rl_active}
        observations.update({name: self._last_observations[name] for name in acting
                             if not self.engine.statuses[name].rl_active and name in self._last_observations})
        self._last_observations = dict(observations)
        is_truncation = frame.termination_reason == TerminationReason.TIME_LIMIT
        terminations = {name: (frame.terminated and not is_truncation) or
                        not self.engine.statuses[name].rl_active for name in acting}
        truncations = {name: frame.terminated and is_truncation and
                       self.engine.statuses[name].rl_active for name in acting}
        infos = {name: {"termination_reason": (TerminationReason.AGENT_DISABLED.value
                        if not self.engine.statuses[name].rl_active else
                        frame.termination_reason.value if frame.termination_reason else None),
                        "disabled_reason": self.engine.statuses[name].disabled_reason,
                        "master_step": frame.master_step,
                        "pending_observation": name in frame.pending_observations,
                        "observation_freshness": frame.observation_freshness.get(name, {}),
                        "contact_events": tuple(event for event in frame.contact_events
                                                if f"vessel:{name}:" in event.contact_id),
                        "grounding": frame.grounding_telemetry.get(name),
                        "reward_components": (frame.reward.individual_components.get(name, {})
                                              if frame.reward else {})} for name in acting}
        self.agents = [] if frame.terminated else self._active()
        return observations, totals, terminations, truncations, infos

    def state(self):
        if self.centralized_state_contract is None:
            raise PhysicalValidationError("Centralized critic state was not declared")
        return self.centralized_state_contract.assemble(self.engine)

    def close(self) -> None:
        pass
