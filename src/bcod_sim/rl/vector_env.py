"""Native vector facade over independent authoritative episode engines."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from bcod_sim.actuators.autopilot import HighLevelCommand
from bcod_sim.batching.population import Population, flatten_population
from bcod_sim.core.engine import EpisodeEngine, EpisodeFrame
from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError
from bcod_sim.core.lifecycle import DirectAction


Action = DirectAction | HighLevelCommand


@dataclass(frozen=True)
class VectorStep:
    frames: Mapping[int, EpisodeFrame]
    rewards: Mapping[int, Mapping[str, float]]


class VectorEnvironment:
    def __init__(self, engines: Mapping[int, EpisodeEngine]) -> None:
        if not engines or any(not isinstance(key, int) or key < 0 for key in engines):
            raise PhysicalValidationError("Vector environment requires nonnegative stable environment IDs")
        if len(engines) != len(set(engines)):
            raise DuplicateIdentityError("Duplicate vector environment ID")
        self.engines = dict(engines)

    @property
    def env_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.engines))

    def reset(self, *, seeds: Mapping[int, int] | None = None,
              episode_indices: Mapping[int, int] | None = None) -> Mapping[int, EpisodeFrame]:
        seeds, episode_indices = seeds or {}, episode_indices or {}
        unknown = (set(seeds) | set(episode_indices)) - set(self.engines)
        if unknown:
            raise PhysicalValidationError("Reset options contain unknown environment ID")
        frames = {env_id: self.engines[env_id].reset(seed=seeds.get(env_id),
                    episode_index=episode_indices.get(env_id, 0)) for env_id in self.env_ids}
        return MappingProxyType(frames)

    def step(self, actions: Mapping[int, Mapping[str, Action]]) -> VectorStep:
        active_envs = {env_id for env_id, engine in self.engines.items() if not engine.terminated}
        if set(actions) != active_envs:
            raise PhysicalValidationError("Vector step requires actions for every active environment")
        frames, rewards = {}, {}
        for env_id in self.env_ids:
            engine = self.engines[env_id]
            if engine.terminated:
                continue
            interval = engine.resolved.config.simulation.policy_every_n_master_steps
            totals: dict[str, float] = {}
            frame = engine.step(actions[env_id])
            for name, value in (frame.reward.per_agent_total if frame.reward else {}).items():
                totals[name] = totals.get(name, 0.0) + value
            for _ in range(1, interval):
                if frame.terminated:
                    break
                frame = engine.step({})
                for name, value in (frame.reward.per_agent_total if frame.reward else {}).items():
                    totals[name] = totals.get(name, 0.0) + value
            frames[env_id], rewards[env_id] = frame, MappingProxyType(totals)
        return VectorStep(MappingProxyType(frames), MappingProxyType(rewards))

    def population(self) -> Population:
        return flatten_population(self.engines)

