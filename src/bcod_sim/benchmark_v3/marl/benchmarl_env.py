"""TorchRL PettingZoo wrapper for the authoritative V3 ParallelEnv.

TorchRL 0.11.1 tensorizes PettingZoo info fields at construction. V3 info
contains human-readable hashes and reason strings, so this adapter hides those
fields from TorchRL while preserving them on ``last_v3_infos`` for logging.
"""
from __future__ import annotations

from torchrl.envs import PettingZooWrapper

from ..parallel_env import NavigationParallelEnv


class TensorInfoNavigationEnv(NavigationParallelEnv):
    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self.last_v3_infos = infos
        return observations, {agent: {} for agent in observations}

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        self.last_v3_infos = infos
        return observations, rewards, terminations, truncations, {agent: {} for agent in infos}


def wrap_for_benchmarl(config, scenario=None, *, scenario_sampler=None,
                       sampler_seed=None, backend=None, device="cpu", return_state=True):
    env = TensorInfoNavigationEnv(config, scenario=scenario,
                                  scenario_sampler=scenario_sampler,
                                  sampler_seed=sampler_seed, backend=backend)
    return PettingZooWrapper(env=env, return_state=return_state,
                             group_map={"agents": list(env.possible_agents)},
                             categorical_actions=True, device=device)
