"""BenchMARL 1.5.2 custom-task adapter for V3 M0.

Only environment/spec wiring lives here. BenchMARL owns MAPPO optimization.
"""
from __future__ import annotations

from benchmarl.environments.pettingzoo.common import PettingZooClass

from ..config import TaskConfig
from .benchmarl_env import wrap_for_benchmarl
from .scenarios import sample_m0_train, sample_m0_curriculum_train
from .curriculum import stage_sampler


class V3M0Task(PettingZooClass):
    def __init__(self, *, deadline_steps: int = 600, stage_path=None):
        super().__init__(name="M0", config={"deadline_steps": deadline_steps,
                                            "agent_count": 2,
                                            "action_mode": "high_level",
                                            "backend": "kinematic"})
        self.stage_path = stage_path

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        if not continuous_actions:
            raise ValueError("V3 M0 uses continuous high-level actions")
        config = TaskConfig(agent_count=2, dynamics="kinematic", action_mode="high_level",
                            deadline_steps=self.config["deadline_steps"])
        sampler = (stage_sampler(self.stage_path, sample_m0_curriculum_train)
                   if self.stage_path is not None else sample_m0_train)
        return lambda: wrap_for_benchmarl(config, scenario_sampler=sampler,
                                          sampler_seed=seed, device=device)

    def supports_continuous_actions(self):
        return True

    def supports_discrete_actions(self):
        return False

    def has_state(self):
        return True

    def has_render(self, env):
        return False

    def max_steps(self, env):
        return self.config["deadline_steps"]

    @staticmethod
    def env_name():
        return "bcod_v3"
