"""BenchMARL 1.5.2 custom-task adapter for V3 M0.

Only environment/spec wiring lives here. BenchMARL owns MAPPO optimization.
"""
from __future__ import annotations

from benchmarl.environments.pettingzoo.common import PettingZooClass

from ..config import TaskConfig
from .benchmarl_env import wrap_for_benchmarl


class V3BenchMARLTask(PettingZooClass):
    def __init__(self, task_config: TaskConfig, *, scenario_sampler,
                 centralized_state=True, task_name="navigation-v3"):
        super().__init__(name=task_name, config={"deadline_steps": task_config.deadline_steps,
                                            "agent_count": task_config.agent_count,
                                            "action_mode": task_config.action_mode,
                                            "backend": task_config.dynamics})
        self.task_config = task_config
        if scenario_sampler is None:
            raise ValueError("A scenario sampler must be supplied by the recipe adapter")
        self.scenario_sampler = scenario_sampler
        self.centralized_state = centralized_state

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        if not continuous_actions:
            raise ValueError("V3 BenchMARL task requires continuous policy actions")
        config = self.task_config
        return lambda: wrap_for_benchmarl(config, scenario_sampler=self.scenario_sampler,
                                          sampler_seed=seed, device=device,
                                          return_state=self.centralized_state)

    def supports_continuous_actions(self):
        return True

    def supports_discrete_actions(self):
        return False

    def has_state(self):
        return self.centralized_state

    def has_render(self, env):
        return False

    def max_steps(self, env):
        return self.config["deadline_steps"]

    @staticmethod
    def env_name():
        return "bcod_v3"


class V3M0Task(V3BenchMARLTask):
    """Backward-compatible established M0 recipe adapter."""
    def __init__(self, *, deadline_steps=600, stage_path=None):
        from .scenarios import sample_m0_train, sample_m0_curriculum_train
        from .curriculum import stage_sampler
        task_config = TaskConfig(agent_count=2, dynamics="kinematic", action_mode="high_level",
                                 deadline_steps=deadline_steps)
        sampler = stage_sampler(stage_path, sample_m0_curriculum_train) if stage_path is not None else sample_m0_train
        super().__init__(task_config, scenario_sampler=sampler, task_name="M0")
