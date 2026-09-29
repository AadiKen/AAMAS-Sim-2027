"""Versioned navigation task configuration; no learner settings live here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    task_version: str = "navigation-v2"
    agent_count: int = 1
    dynamics: str = "BCOD-reduced"
    dt_s: float = 0.2
    deadline_steps: int = 600
    half_width_m: float = 50.0
    goal_radius_m: float = 2.0
    vessel_radius_m: float = 1.0
    max_surge_mps: float = 2.0
    max_yaw_rps: float = 0.25
    visibility_m: float = 30.0
    discount_per_second: float = 0.99
    progress_weight: float = 1.0
    goal_bonus: float = 20.0
    collision_penalty: float = 50.0
    step_penalty: float = 0.01

    def __post_init__(self):
        if not 1 <= self.agent_count <= 4 or self.dt_s <= 0 or self.deadline_steps <= 0:
            raise ValueError("Invalid agent count, time step, or deadline")
        if self.dynamics != "BCOD-reduced":
            raise ValueError("Only BCOD-reduced is supported in V2")

    @property
    def gamma(self) -> float:
        return self.discount_per_second ** self.dt_s


# Common adapter-compatible names, deliberately independent of legacy learners.
BenchmarkConfig = TaskConfig
