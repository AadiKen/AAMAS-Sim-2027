"""Frozen physical task contract shared by every V3 backend."""
from dataclasses import dataclass
import math

from . import TASK_VERSION


@dataclass(frozen=True)
class TaskConfig:
    task_version: str = TASK_VERSION
    agent_count: int = 1
    dynamics: str = "kinematic"
    dt_s: float = 0.2
    deadline_steps: int = 600
    half_width_m: float = 50.0
    goal_radius_m: float = 2.0
    vessel_radius_m: float = 1.0
    max_surge_mps: float = 2.0
    max_yaw_rps: float = 0.25
    action_mode: str = "low_level"
    max_heading_offset_rad: float = math.pi
    heading_k_p: float = 0.5
    visibility_m: float = 30.0
    discount_per_second: float = 0.99
    progress_weight: float = 1.0
    goal_bonus: float = 20.0
    collision_penalty: float = 50.0
    step_penalty: float = 0.01

    def __post_init__(self):
        if self.task_version != TASK_VERSION:
            raise ValueError("Unsupported V3 task version")
        if self.dynamics not in {"kinematic", "bcod-reduced", "bcod-full"}:
            raise ValueError("Unsupported V3 backend")
        if not 1 <= self.agent_count <= 4 or self.dt_s <= 0 or self.deadline_steps <= 0:
            raise ValueError("Invalid agent count, time step, or deadline")
        if self.max_surge_mps != 2.0 or self.max_yaw_rps != 0.25:
            raise ValueError("V3 action limits are frozen at 2 m/s and 0.25 rad/s")
        if self.action_mode not in {"low_level", "high_level"}:
            raise ValueError("Unsupported policy action mode")
        if not (0 < self.max_heading_offset_rad <= math.pi and self.heading_k_p > 0):
            raise ValueError("Invalid heading controller configuration")

    @property
    def gamma(self) -> float:
        return self.discount_per_second ** self.dt_s
