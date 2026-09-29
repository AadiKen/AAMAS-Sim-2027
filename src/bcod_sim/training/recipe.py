"""Small, versioned configuration shared by the paper training entry points."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentConfig(StrictModel):
    task: Literal["navigation-v3"] = "navigation-v3"
    scenario: str
    backend: Literal["kinematic", "bcod-reduced", "bcod-full"] = "kinematic"
    agent_count: int = Field(ge=1, le=4)
    action_mode: Literal["high_level", "low_level"] = "high_level"
    deadline_steps: int = Field(default=600, ge=1)
    dt_s: float = Field(default=0.2, gt=0)
    half_width_m: float = Field(default=50.0, gt=0)
    goal_radius_m: float = Field(default=2.0, gt=0)
    vessel_radius_m: float = Field(default=1.0, gt=0)
    max_surge_mps: float = Field(default=2.0, gt=0)
    max_yaw_rps: float = Field(default=0.25, gt=0)
    max_heading_offset_rad: float = Field(default=3.141592653589793, gt=0)
    heading_k_p: float = Field(default=0.5, gt=0)
    visibility_m: float = Field(default=30.0, gt=0)
    discount_per_second: float = Field(default=0.99, gt=0, le=1)
    progress_weight: float = Field(default=1.0, ge=0)
    goal_bonus: float = Field(default=20.0, ge=0)
    collision_penalty: float = Field(default=50.0, ge=0)
    step_penalty: float = Field(default=0.01, ge=0)


class TrainerConfig(StrictModel):
    library: Literal["stable_baselines3", "benchmarl"]
    algorithm: Literal["PPO", "MAPPO"]
    device: str = "cpu"
    total_steps: int | None = Field(default=None, ge=1)
    rollout_steps: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    epochs: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    rollout_frames: int | None = Field(default=None, ge=1)
    environments: int | None = Field(default=None, ge=1)
    minibatch_size: int | None = Field(default=None, ge=1)
    minibatch_iterations: int | None = Field(default=None, ge=1)
    centralized_critic: bool | None = None
    n_envs: int | None = Field(default=None, ge=1)
    gae_lambda: float | None = Field(default=None, gt=0, le=1)
    clip_range: float | None = Field(default=None, gt=0)
    ent_coef: float | None = Field(default=None, ge=0)
    vf_coef: float | None = Field(default=None, ge=0)
    max_grad_norm: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def supported_pair(self):
        if (self.library, self.algorithm) not in {("stable_baselines3", "PPO"), ("benchmarl", "MAPPO")}:
            raise ValueError("Supported trainer pairs are stable_baselines3/PPO and benchmarl/MAPPO")
        return self


class EvaluationConfig(StrictModel):
    interval_steps: int = Field(ge=1)
    bank: str
    cases: int = Field(ge=1)
    seed: int = Field(default=2027, ge=0)


class TrainingRecipe(StrictModel):
    recipe_version: Literal[1]
    environment: EnvironmentConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    selection: dict = Field(default_factory=dict)
    seeds: tuple[int, ...] = Field(min_length=1)
    curriculum: dict | None = None

    @model_validator(mode="after")
    def mode_consistency(self):
        multi = self.environment.agent_count > 1
        if multi != (self.trainer.library == "benchmarl"):
            raise ValueError("MARL recipes require BenchMARL MAPPO and SARL recipes require SB3 PPO")
        return self


def load_recipe(path: str | Path) -> TrainingRecipe:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        return TrainingRecipe.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid training recipe: {exc}") from exc
