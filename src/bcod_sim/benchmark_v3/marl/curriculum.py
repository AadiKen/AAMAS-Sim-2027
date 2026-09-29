"""Scenario-stage scheduling independent of MAPPO optimization and M0 geometry."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    scenario_distribution: str
    promotion_metric: str
    success_threshold: float
    collision_ceiling: float | None
    consecutive_evaluations: int
    minimum_training_steps: int
    maximum_training_steps: int


@dataclass(frozen=True)
class Curriculum:
    stages: tuple[CurriculumStage, ...]

    def __post_init__(self):
        if not self.stages or len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("Curriculum needs unique ordered stages")
        for stage in self.stages:
            if not 0 <= stage.success_threshold <= 1 or stage.consecutive_evaluations < 1:
                raise ValueError("Invalid curriculum promotion gate")
            if not 0 <= stage.minimum_training_steps <= stage.maximum_training_steps:
                raise ValueError("Invalid stage budget")


class CurriculumController:
    """Tracks promotion; the trainer remains agnostic to scenario semantics."""

    def __init__(self, curriculum: Curriculum):
        self.curriculum = curriculum
        self.index = 0
        self.entry_step = 0
        self.consecutive_passes = 0
        self.history = []

    @property
    def stage(self):
        return self.curriculum.stages[self.index]

    def record(self, total_steps: int, *, success_rate: float, collision_rate: float):
        stage = self.stage
        stage_steps = total_steps - self.entry_step
        meets = (success_rate >= stage.success_threshold and
                 (stage.collision_ceiling is None or collision_rate <= stage.collision_ceiling))
        self.consecutive_passes = self.consecutive_passes + 1 if meets else 0
        event = {"stage": stage.name, "step": total_steps, "stage_steps": stage_steps,
                 "success_rate": success_rate, "collision_rate": collision_rate,
                 "consecutive_passes": self.consecutive_passes}
        self.history.append(event)
        if self.index == len(self.curriculum.stages)-1:
            return "complete" if stage_steps >= stage.maximum_training_steps else "continue"
        if (stage_steps >= stage.minimum_training_steps and
                self.consecutive_passes >= stage.consecutive_evaluations):
            event["promoted_to"] = self.curriculum.stages[self.index+1].name
            self.index += 1
            self.entry_step = total_steps
            self.consecutive_passes = 0
            return "promote"
        if stage_steps >= stage.maximum_training_steps:
            event["failed_promotion"] = True
            return "fail"
        return "continue"


def write_stage(path: Path, stage: str):
    """Atomic stage publication, visible to existing serial/parallel env workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"stage": stage}) + "\n")
    temporary.replace(path)


def stage_sampler(stage_path: Path, sample: Callable):
    """At each reset, sample the currently published stage with that env's RNG."""
    path = Path(stage_path)
    def draw(rng):
        stage = json.loads(path.read_text())["stage"]
        return sample(rng, stage)
    return draw
