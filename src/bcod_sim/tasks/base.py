"""Task outputs, reward composition, and checkpointable task state."""

from dataclasses import dataclass
import copy
from typing import Mapping, Protocol

from bcod_sim.config.models import Reward
from bcod_sim.state.vessel_state import VesselState


@dataclass(frozen=True)
class TaskEvaluation:
    individual_components: Mapping[str, Mapping[str, float]]
    team_components: Mapping[str, float]
    success: bool = False
    failure: bool = False


@dataclass(frozen=True)
class RewardReport:
    individual_components: Mapping[str, Mapping[str, float]]
    team_components: Mapping[str, float]
    per_agent_total: Mapping[str, float]


def compose_reward(evaluation: TaskEvaluation, weights: Reward,
                   active_agent_ids: tuple[str, ...]) -> RewardReport:
    team = sum(evaluation.team_components.values())
    totals = {agent_id: weights.individual_weight * sum(evaluation.individual_components.get(agent_id, {}).values()) +
              weights.team_weight * team for agent_id in active_agent_ids}
    return RewardReport(evaluation.individual_components, evaluation.team_components, totals)


class Task(Protocol):
    def reset(self, states: Mapping[str, VesselState]) -> None: ...
    def evaluate(self, states: Mapping[str, VesselState], active_agent_ids: tuple[str, ...]) -> TaskEvaluation: ...
    def snapshot(self) -> object: ...
    def restore(self, snapshot: object) -> None: ...


class StatefulTask:
    def snapshot(self) -> object:
        return copy.deepcopy(self.__dict__)

    def restore(self, snapshot: object) -> None:
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(snapshot))
