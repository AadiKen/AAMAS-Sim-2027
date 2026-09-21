"""Opt-in metrics registry and extension SDK."""

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Protocol

from bcod_sim.core.engine import EpisodeFrame
from bcod_sim.core.errors import UnknownReferenceError


MetricRow = dict[str, object]


class Metric(Protocol):
    def reset(self, frame: EpisodeFrame) -> None: ...
    def update(self, frame: EpisodeFrame, actions: Mapping[str, object] | None = None) -> tuple[MetricRow, ...]: ...


class RewardMetric:
    def reset(self, frame): pass
    def update(self, frame, actions=None):
        if frame.reward is None: return ()
        rows = [{"step": frame.master_step, "metric": "reward_total", "scope": agent, "value": value}
                for agent, value in sorted(frame.reward.per_agent_total.items())]
        for agent, components in sorted(frame.reward.individual_components.items()):
            rows.extend({"step": frame.master_step, "metric": f"reward.{name}", "scope": agent, "value": value}
                        for name, value in sorted(components.items()))
        rows.extend({"step": frame.master_step, "metric": f"reward.{name}", "scope": "team", "value": value}
                    for name, value in sorted(frame.reward.team_components.items()))
        return tuple(rows)


class SuccessMetric:
    def reset(self, frame): pass
    def update(self, frame, actions=None):
        return ({"step": frame.master_step, "metric": "success", "scope": "team",
                 "value": bool(frame.task_evaluation and frame.task_evaluation.success)},)


class CollisionMetric:
    def reset(self, frame): pass
    def update(self, frame, actions=None):
        severity = sum(abs(event.normal_impulse_ns) + abs(event.friction_impulse_ns) for event in frame.contact_events)
        return ({"step": frame.master_step, "metric": "collision_count", "scope": "team",
                 "value": len(frame.contact_events)},
                {"step": frame.master_step, "metric": "collision_severity_ns", "scope": "team",
                 "value": severity})


class DistanceMetric:
    def __init__(self): self.previous = {}
    def reset(self, frame): self.previous = {name: state.position_ned.clone() for name, state in frame.states.items()}
    def update(self, frame, actions=None):
        rows = []
        for name, state in sorted(frame.states.items()):
            distance = (state.position_ned-self.previous[name]).square().sum().sqrt().item()
            self.previous[name] = state.position_ned.clone()
            rows.append({"step": frame.master_step, "metric": "distance_traveled_m", "scope": name, "value": distance})
        return tuple(rows)


class TimeToCompletionMetric:
    def reset(self, frame): pass
    def update(self, frame, actions=None):
        if not frame.terminated: return ()
        return ({"step": frame.master_step, "metric": "time_to_completion_s", "scope": "team",
                 "value": frame.sim_time_s},)


class ControlEffortMetric:
    def reset(self, frame): pass
    def update(self, frame, actions=None):
        rows = []
        for agent, action in sorted((actions or {}).items()):
            effort = 0.0
            for _, command in getattr(action, "commands", ()):
                values = vars(command).values()
                effort += sum(abs(float(value)) for value in values if isinstance(value, (int, float)))
            rows.append({"step": frame.master_step, "metric": "control_effort", "scope": agent, "value": effort})
        return tuple(rows)


BUILTINS: dict[str, Callable[[], Metric]] = {
    "reward": RewardMetric, "success": SuccessMetric, "collision_count": CollisionMetric,
    "distance_traveled": DistanceMetric, "time_to_completion": TimeToCompletionMetric,
    "control_effort": ControlEffortMetric,
}


class MetricsRegistry:
    def __init__(self) -> None:
        self._factories = dict(BUILTINS)

    def register(self, name: str, factory: Callable[[], Metric]) -> None:
        if not name or name in self._factories:
            raise ValueError("Metric identity must be nonempty and unique")
        self._factories[name] = factory

    def build(self, names: tuple[str, ...]) -> tuple[Metric, ...]:
        unknown = set(names) - set(self._factories)
        if unknown:
            raise UnknownReferenceError(f"Unknown metrics: {sorted(unknown)}")
        return tuple(self._factories[name]() for name in names)


class MetricSet:
    def __init__(self, metrics: tuple[Metric, ...]) -> None: self.metrics = metrics
    def reset(self, frame: EpisodeFrame) -> None:
        for metric in self.metrics: metric.reset(frame)
    def update(self, frame: EpisodeFrame, actions: Mapping[str, object] | None = None) -> tuple[MetricRow, ...]:
        rows = tuple(row for metric in self.metrics for row in metric.update(frame, actions))
        if any(not isinstance(row.get("value"), (bool, int, float)) or
               isinstance(row.get("value"), float) and not math.isfinite(row["value"]) for row in rows):
            raise ValueError("Metric output must be finite scalar")
        return rows

