"""Simulator-independent navigation scoring and finite-horizon lifecycle."""
import math
from typing import Mapping

from .backend import BackendFrame, Truth
from .config import TaskConfig
from .scenarios import Scenario, validate_scenario
from .observations import observation


def segment_distance(a, b, center):
    vx, vy = b[0] - a[0], b[1] - a[1]
    denominator = vx * vx + vy * vy
    fraction = 0. if denominator == 0 else max(0., min(1.,
        ((center[0] - a[0]) * vx + (center[1] - a[1]) * vy) / denominator))
    return math.dist((a[0] + fraction * vx, a[1] + fraction * vy), center)


class NavigationTask:
    def __init__(self, config: TaskConfig):
        self.config = config
        self.scenario = None

    def _validate_frame(self, frame: BackendFrame):
        if set(frame.readings) != set(self.scenario.agent_ids) or set(frame.truth) != set(self.scenario.agent_ids):
            raise ValueError("Backend agents differ from scenario")

    def _observations(self, frame):
        return {name: observation(frame.readings[name], self.scenario.goals[i], self.steps, self.config)
                for i, name in enumerate(self.scenario.agent_ids)}

    def reset(self, scenario: Scenario, frame: BackendFrame):
        validate_scenario(scenario, self.config)
        self.scenario = scenario
        self._validate_frame(frame)
        self.truth = frame.truth
        self.steps = 0
        self.done = False
        self.reached = {name: False for name in scenario.agent_ids}
        self.path_lengths = {name: 0. for name in scenario.agent_ids}
        self.minimum_distances = {name: math.dist((frame.truth[name].x_m, frame.truth[name].y_m), scenario.goals[i])
                                  for i, name in enumerate(scenario.agent_ids)}
        return self._observations(frame)

    def step(self, frame: BackendFrame):
        if self.done:
            raise RuntimeError("Episode has ended")
        self._validate_frame(frame)
        before_truth = self.truth
        self.truth = frame.truth
        self.steps += 1
        names = self.scenario.agent_ids
        colliders = set()
        newly_reached = {}
        for i, name in enumerate(names):
            old, new = before_truth[name], frame.truth[name]
            self.path_lengths[name] += math.dist((old.x_m, old.y_m), (new.x_m, new.y_m))
            goal = self.scenario.goals[i]
            self.minimum_distances[name] = min(self.minimum_distances[name],
                                               math.dist((new.x_m, new.y_m), goal))
            radius = self.config.vessel_radius_m
            if abs(new.x_m) + radius > self.config.half_width_m or abs(new.y_m) + radius > self.config.half_width_m:
                colliders.add(name)
            if any(segment_distance((old.x_m, old.y_m), (new.x_m, new.y_m), (o.x_m, o.y_m))
                   <= radius + o.radius_m for o in self.scenario.obstacles):
                colliders.add(name)
            for other in names[i + 1:]:
                q, old_q = frame.truth[other], before_truth[other]
                if segment_distance((old.x_m - old_q.x_m, old.y_m - old_q.y_m),
                                    (new.x_m - q.x_m, new.y_m - q.y_m), (0., 0.)) <= 2 * radius:
                    colliders.update((name, other))
            reached = not self.reached[name] and segment_distance(
                (old.x_m, old.y_m), (new.x_m, new.y_m), goal) <= self.config.goal_radius_m
            newly_reached[name] = reached
            self.reached[name] |= reached
        native_contacts = frame.diagnostics.get("native_contacts", ())
        # The native engine's contact events must never be ignored. Parse vessel IDs
        # when available; otherwise conservatively classify the episode as collision.
        if native_contacts:
            for event in native_contacts:
                colliders.update(name for name in names if name in str(event))
            if not colliders:
                colliders.update(names)
        native_reason = str(frame.diagnostics.get("native_termination_reason") or "")
        native_failure = bool(frame.diagnostics.get("native_terminated")) and "FAILURE" in native_reason.upper()
        collision = bool(colliders) or native_failure
        success = all(self.reached.values()) and not collision
        deadline = self.steps >= self.config.deadline_steps
        terminated = collision or success or deadline
        self.done = terminated
        reason = "collision" if collision else "success" if success else "deadline" if deadline else None
        components = {}
        rewards = {}
        for i, name in enumerate(names):
            old, new = before_truth[name], frame.truth[name]
            goal = self.scenario.goals[i]
            phi_before = -math.dist((old.x_m, old.y_m), goal)
            phi_after = 0. if terminated else -math.dist((new.x_m, new.y_m), goal)
            component = {
                "potential_shaping": self.config.progress_weight * (self.config.gamma * phi_after - phi_before),
                "goal": self.config.goal_bonus if newly_reached[name] and not collision else 0.,
                "collision": -self.config.collision_penalty if collision else 0.,
                "step": -self.config.step_penalty,
            }
            components[name] = component
            rewards[name] = sum(component.values())
        info = {
            "reason": reason, "terminal_reason": reason,
            "physical_colliders": sorted(colliders), "fleet_failure_due_to_collision": collision,
            "fleet_success": success, "per_agent_success": dict(self.reached),
            "reward_components": components, "steps": self.steps,
            "path_length_m": dict(self.path_lengths),
            "minimum_goal_distance_m": dict(self.minimum_distances),
            "final_goal_distance_m": {name: math.dist((frame.truth[name].x_m, frame.truth[name].y_m), self.scenario.goals[i])
                                      for i, name in enumerate(names)},
        }
        return self._observations(frame), rewards, terminated, False, info
