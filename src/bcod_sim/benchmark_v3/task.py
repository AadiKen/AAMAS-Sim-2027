"""V3 policy observation and episode metrics atop simulator-independent scoring."""
import math

from bcod_sim.benchmark_v2.task import NavigationTask as ScoringTask

from .observations import build_observation


class NavigationTask(ScoringTask):
    def reset(self, scenario, frame):
        self._previous_readings = None
        self._current_readings = frame.readings
        self._previous_truth = frame.truth
        self.min_separation_m = float("inf")
        self.min_obstacle_clearance_m = float("inf")
        self.control_effort = {name: 0.0 for name in scenario.agent_ids}
        self.episode_returns = {name: 0.0 for name in scenario.agent_ids}
        self.episode_reward_components = {name: {key: 0.0 for key in
            ("potential_shaping", "goal", "collision", "step")} for name in scenario.agent_ids}
        return super().reset(scenario, frame)

    def _observations(self, frame):
        return {name: build_observation(name, frame.readings[name], self.scenario.goals[i],
                                       frame.readings, self._previous_readings,
                                       self.steps, self.config)
                for i, name in enumerate(self.scenario.agent_ids)}

    def step(self, frame, commands=None):
        self._previous_readings = self._current_readings
        self._current_readings = frame.readings
        names = self.scenario.agent_ids
        for i, name in enumerate(names):
            truth = frame.truth[name]
            for other in names[i + 1:]:
                q = frame.truth[other]
                self.min_separation_m = min(self.min_separation_m,
                                            math.dist((truth.x_m, truth.y_m), (q.x_m, q.y_m)))
            for obstacle in self.scenario.obstacles:
                clearance = math.dist((truth.x_m, truth.y_m), (obstacle.x_m, obstacle.y_m)) \
                    - obstacle.radius_m - self.config.vessel_radius_m
                self.min_obstacle_clearance_m = min(self.min_obstacle_clearance_m, clearance)
            if commands is not None:
                speed, yaw = commands[name]
                self.control_effort[name] += (speed / self.config.max_surge_mps) ** 2 \
                    + (yaw / self.config.max_yaw_rps) ** 2
        observations, rewards, terminated, truncated, info = super().step(frame)
        for name in names:
            self.episode_returns[name] += rewards[name]
            for component, value in info["reward_components"][name].items():
                self.episode_reward_components[name][component] += value
        info["episode_returns"] = dict(self.episode_returns)
        info["episode_reward_components"] = {name: dict(components)
                                             for name, components in self.episode_reward_components.items()}
        info["minimum_separation_m"] = None if math.isinf(self.min_separation_m) else self.min_separation_m
        info["minimum_obstacle_clearance_m"] = (None if math.isinf(self.min_obstacle_clearance_m)
                                                else self.min_obstacle_clearance_m)
        info["control_effort"] = dict(self.control_effort)
        info["path_efficiency"] = {name: min(1., math.dist(
            (self.scenario.starts[i].x_m, self.scenario.starts[i].y_m), self.scenario.goals[i])
            / max(self.path_lengths[name], 1e-9)) for i, name in enumerate(names)}
        return observations, rewards, terminated, truncated, info
