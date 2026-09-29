"""Task-metric M0 evaluation, independent of the MAPPO trainer."""
from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from ..config import TaskConfig
from ..parallel_env import NavigationParallelEnv


class BenchMARLActorPolicy:
    """Deterministic local-observation execution of a shared BenchMARL actor."""

    def __init__(self, actor, agent_ids=("vessel_0", "vessel_1")):
        self.actor = actor
        self.agent_ids = tuple(agent_ids)

    def actions(self, observations):
        if set(observations) != set(self.agent_ids):
            raise ValueError("Actor evaluation requires all declared local observations")
        matrix = np.stack([observations[name] for name in self.agent_ids])
        td = TensorDict({"agents": TensorDict(
            {"observation": torch.as_tensor(matrix, dtype=torch.float32)},
            batch_size=[len(self.agent_ids)])}, batch_size=[])
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            action = self.actor(td)["agents", "action"].detach().cpu().numpy()
        return {name: action[i].astype(np.float32) for i, name in enumerate(self.agent_ids)}


class M0GeometricController:
    """Observation-only navigation baseline with local collision avoidance."""

    def actions(self, observations):
        result = {}
        for name, obs in observations.items():
            goal_bearing = math.atan2(float(obs[7]), float(obs[8]))
            heading_error = goal_bearing
            speed = 1.25
            for i in range(3):
                fields = obs[9 + 6*i:15 + 6*i]
                if fields[5] < .5:
                    continue
                distance = float(fields[0]) * 30.
                bearing = math.atan2(float(fields[1]), float(fields[2]))
                rx, ry = distance * math.cos(bearing), distance * math.sin(bearing)
                rvx, rvy = float(fields[3])*4., float(fields[4])*4.
                velocity2 = rvx*rvx + rvy*rvy
                tca = max(0., min(7., -(rx*rvx + ry*rvy)/max(velocity2, 1e-9)))
                closest = math.hypot(rx + rvx*tca, ry + rvy*tca)
                if distance < 10. and closest < 3.5:
                    # Both vessels steer to their own right on a head-on encounter.
                    if abs(bearing) < .8 and rvx < -.15:
                        heading_error = min(heading_error, -.75)
                        speed = min(speed, .95)
                    elif name == "vessel_1":
                        speed = .1 if distance < 6. else .45
                    else:
                        heading_error += (-.35 if bearing >= 0 else .35)
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            result[name] = np.array([speed - 1., np.clip(heading_error / math.pi, -1., 1.)],
                                    dtype=np.float32)
        return result


class StraightToGoalController:
    def actions(self, observations):
        return {name: np.array([.0, math.atan2(float(obs[7]), float(obs[8])) / math.pi],
                               dtype=np.float32) for name, obs in observations.items()}


def evaluate_m0(policy, scenarios, *, config=None, trace_dir: Path | None = None,
                trace_cases: int = 3):
    config = config or TaskConfig(agent_count=2, action_mode="high_level")
    episodes = []
    for case_index, scenario in enumerate(scenarios):
        env = NavigationParallelEnv(config, scenario=scenario)
        try:
            observations, _ = env.reset(seed=scenario.seed)
            trace = []
            saturated, total_commands = 0, 0
            commanded_speed, heading_command, command_count = 0., 0., 0
            low_speed_steps = deadlock_steps = deadlock_events = near_miss_steps = 0
            stalled_steps = 0
            previous_distances = {name: math.dist(
                (env._frame.readings[name].x_m, env._frame.readings[name].y_m),
                scenario.goals[i]) for i, name in enumerate(scenario.agent_ids)}
            for step in range(1, config.deadline_steps + 2):
                actions = policy.actions(observations)
                before = env._frame.readings
                observations, rewards, terms, truncs, infos = env.step(actions)
                saturated += sum(bool(infos[name]["heading_controller_saturated"])
                                 for name in scenario.agent_ids)
                total_commands += len(scenario.agent_ids)
                commanded_speed += sum(infos[name]["physical_command"][0]
                                       for name in scenario.agent_ids)
                heading_command += sum(abs(float(actions[name][1]))
                                       for name in scenario.agent_ids)
                command_count += len(scenario.agent_ids)
                current_distances = {name: math.dist(
                    (env._frame.readings[name].x_m, env._frame.readings[name].y_m),
                    scenario.goals[i]) for i, name in enumerate(scenario.agent_ids)}
                both_slow = all(env._frame.readings[name].surge_mps < .15 and
                                not env.task.reached[name] for name in scenario.agent_ids)
                if both_slow:
                    low_speed_steps += 1
                stalled = both_slow and all(previous_distances[name] - current_distances[name]
                                            < .01 for name in scenario.agent_ids)
                stalled_steps = stalled_steps + 1 if stalled else 0
                if stalled_steps == 25:
                    deadlock_events += 1
                if stalled_steps >= 25:
                    deadlock_steps += 1
                previous_distances = current_distances
                first, second = (env._frame.readings[name] for name in scenario.agent_ids)
                instantaneous_separation = math.dist((first.x_m, first.y_m),
                                                      (second.x_m, second.y_m))
                if 2. < instantaneous_separation < 3.:
                    near_miss_steps += 1
                if trace_dir is not None and case_index < trace_cases:
                    trace.append({"step": step, "minimum_agent_separation_m":
                        env.task.min_separation_m,
                        "agents": {name: {"position_m": [env._frame.readings[name].x_m,
                                                      env._frame.readings[name].y_m],
                          "heading_rad": env._frame.readings[name].heading_rad,
                          "speed_mps": env._frame.readings[name].surge_mps,
                          "achieved_yaw_rate_radps": env._frame.readings[name].yaw_rps,
                          "goal_m": scenario.goals[i],
                          "action": np.asarray(actions[name]).tolist(),
                          "desired_speed_mps": infos[name]["physical_command"][0],
                          "desired_heading_rad": math.atan2(math.sin(
                              before[name].heading_rad + float(actions[name][1])*math.pi),
                              math.cos(before[name].heading_rad + float(actions[name][1])*math.pi)),
                          "heading_controller_saturated":
                              infos[name]["heading_controller_saturated"],
                          "reward": rewards[name],
                          "termination": infos[name]["terminal_reason"]}
                          for i, name in enumerate(scenario.agent_ids)}})
                if all(terms.values()) or all(truncs.values()):
                    break
            else:
                raise AssertionError("M0 evaluation exceeded deadline")
            reason = infos[scenario.agent_ids[0]]["terminal_reason"]
            episodes.append({"scenario_hash": scenario.geometry_hash(),
                             "fleet_success": reason == "success",
                             "per_agent_success": dict(env.task.reached),
                             "collision": reason == "collision",
                             "deadline": reason == "deadline",
                             "episode_length": step,
                             "minimum_agent_separation_m": (
                                 None if math.isinf(env.task.min_separation_m)
                                 else env.task.min_separation_m),
                             "fleet_reward": sum(env.task.episode_returns.values()),
                             "heading_controller_saturation_rate": saturated/max(total_commands, 1),
                             "mean_commanded_speed_mps": commanded_speed/max(command_count, 1),
                             "mean_absolute_relative_heading_action": heading_command/max(command_count, 1),
                             "both_low_speed_fraction": low_speed_steps/max(step, 1),
                             "deadlock_count": deadlock_events,
                             "deadlock_duration_s": deadlock_steps*config.dt_s,
                             "near_miss_steps": near_miss_steps,
                             "reward_components": dict(env.task.episode_reward_components),
                             "physical_colliders": infos[scenario.agent_ids[0]]["physical_colliders"],
                             "path_efficiency": float(np.mean([
                                 min(1., math.dist((scenario.starts[i].x_m,
                                                   scenario.starts[i].y_m), scenario.goals[i]) /
                                     max(env.task.path_lengths[name], 1e-9))
                                 for i, name in enumerate(scenario.agent_ids)]))
                             if reason == "success" else 0.})
            if trace_dir is not None and case_index < trace_cases:
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / f"case-{case_index}.json").write_text(json.dumps(
                    {"scenario_hash": scenario.geometry_hash(), "trajectory": trace},
                    indent=2, default=str) + "\n")
        finally:
            env.close()
    return {"scenario_count": len(episodes),
            "fleet_success_count": sum(e["fleet_success"] for e in episodes),
            "per_agent_success_count": {name: sum(e["per_agent_success"][name]
                for e in episodes) for name in scenarios[0].agent_ids},
            "collision_count": sum(e["collision"] for e in episodes),
            "deadline_count": sum(e["deadline"] for e in episodes),
            "mean_episode_length": float(np.mean([e["episode_length"] for e in episodes])),
            "mean_success_completion_steps": (float(np.mean([e["episode_length"]
                for e in episodes if e["fleet_success"]]))
                if any(e["fleet_success"] for e in episodes) else None),
            "mean_path_efficiency": float(np.mean([e["path_efficiency"] for e in episodes])),
            "mean_minimum_agent_separation_m": float(np.mean([
                e["minimum_agent_separation_m"] for e in episodes
                if e["minimum_agent_separation_m"] is not None])),
            "mean_heading_controller_saturation_rate": float(np.mean([
                e["heading_controller_saturation_rate"] for e in episodes])),
            "mean_commanded_speed_mps": float(np.mean([
                e["mean_commanded_speed_mps"] for e in episodes])),
            "mean_absolute_relative_heading_action": float(np.mean([
                e["mean_absolute_relative_heading_action"] for e in episodes])),
            "mean_both_low_speed_fraction": float(np.mean([
                e["both_low_speed_fraction"] for e in episodes])),
            "deadlock_episode_count": sum(e["deadlock_count"] > 0 for e in episodes),
            "mean_deadlock_duration_s": float(np.mean([
                e["deadlock_duration_s"] for e in episodes])),
            "near_miss_steps": sum(e["near_miss_steps"] for e in episodes),
            "episodes": episodes}
