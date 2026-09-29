"""Backend-independent deterministic evaluation, using task outcomes as primary metrics."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .gym_env import NavigationGymEnv
from .control.actions import ACTION_SCHEMAS
from .control.high_level import CONTROLLER_VERSION


@contextmanager
def preserve_rng():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def evaluate_bank(model, task_config, scenarios, *, checkpoint_step: int,
                  trajectory_dir: Path | None = None,
                  trajectory_indices: tuple[int, ...] = (0, 1, 2)) -> dict:
    scenarios = tuple(scenarios)
    hashes = [case.geometry_hash() for case in scenarios]
    if not scenarios or len(set(hashes)) != len(hashes):
        raise ValueError("Evaluation bank must have distinct geometries")
    episodes = []
    with preserve_rng():
        env = NavigationGymEnv(task_config)
        try:
            for index, scenario in enumerate(scenarios):
                obs, info = env.reset(seed=scenario.seed, options={"scenario": scenario})
                initial_distance = math.dist((info["truth"].x_m, info["truth"].y_m), scenario.goals[0])
                records = []
                total_reward = 0.0
                component_totals = {}
                commanded_speeds = []
                achieved_speeds = []
                commanded_yaws = []
                achieved_yaws = []
                for step in range(1, task_config.deadline_steps + 2):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    for key, value in info["reward_components"]["vessel_0"].items():
                        component_totals[key] = component_totals.get(key, 0.) + float(value)
                    command = info["physical_command"]
                    reading = info["reading"]
                    truth = info["truth"]
                    commanded_speeds.append(float(command["speed_mps"]))
                    achieved_speeds.append(float(reading.surge_mps))
                    commanded_yaws.append(float(command["yaw_rate_radps"]))
                    achieved_yaws.append(float(reading.yaw_rps))
                    if trajectory_dir is not None and index in trajectory_indices:
                        goal = scenario.goals[0]
                        bearing = math.atan2(goal[1] - reading.y_m, goal[0] - reading.x_m) - reading.heading_rad
                        nearest = min(scenario.obstacles,
                                      key=lambda o: math.dist((reading.x_m, reading.y_m),
                                                               (o.x_m, o.y_m))) if scenario.obstacles else None
                        obstacle_bearing = (math.atan2(nearest.y_m - reading.y_m,
                                                       nearest.x_m - reading.x_m) - reading.heading_rad
                                            if nearest is not None else None)
                        records.append({"step": step, "truth": asdict(truth),
                                        "time_s": step * task_config.dt_s,
                                        "reading": asdict(reading),
                                        "physical_6dof": info["backend_diagnostics"].get(
                                            "physical_6dof", {}).get("vessel_0"),
                                        "actuator_saturated": info["backend_diagnostics"].get(
                                            "saturated", {}).get("vessel_0"),
                                        "goal_bearing_rad": math.atan2(math.sin(bearing), math.cos(bearing)),
                                        "commanded_speed_mps": command["speed_mps"],
                                        "achieved_surge_mps": reading.surge_mps,
                                        "commanded_yaw_rate_radps": command["yaw_rate_radps"],
                                        "achieved_yaw_rate_radps": reading.yaw_rps,
                                        "distance_to_goal_m": math.dist((truth.x_m, truth.y_m), goal),
                                        "nearest_obstacle_range_m": (math.dist(
                                            (reading.x_m, reading.y_m), (nearest.x_m, nearest.y_m))
                                            if nearest is not None else None),
                                        "nearest_obstacle_bearing_rad": (
                                            math.atan2(math.sin(obstacle_bearing), math.cos(obstacle_bearing))
                                            if obstacle_bearing is not None else None),
                                        "minimum_obstacle_clearance_m": info["minimum_obstacle_clearance_m"],
                                        "action": np.asarray(action).tolist(),
                                        "reward": float(reward),
                                        "reward_components": info["reward_components"]["vessel_0"],
                                        "termination_reason": info["terminal_reason"]})
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("Evaluation exceeded task deadline")
                reason = info["terminal_reason"]
                path = info["path_length_m"]["vessel_0"]
                episode = {
                    "scenario_hash": scenario.geometry_hash(), "scenario_seed": scenario.seed,
                    "success": reason == "success", "collision": reason == "collision",
                    "deadline": reason == "deadline", "terminal_reason": reason,
                    "episode_length": step, "initial_distance_m": initial_distance,
                    "minimum_goal_distance_m": info["minimum_goal_distance_m"]["vessel_0"],
                    "final_goal_distance_m": info["final_goal_distance_m"]["vessel_0"],
                    "path_length_m": path,
                    "path_efficiency": (info["path_efficiency"]["vessel_0"]
                                        if reason == "success" else 0.),
                    "minimum_separation_m": info["minimum_separation_m"],
                    "minimum_obstacle_clearance_m": info["minimum_obstacle_clearance_m"],
                    "control_effort": info["control_effort"]["vessel_0"],
                    "mean_commanded_speed_mps": float(np.mean(commanded_speeds)),
                    "mean_achieved_surge_mps": float(np.mean(achieved_speeds)),
                    "mean_commanded_yaw_rate_radps": float(np.mean(commanded_yaws)),
                    "mean_achieved_yaw_rate_radps": float(np.mean(achieved_yaws)),
                    "reward": total_reward,
                    "reward_components": component_totals,
                }
                episodes.append(episode)
                if trajectory_dir is not None and index in trajectory_indices:
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    (trajectory_dir / f"step-{checkpoint_step}-case-{index}.json").write_text(
                        json.dumps({"episode": episode, "trajectory": records}, indent=2) + "\n")
        finally:
            env.close()
    n = len(episodes)
    successes = [e for e in episodes if e["success"]]
    mean = lambda key: float(np.mean([e[key] for e in episodes]))
    mean_optional = lambda key: (float(np.mean([e[key] for e in episodes if e[key] is not None]))
                                 if any(e[key] is not None for e in episodes) else None)
    return {
        "checkpoint_step": checkpoint_step, "scenario_count": n,
        "action_mode": task_config.action_mode,
        "action_schema_version": ACTION_SCHEMAS[task_config.action_mode],
        "max_surge_mps": task_config.max_surge_mps,
        "max_yaw_rps": task_config.max_yaw_rps,
        "max_heading_offset_rad": task_config.max_heading_offset_rad,
        "heading_controller": {"version": CONTROLLER_VERSION, "k_p": task_config.heading_k_p},
        "scenario_bank_hash": hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest(),
        "success_count": len(successes), "success_rate": len(successes) / n,
        "collision_count": sum(e["collision"] for e in episodes),
        "collision_rate": sum(e["collision"] for e in episodes) / n,
        "deadline_count": sum(e["deadline"] for e in episodes),
        "deadline_rate": sum(e["deadline"] for e in episodes) / n,
        "mean_episode_length": mean("episode_length"),
        "mean_success_completion_steps": (float(np.mean([e["episode_length"] for e in successes]))
                                          if successes else None),
        "mean_minimum_goal_distance_m": mean("minimum_goal_distance_m"),
        "mean_final_goal_distance_m": mean("final_goal_distance_m"),
        "mean_path_length_m": mean("path_length_m"),
        "mean_path_efficiency": mean("path_efficiency"),
        "mean_control_effort": mean("control_effort"),
        "mean_commanded_speed_mps": mean("mean_commanded_speed_mps"),
        "mean_achieved_surge_mps": mean("mean_achieved_surge_mps"),
        "mean_commanded_yaw_rate_radps": mean("mean_commanded_yaw_rate_radps"),
        "mean_achieved_yaw_rate_radps": mean("mean_achieved_yaw_rate_radps"),
        "completion_steps_median": (float(np.median([e["episode_length"] for e in successes]))
                                    if successes else None),
        "completion_steps_p90": (float(np.percentile([e["episode_length"] for e in successes], 90))
                                 if successes else None),
        "final_goal_distance_median": float(np.median([e["final_goal_distance_m"] for e in episodes])),
        "final_goal_distance_p90": float(np.percentile([e["final_goal_distance_m"] for e in episodes], 90)),
        "path_efficiency_median": float(np.median([e["path_efficiency"] for e in episodes])),
        "path_efficiency_p10": float(np.percentile([e["path_efficiency"] for e in episodes], 10)),
        "path_efficiency_p90": float(np.percentile([e["path_efficiency"] for e in episodes], 90)),
        "mean_minimum_separation_m": mean_optional("minimum_separation_m"),
        "mean_minimum_obstacle_clearance_m": mean_optional("minimum_obstacle_clearance_m"),
        "minimum_obstacle_clearance_median": (float(np.median([e["minimum_obstacle_clearance_m"]
            for e in episodes if e["minimum_obstacle_clearance_m"] is not None]))
            if any(e["minimum_obstacle_clearance_m"] is not None for e in episodes) else None),
        "minimum_obstacle_clearance_p10": (float(np.percentile([e["minimum_obstacle_clearance_m"]
            for e in episodes if e["minimum_obstacle_clearance_m"] is not None], 10))
            if any(e["minimum_obstacle_clearance_m"] is not None for e in episodes) else None),
        "mean_reward_diagnostic_only": mean("reward"), "episodes": episodes,
    }


def task_metric_key(report: dict, *, stage: str = "S0") -> tuple:
    """Predeclared best-model order: success, safety, completion, efficiency."""
    completion = report["mean_success_completion_steps"]
    return (report["success_rate"], -report["collision_rate"],
            *((-report["deadline_rate"],) if stage == "S1" else ()),
            -completion if completion is not None else -1e9,
            report["mean_path_efficiency"])
