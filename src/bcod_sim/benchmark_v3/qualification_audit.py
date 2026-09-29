"""Independent replay checks for the prescribed S1 failure audit."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .gym_env import NavigationGymEnv
from .runner import load_policy, read_config, write_json
from .scenarios import s1_test_bank, validate_s1_scenario
from .scripted import ObstacleGeometricController


def _replay(policy, task, scenario):
    env = NavigationGymEnv(task)
    try:
        observation, _ = env.reset(seed=scenario.seed, options={"scenario": scenario})
        initial_distance = math.dist((scenario.starts[0].x_m, scenario.starts[0].y_m),
                                     scenario.goals[0])
        discounted = {key: 0. for key in ("potential_shaping", "goal", "collision", "step")}
        shaping_discounted = 0.
        for t in range(task.deadline_steps + 1):
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            components = info["reward_components"]["vessel_0"]
            assert math.isclose(reward, sum(components.values()), abs_tol=1e-8)
            for key, value in components.items():
                discounted[key] += task.gamma ** t * value
            shaping_discounted += task.gamma ** t * components["potential_shaping"]
            if terminated or truncated:
                break
        else:
            raise AssertionError("No S1 termination at deadline")
        assert math.isclose(shaping_discounted, task.progress_weight * initial_distance,
                            abs_tol=1e-5), (shaping_discounted, initial_distance)
        reason = info["terminal_reason"]
        assert reason in ("success", "collision", "deadline")
        assert (reason == "success") == bool(info["fleet_success"])
        assert (reason == "collision") == bool(info["fleet_failure_due_to_collision"])
        assert (reason == "deadline") == (t + 1 >= task.deadline_steps and
                                             not info["fleet_success"] and
                                             not info["fleet_failure_due_to_collision"])
        return {"reason": reason, "steps": t + 1,
                "minimum_goal_distance_m": info["minimum_goal_distance_m"]["vessel_0"],
                "minimum_obstacle_clearance_m": info["minimum_obstacle_clearance_m"],
                "discounted_components": discounted,
                "discounted_return": sum(discounted.values()),
                "telescoping_error": shaping_discounted - task.progress_weight * initial_distance}
    finally:
        env.close()


def audit_failed_cases(config_path: Path, run_root: Path, output: Path):
    _, task = read_config(config_path, backend="kinematic")
    bank = s1_test_bank()
    records = []
    for seed in (44, 55, 66):
        run = run_root / f"D-kinematic-seed{seed}"
        report = json.loads((run / "test" / "best-dev-once.json").read_text())
        model = load_policy(run / "best-dev.zip", task)
        for index, episode in enumerate(report["episodes"]):
            if episode["success"]:
                continue
            scenario = bank[index]
            assert episode["scenario_hash"] == scenario.geometry_hash()
            validate_s1_scenario(scenario)
            failed = _replay(model, task, scenario)
            scripted = _replay(ObstacleGeometricController(), task, scenario)
            assert failed["reason"] == episode["terminal_reason"]
            if scripted["reason"] != "success":
                raise AssertionError(f"Scripted case {index} failed: {scripted['reason']}")
            if scripted["discounted_return"] <= failed["discounted_return"]:
                raise AssertionError(f"Reward inversion in case {index}, seed {seed}")
            records.append({"seed": seed, "case_index": index,
                            "scenario_hash": scenario.geometry_hash(),
                            "geometry_feasible": True,
                            "failed_policy": failed, "scripted_success": scripted})
    result = {"classification": "TRAINING/GENERALIZATION MACHINERY ISSUE",
              "failed_replays": len(records), "all_scripted_success": True,
              "all_reward_orderings_correct": True, "all_shaping_identities_hold": True,
              "records": records}
    write_json(output, result)
    return result
