#!/usr/bin/env python3
"""Development-only pair-scenario diagnosis for a MAPPO checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bcod_sim.benchmark_v3.deadline_harness.algorithms import MAPPO
from bcod_sim.benchmark_v3.deadline_harness.marl import (
    GeometricCoordinator,
    ResidualCoordinatorEnv,
    make_four_vessel_scenario,
)


def episode(scenario, model=None):
    env = ResidualCoordinatorEnv(scenario=scenario)
    obs, _ = env.reset(seed=scenario.seed)
    actions_seen = []
    min_separation = float("inf")
    try:
        for tick in range(env.config.deadline_steps):
            names = list(env.possible_agents)
            if model is None:
                actions = GeometricCoordinator().actions(env)
            else:
                with torch.no_grad():
                    batch = torch.as_tensor(np.stack([obs[n] for n in names]), device=next(model.parameters()).device)
                    values = model.act(batch, deterministic=True)[0].cpu().numpy()
                actions = {name: values[j].astype(np.float32) for j, name in enumerate(names)}
            actions_seen.extend(np.stack(list(actions.values())).tolist())
            obs, _, terms, truncs, infos = env.step(actions)
            readings = list(env._frame.readings.values())
            for i in range(len(readings)):
                for j in range(i + 1, len(readings)):
                    min_separation = min(min_separation, float(np.hypot(
                        readings[i].x_m - readings[j].x_m,
                        readings[i].y_m - readings[j].y_m,
                    )))
            if all(terms.values()) or all(truncs.values()):
                break
        a = np.asarray(actions_seen)
        return {
            "reason": infos[names[0]]["terminal_reason"],
            "ticks": tick + 1,
            "min_separation_m": min_separation,
            "mean_speed_action": float(a[:, 0].mean()),
            "mean_abs_heading_action": float(np.abs(a[:, 1]).mean()),
            "speed_near_zero_fraction": float((a[:, 0] < 0.05).mean()),
            "action_saturation_fraction": float((np.abs(a) > 0.98).mean()),
        }
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = MAPPO().to(a.device)
    model.load_state_dict(payload["model"])
    model.eval()
    rng = np.random.default_rng(81001)
    rows = []
    for i in range(32):
        family = ("easy", "pair", "fourway", "random")[i % 4]
        scenario = make_four_vessel_scenario(rng, family, split="marl-validation")
        if family != "pair":
            continue
        rows.append({
            "development_index": i,
            "scenario_seed": scenario.seed,
            "scenario_hash": scenario.geometry_hash(),
            "mappo": episode(scenario, model),
            "scripted": episode(scenario),
        })
    result = {
        "split": "development",
        "checkpoint": str(a.checkpoint),
        "checkpoint_steps": payload.get("steps"),
        "device": a.device,
        "pair_episodes": len(rows),
        "rows": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pair_episodes": len(rows), "mappo_successes": sum(x["mappo"]["reason"] == "success" for x in rows), "scripted_successes": sum(x["scripted"]["reason"] == "success" for x in rows)}))


if __name__ == "__main__":
    main()
