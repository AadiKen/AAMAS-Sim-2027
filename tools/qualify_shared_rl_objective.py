"""Prescribed same-map trajectory ranking gate for the shared RL objective."""
import json
from pathlib import Path

from bcod_sim.benchmark.core import Benchmark, BenchmarkConfig, NAMES, Obstacle, Scenario, VesselPose
from bcod_sim.benchmark.qualify_scoring import PrescribedTrajectory, path
from bcod_sim.benchmark.rl_diagnostics import EpisodeDiagnostics


def main():
    cases = [
        ("aggressive_collision", [(72, 0)], 2.0),
        ("slow_safe_detour", [(0, 12), (60, 12), (72, 0)], 0.8),
        ("partial_timeout", [(30, 0)], 1.0),
    ]
    offsets = (-150, -50, 50, 150)
    obstacles = [Obstacle(65, -150, 1)] + [Obstacle(-100, y, 1) for y in offsets] + [Obstacle(-100, 250, 1)]
    scenario = Scenario("nominal", 123, tuple(VesselPose(0, y, 0) for y in offsets),
                        tuple((72, y) for y in offsets), tuple(obstacles))
    rows = []
    for name, route, speed in cases:
        config = BenchmarkConfig(half_width_m=500)
        env = Benchmark(PrescribedTrajectory(path(route, speed)), config)
        env.reset(scenario)
        diagnostics = EpisodeDiagnostics(config.gamma)
        component_steps = []
        for _ in range(config.max_steps):
            _, reward, done, truncated, info = env.step({n: (0, 0) for n in NAMES})
            diagnostics.add(reward, info, env)
            component_steps.append(info["reward_components"])
            if done or truncated:
                break
        result = diagnostics.result(info)
        components = {key: sum(result["reward_decomposition"][n][key] for n in NAMES)
                      for key in ("potential_shaping", "goal", "collision", "step")}
        discounted_components = {key: sum(config.gamma ** t * sum(
            record[n][key] for n in NAMES) for t, record in enumerate(component_steps))
            for key in components}
        rows.append({"trajectory": name, "steps": result["steps"],
                     "fleet_success": result["fleet_success"],
                     "fleet_failure_due_to_collision": result["fleet_failure_due_to_collision"],
                     "undiscounted_fleet_return": result["fleet_return"],
                     "discounted_fleet_return": result["discounted_fleet_return"],
                     "components": {**components, "total": sum(components.values())},
                     "discounted_components": {**discounted_components,
                                               "total": sum(discounted_components.values())}})
    values = {r["trajectory"]: r["discounted_fleet_return"] for r in rows}
    safe, timeout, collision = (values[n] for n in
                                ("slow_safe_detour", "partial_timeout", "aggressive_collision"))
    # A full reward unit of separation avoids treating numerical near-ties as success.
    passed = safe - timeout > 1 and timeout - collision > 1
    report = {"gamma_per_step": BenchmarkConfig().gamma, "gamma_per_second": 0.99,
              "terminal_convention": "Phi=0 on true termination; Phi(final state) retained on timeout",
              "trajectories": rows, "margins": {"success_minus_timeout": safe-timeout,
                                                "timeout_minus_collision": timeout-collision},
              "passed": passed}
    destination = Path("artifacts/benchmark-objective-fix/trajectory-ranking.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Trajectory ranking gate failed; do not train")


if __name__ == "__main__":
    main()
