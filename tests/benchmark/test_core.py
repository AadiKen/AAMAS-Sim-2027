import math

import pytest

from bcod_sim.benchmark.core import (Benchmark, BenchmarkConfig, NAMES, Truth,
                                     VesselReading, generate_scenario, Scenario)


class FakeAdapter:
    def reset(self, scenario, seed):
        self.scenario = scenario
        self.positions = {name: Truth(s.x_m, s.y_m, s.heading_rad)
                          for name, s in zip(NAMES, scenario.starts)}
        return self.readings(), self.positions.copy()

    def readings(self):
        return {name: VesselReading(p.x_m, p.y_m, p.heading_rad, 0, 0,
                                   tuple((q.x_m, q.y_m) for other, q in self.positions.items() if other != name),
                                   tuple((o.x_m, o.y_m, o.radius_m) for o in self.scenario.obstacles))
                for name, p in self.positions.items()}

    def step(self, actions):
        for name, (speed, yaw) in actions.items():
            p = self.positions[name]
            self.positions[name] = Truth(p.x_m + speed * math.cos(p.heading_rad),
                                         p.y_m + speed * math.sin(p.heading_rad), p.heading_rad + yaw)
        return self.readings(), self.positions.copy()

    def close(self):
        pass


def test_scenario_replay_and_portable_file(tmp_path):
    a = generate_scenario(42)
    assert a == generate_scenario(42)
    assert a != generate_scenario(43)
    assert a != generate_scenario(42, "nominal")
    path = tmp_path / "scenario.json"
    a.save(path)
    assert Scenario.load(path) == a


def test_observation_action_reward_and_timeout():
    config = BenchmarkConfig(max_steps=2)
    env = Benchmark(FakeAdapter(), config)
    obs = env.reset(generate_scenario(7))
    assert all(v.shape == (47,) and abs(v).max() <= 1 for v in obs.values())
    with pytest.raises(ValueError):
        env.step({name: (2, 0) for name in NAMES})
    _, reward, done, truncated, info = env.step({name: (0, 0) for name in NAMES})
    assert all(value == -config.step_penalty for value in reward.values())
    assert not done and not truncated and info["steps"] == 1
    _, _, done, truncated, _ = env.step({name: (0, 0) for name in NAMES})
    assert not done and truncated


def test_progress_goal_and_collision():
    env = Benchmark(FakeAdapter())
    scenario = generate_scenario(4)
    env.reset(scenario)
    name = NAMES[0]
    p = env.truth[name]
    goal = scenario.goals[0]
    heading = math.atan2(goal[1] - p.y_m, goal[0] - p.x_m)
    env.adapter.positions[name] = Truth(p.x_m, p.y_m, heading)
    env.truth[name] = env.adapter.positions[name]
    _, rewards, _, _, _ = env.step({n: (1, 0) if n == name else (0, 0) for n in NAMES})
    assert rewards[name] > 0
    env.adapter.positions[name] = Truth(*goal, heading)
    _, rewards, _, _, info = env.step({n: (0, 0) for n in NAMES})
    assert rewards[name] >= env.config.goal_bonus - env.config.step_penalty
    assert info["per_agent_success"][name]
    env.reset(scenario)
    env.adapter.positions[NAMES[1]] = env.adapter.positions[name]
    _, rewards, done, _, info = env.step({n: (0, 0) for n in NAMES})
    assert done and info["collision_count"] >= 2
    assert rewards[name] <= -env.config.collision_penalty


def test_swept_collision_catches_pass_through():
    env = Benchmark(FakeAdapter())
    scenario = generate_scenario(23)
    env.reset(scenario)
    obstacle = scenario.obstacles[0]
    name = NAMES[0]
    start = Truth(obstacle.x_m - obstacle.radius_m - 2., obstacle.y_m, 0.)
    env.truth[name] = start
    # Move across the complete circle in one native transition. Endpoint is clear.
    env.adapter.positions[name] = Truth(obstacle.x_m + obstacle.radius_m + 2., obstacle.y_m, 0.)
    env.truth[name] = start
    _, rewards, done, _, info = env.step({n: (0., 0.) for n in NAMES})
    assert done and info["collision_count"] >= 1
    assert rewards[name] < 0
