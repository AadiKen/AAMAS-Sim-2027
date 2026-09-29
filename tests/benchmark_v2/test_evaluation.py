"""Deterministic V2 evaluation and fixed geometry bank."""
import json
import math
import random

import numpy as np
import pytest

from bcod_sim.benchmark_v2.backend import BackendFrame, Truth, VesselReading
from bcod_sim.benchmark_v2.config import TaskConfig
from bcod_sim.benchmark_v2.evaluate import evaluate_bank, evaluate_scenario
from bcod_sim.benchmark_v2.scenarios import case_a_reference, case_a_validation, Scenario


class StraightModel:
    def predict(self, observation, deterministic=False):
        assert deterministic is True
        # Deliberately consume global RNG; evaluator must restore it.
        random.random()
        np.random.random()
        return np.array([0.0, 0.0], dtype=np.float32), None


class FakeEnv:
    config = TaskConfig(deadline_steps=3)

    def __init__(self):
        self.closed = False
        self.actions = []

    def reset(self, *, seed=None, options=None):
        random.random()
        np.random.random()
        self.scenario = options["scenario"]
        self.x = self.scenario.starts[0].x_m
        self.y = self.scenario.starts[0].y_m
        self.heading = self.scenario.starts[0].heading_rad
        self.steps = 0
        return np.zeros(48, dtype=np.float32), self._info()

    def _info(self):
        return {"truth": Truth(self.x, self.y, self.heading),
                "reading": VesselReading(self.x, self.y, self.heading, 1., 0.)}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        self.x += math.cos(self.heading)
        self.y += math.sin(self.heading)
        info = self._info()
        info.update({"terminal_reason": "deadline" if self.steps == 3 else None,
                     "reward_components": {"vessel_0": {
                         "potential_shaping": 1.0, "goal": 0., "collision": 0., "step": -0.01}}})
        return np.zeros(48, dtype=np.float32), 0.99, self.steps == 3, False, info

    def close(self):
        self.closed = True


def test_reference_and_validation_geometry():
    reference = case_a_reference()
    assert reference.starts[0].x_m == -20
    assert reference.starts[0].y_m == 0
    assert reference.goals == ((20., 0.),)
    assert reference.starts[0].heading_rad == 0
    assert not reference.obstacles
    cases = case_a_validation()
    assert len(cases) == len({case.geometry_hash() for case in cases}) == 10
    assert reference.geometry_hash() not in {case.geometry_hash() for case in cases}
    for case in cases:
        start = case.starts[0]
        goal = case.goals[0]
        assert abs(math.atan2(goal[1] - start.y_m, goal[0] - start.x_m)
                   - start.heading_rad) < 1e-12


def test_geometry_hash_ignores_seed():
    assert case_a_reference(1).geometry_hash() == case_a_reference(999).geometry_hash()
    assert case_a_reference().geometry_hash() != case_a_validation()[0].geometry_hash()


def test_evaluate_scenario_reports_metrics_and_trajectory(tmp_path):
    env = FakeEnv()
    path = tmp_path / "reference.json"
    result = evaluate_scenario(StraightModel(), env, case_a_reference(),
                               checkpoint_step=2048, trajectory_path=path)
    assert result["checkpoint_step"] == 2048
    assert result["episode_length"] == 3
    assert result["deadline"] and not result["success"] and not result["collision"]
    assert result["final_distance_to_goal_m"] == pytest.approx(37.)
    assert result["path_length_m"] == pytest.approx(3.)
    assert result["mean_speed_mps"] == 1.
    assert result["reward_components"] == {"potential_shaping": 3., "goal": 0.,
                                            "collision": 0., "step": -.03}
    saved = json.loads(path.read_text())
    assert len(saved["trajectory"]) == 4
    assert saved["trajectory"][0]["truth"]["x_m"] == -20
    assert all(row["action"] == [0., 0.] for row in saved["trajectory"][1:])


def test_evaluation_restores_process_rng_and_closes_env():
    random.seed(17)
    np.random.seed(17)
    before_python, before_numpy = random.getstate(), np.random.get_state()
    envs = []

    def factory():
        env = FakeEnv()
        envs.append(env)
        return env

    result = evaluate_bank(StraightModel(), factory, case_a_validation())
    assert result["scenario_count"] == 10
    assert result["deadline_count"] == 10
    assert len(envs) == 1 and envs[0].closed
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    assert np.array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]


def test_duplicate_geometry_is_rejected_before_env_creation():
    def fail_factory():
        raise AssertionError("should not create an environment")

    with pytest.raises(ValueError, match="repeats scenario geometry"):
        evaluate_bank(StraightModel(), fail_factory, (case_a_reference(1), case_a_reference(2)))


def test_evaluator_accepts_real_gym_info_contract():
    from bcod_sim.benchmark_v2.gym_env import NavigationGymEnv

    class Backend:
        def reset(self, scenario, seed):
            self.x = scenario.starts[0].x_m
            self.closed = False
            return self.frame()

        def frame(self):
            return BackendFrame({"vessel_0": VesselReading(self.x, 0., 0., 0., 0.)},
                                {"vessel_0": Truth(self.x, 0., 0.)})

        def step(self, commands, dt_s):
            self.x += commands["vessel_0"][0] * dt_s
            return self.frame()

        def close(self):
            self.closed = True

    backend = Backend()
    env = NavigationGymEnv(TaskConfig(deadline_steps=2), backend=backend)
    result = evaluate_scenario(StraightModel(), env, case_a_reference())
    assert result["deadline"] and result["episode_length"] == 2
    assert result["reward_components"]["potential_shaping"] > 0
    env.close()
    assert backend.closed
