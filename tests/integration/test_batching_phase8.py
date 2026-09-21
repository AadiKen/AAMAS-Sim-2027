import itertools

import pytest
import torch

from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.batching.grouping import group_components
from bcod_sim.batching.population import flatten_population
from bcod_sim.batching.reduction import Contribution, deterministic_segmented_sum
from bcod_sim.collision.shapes import Sphere
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.core.lifecycle import DirectAction
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.rl.centralized_state import CentralizedStateContract
from bcod_sim.rl.pettingzoo_env import PettingZooParallelEnv
from bcod_sim.rl.vector_env import VectorEnvironment


def engine(*, experiment_id="batch", spawn=0, max_steps=20):
    registry = Registry()
    registry.register("vessel", "vessel", "1", {}, "test")
    registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": "agent",
        "target_ned_m": [100, 0, 0], "radius_m": 0.01}, "test")
    config = resolve({
        "schema_version": 1, "experiment": {"id": experiment_id, "seed": 4},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.1, "dynamics_substeps": 1,
                       "policy_every_n_master_steps": 2, "max_master_steps": max_steps},
        "world": {"source": {"kind": "parametric"}, "environment": {
            "current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]}, "waves": {"kind": "calm"},
            "visibility_m": 1000},
            "bathymetry": {"kind": "flat", "bottom_ned_z_m": 100, "vertical_datum": "MSL"}},
        "vessels": [{"instance_id": "agent", "definition": "vessel@1",
                     "controller": {"mode": "direct_actuator"},
                     "spawn": {"ned_m": [spawn, 0, 0], "rpy_rad": [0, 0, 0]}}],
        "task": {"type": "waypoint@1", "reward": {"individual_weight": 1, "team_weight": 0},
                 "disabled_agent_behavior": "deactivate_keep_physical"}}, registry)
    v = lambda values: torch.tensor(values, dtype=torch.float64)
    mass = MassProperties(10, v((0, 0, 0)), torch.diag(v((4, 5, 6))),
                          torch.zeros((6, 6), dtype=torch.float64))
    plant = Plant6(mass, Damping(v((0,)*6), v((0,)*6)), Hydrostatics(10*9.80665, v((0, 0, 0))),
                   OperatingEnvelope(v((1e4,)*6), max_substep_s=0.2))
    prop = FixedThruster(ActuatorConfig("prop", 0, 1, (0, 0, 0), (1, 0, 0, 0), Bounds(-100, 100)))
    return EpisodeEngine(config, (EpisodeVessel("agent", 1, plant, Sphere(0.2), (prop,), (),
                                                ExplicitZeroLoads()),))


def action(value):
    return DirectAction((("prop", ThrustCommand(value)),))


def rollout(env_ids, thrusts, *, extra=False):
    engines = {env_id: engine(experiment_id=f"e{env_id}", spawn=env_id) for env_id in env_ids}
    if extra:
        engines[99] = engine(experiment_id="unrelated", spawn=-20)
    vector = VectorEnvironment(engines)
    vector.reset()
    result = vector.step({env_id: {"agent": action(thrusts.get(env_id, 0))} for env_id in engines})
    return {env_id: result.frames[env_id].states["agent"].position_ned.clone() for env_id in env_ids}, vector


def test_batch_order_size_and_unrelated_environment_invariance():
    baseline, _ = rollout((2,), {2: 10})
    reordered, _ = rollout((7, 2), {2: 10, 7: -8})
    with_extra, _ = rollout((2,), {2: 10}, extra=True)
    assert torch.equal(baseline[2], reordered[2])
    assert torch.equal(baseline[2], with_extra[2])


def test_environment_action_isolation_and_flat_population_identity():
    states, vector = rollout((3, 8), {3: 10, 8: 0})
    assert states[3][0].item() > 3
    assert states[8][0].item() == 8
    population = vector.population()
    assert population.vessels.env_id.tolist() == [3, 8]
    assert population.vessels.vessel_id.tolist() == [1, 1]
    assert population.row_by_identity == {(3, 1): 0, (8, 1): 1}
    groups = group_components(vector.engines)
    assert len(groups) == 1 and [x.env_id for x in groups[0].identities] == [3, 8]


def test_deterministic_reduction_is_order_invariant_and_segmented():
    rows = (
        Contribution(1, 2, "z", torch.tensor([1e16, 2.], dtype=torch.float64)),
        Contribution(0, 1, "a", torch.tensor([4., 8.], dtype=torch.float64)),
        Contribution(1, 2, "a", torch.tensor([-1e16, 3.], dtype=torch.float64)),
        Contribution(1, 2, "m", torch.tensor([1., 5.], dtype=torch.float64)),
    )
    expected = deterministic_segmented_sum(rows)
    for permutation in itertools.permutations(rows):
        actual = deterministic_segmented_sum(permutation)
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert expected[(0, 1)].tolist() == [4, 8]
    assert expected[(1, 2)].tolist() == [0, 10]


def test_parallel_adapter_policy_interval_spaces_and_centralized_state():
    core = engine(max_steps=2)
    observation_space, action_space = object(), object()
    adapter = PettingZooParallelEnv(core, observation_spaces={"agent": observation_space},
        action_spaces={"agent": action_space},
        centralized_state=CentralizedStateContract(("position_ned", "nu_body", "rl_active")))
    observations, infos = adapter.reset(seed=6)
    assert adapter.observation_space("agent") is observation_space
    assert adapter.action_space("agent") is action_space
    assert "position_ned" not in observations["agent"]
    critic = adapter.state()
    assert set(critic) == {"position_ned", "nu_body", "rl_active", "vessel_id"}
    _, rewards, terminations, truncations, infos = adapter.step({"agent": action(0)})
    assert rewards == {"agent": 0}
    assert terminations == {"agent": False} and truncations == {"agent": True}
    assert infos["agent"]["master_step"] == 2
    assert adapter.agents == []


def test_vector_and_parallel_adapters_fail_closed_on_missing_actions():
    core = engine()
    vector = VectorEnvironment({4: core})
    vector.reset()
    with pytest.raises(PhysicalValidationError):
        vector.step({})
    adapter = PettingZooParallelEnv(engine(), observation_spaces={"agent": object()},
                                    action_spaces={"agent": object()})
    adapter.reset()
    with pytest.raises(PhysicalValidationError):
        adapter.step({})
