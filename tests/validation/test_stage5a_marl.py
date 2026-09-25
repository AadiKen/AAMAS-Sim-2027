"""Stage 5A deterministic MARL interface checks."""

import math
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict as DictSpace, Discrete
from pettingzoo.test import parallel_api_test, parallel_seed_test

from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.autopilot import HeadingSpeedAutopilot, HighLevelCommand
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.collision.shapes import Sphere
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.core.errors import CommandBoundsError, ConfigSchemaError, PhysicalValidationError
from bcod_sim.core.lifecycle import DirectAction
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.rl.observation import ObservationContract, ObservationField
from bcod_sim.rl.pettingzoo_env import PettingZooParallelEnv
from bcod_sim.rl.vector_env import VectorEnvironment
from bcod_sim.logging.recorder import RunRecorder
from bcod_sim.scenario.generator import ScenarioTemplate
from bcod_sim.sensors.base import SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.configs import GPSErrorModel
from bcod_sim.sensors.ground_truth import GroundTruthState


def make_world(names=("a",), *, order=None, sensors=False, max_steps=20,
               goal=(1000., 0., 0.), spawn_by_name=None, sensor_by_name=None,
               extra_actuator=(), mass_by_name=None, controller_by_name=None,
               static_entities=(), bottom=100., seabed_collision=False,
               task_payload=None, reward_weights=None, environment_config=None,
               environment_loads_by_name=None, heading_by_name=None,
               thrust_bounds_by_name=None, actuator_rate_by_name=None,
               inertia_by_name=None, collision_shape_by_name=None,
               sensor_latency_by_name=None):
    order = order or names
    spawn_by_name = spawn_by_name or {}
    sensor_by_name = sensor_by_name or {}
    mass_by_name = mass_by_name or {}
    controller_by_name = controller_by_name or {}
    environment_loads_by_name = environment_loads_by_name or {}
    heading_by_name = heading_by_name or {}
    thrust_bounds_by_name = thrust_bounds_by_name or {}
    actuator_rate_by_name = actuator_rate_by_name or {}
    inertia_by_name = inertia_by_name or {}
    collision_shape_by_name = collision_shape_by_name or {}
    sensor_latency_by_name = sensor_latency_by_name or {}
    registry = Registry()
    registry.register("vessel", "vessel", "1", {}, "stage5a")
    registry.register("task", "waypoint", "1", task_payload or
        {"kind": "waypoint", "agent_id": names[0],
         "target_ned_m": goal, "radius_m": 0.01}, "stage5a")
    config = resolve({
        "schema_version": 1, "experiment": {"id": "stage5a", "seed": 73},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.1,
            "dynamics_substeps": 1, "policy_every_n_master_steps": 1,
            "max_master_steps": max_steps},
        "world": {"source": {"kind": "parametric"}, "environment": environment_config or {
            "current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "waves": {"kind": "calm"}, "visibility_m": 1000},
            "bathymetry": ({"kind": "flat", "bottom_ned_z_m": bottom,
                "vertical_datum": "MSL", "collision": {"enabled": seabed_collision}}
                if bottom is not None else None),
            "static_entities": list(static_entities)},
        "vessels": [{"instance_id": name, "definition": "vessel@1",
            "controller": {"mode": controller_by_name.get(name, "direct_actuator")},
            "spawn": {"ned_m": list(spawn_by_name.get(name, (i * 20., 0, 0))),
                      "rpy_rad": [0, 0, heading_by_name.get(name, 0)]}}
            for i, name in enumerate(names)],
        "task": {"type": "waypoint@1", "reward": reward_weights or
            {"individual_weight": 1, "team_weight": 0},
            "disabled_agent_behavior": "deactivate_keep_physical"}}, registry)
    vec = lambda values: torch.tensor(values, dtype=torch.float64)
    vessels = []
    contracts = {}
    for name in order:
        vessel_id = names.index(name) + 1
        vessel_mass = mass_by_name.get(name, 10)
        mass = MassProperties(vessel_mass, vec((0, 0, 0)),
                              torch.diag(vec(inertia_by_name.get(name, (4, 5, 6)))),
                              torch.zeros((6, 6), dtype=torch.float64))
        plant = Plant6(mass, Damping(vec((0,) * 6), vec((0,) * 6)),
            Hydrostatics(vessel_mass * 9.80665, vec((0, 0, 0))),
            OperatingEnvelope(vec((1e4,) * 6), max_substep_s=0.2))
        prop = FixedThruster(ActuatorConfig("prop", 0, vessel_id, (0, 0, 0),
            (1, 0, 0, 0), Bounds(*thrust_bounds_by_name.get(name, (-100, 100))),
            thrust_rate_limit_nps=actuator_rate_by_name.get(name)))
        actuators = (prop,)
        if name in extra_actuator:
            actuators += (FixedThruster(ActuatorConfig("aux", 0, vessel_id,
                (0, 1, 0), (1, 0, 0, 0), Bounds(-50, 50))),)
        sensor_settings = sensor_by_name.get(name, ("truth" if sensors else "none", 10, 0))
        sensor_kind, rate, noise = sensor_settings[:3]
        dropout = sensor_settings[3] if len(sensor_settings) > 3 else 0
        sensor_seed = sensor_settings[4] if len(sensor_settings) > 4 else 5
        sensor = ()
        if sensor_kind == "truth":
            sensor = (GroundTruthState(SensorConfig("truth", "ground_truth_state", 0,
                vessel_id, (0, 0, 0), (1, 0, 0, 0), rate,
                sensor_latency_by_name.get(name, 0), 0, sensor_seed, "1", "stage5a",
                dropout_probability=dropout)),)
            contracts[name] = ObservationContract((ObservationField(
                "position", "truth", "ground_truth_state", "1", "stage5a",
                "ground_truth", "position_ned_m", (3,), "float64", "m", "NED"),),
                allow_ground_truth=True)
        elif sensor_kind == "gps":
            sensor = (GPS(SensorConfig("gps", "gps", 0, vessel_id, (0, 0, 0),
                (1, 0, 0, 0), rate, sensor_latency_by_name.get(name, 0),
                noise, sensor_seed, "1", "stage5a",
                dropout_probability=dropout),
                origin_wgs84_rad_m=(0, 0, 0), error_model=GPSErrorModel(noise, noise)),)
            contracts[name] = ObservationContract((ObservationField(
                "velocity", "gps", "gps", "1", "stage5a", "physical",
                "velocity_ned_mps", (3,), "float64", "m/s", "NED"),))
        autopilot = (HeadingSpeedAutopilot(10, 5)
                     if controller_by_name.get(name) == "high_level" else None)
        vessels.append(EpisodeVessel(name, vessel_id, plant,
                                     collision_shape_by_name.get(name, Sphere(0.2)), actuators,
                                     sensor, environment_loads_by_name.get(name, ExplicitZeroLoads()),
                                     autopilot=autopilot))
    engine = EpisodeEngine(config, tuple(vessels), observation_contracts=contracts)
    # The adapter accepts opaque declared spaces; native typed actions and observation
    # contracts are validated by the engine. This is an API compliance limitation.
    env = PettingZooParallelEnv(engine, observation_spaces={n: object() for n in names},
        action_spaces={n: object() for n in names})
    return env


def command(value):
    return DirectAction((("prop", ThrustCommand(value)),))


def make_pz_world(names=("a", "b"), *, max_steps=8):
    engine = make_world(names, sensors=True, max_steps=max_steps).engine
    observation_spaces = {name: DictSpace({"position": Box(-np.inf, np.inf,
        shape=(3,), dtype=np.float64)}) for name in names}
    action_spaces = {name: Box(-100, 100, shape=(1,), dtype=np.float64)
                     for name in names}
    return PettingZooParallelEnv(engine, observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        action_decoders={name: lambda value: command(float(value[0])) for name in names},
        observation_encoders={name: lambda raw: {"position": raw["position"].numpy()}
                              for name in names})


def test_pettingzoo_parallel_api_and_spaces():
    env = make_pz_world()
    assert env.possible_agents == ["a", "b"]
    assert set(env.action_spaces) == set(env.observation_spaces) == {"a", "b"}
    obs, infos = env.reset(seed=5)
    assert set(obs) == set(infos) == set(env.agents)
    assert all(env.observation_space(name).contains(obs[name]) for name in env.agents)
    parallel_api_test(env, num_cycles=8)


def test_pettingzoo_seed_compliance():
    parallel_seed_test(lambda: make_pz_world(max_steps=8), num_cycles=8)


def test_one_agent_terminates_others_continue_and_mixed_time_limit():
    env = make_pz_world(max_steps=3)
    env.reset(seed=5)
    zero = np.array([0.], dtype=np.float64)
    env.step({"a": zero, "b": zero})
    env.disable_agent("a", reason="test_failure")
    obs, rewards, terminations, truncations, infos = env.step({"a": zero, "b": zero})
    assert terminations == {"a": True, "b": False}
    assert truncations == {"a": False, "b": False}
    assert set(obs) == {"a", "b"} and rewards["a"] == 0
    assert infos["a"]["master_step"] == 2
    assert env.agents == ["b"]
    with pytest.raises(PhysicalValidationError):
        env.step({"a": zero, "b": zero})
    obs, rewards, terminations, truncations, _ = env.step({"b": zero})
    assert set(obs) == {"b"}
    assert terminations == {"b": False} and truncations == {"b": True}
    assert rewards["b"] == 0 and env.agents == []


def test_all_agents_terminate_on_goal_once():
    env = make_world(("a", "b"), sensors=True, goal=(0., 0., 0.))
    env.reset()
    obs, rewards, terminations, truncations, _ = env.step({"a": command(0), "b": command(0)})
    assert set(obs) == {"a", "b"}
    assert terminations == {"a": True, "b": True}
    assert truncations == {"a": False, "b": False}
    assert rewards == {"a": 10, "b": 0}
    with pytest.raises(PhysicalValidationError):
        env.step({})


def test_mixed_termination_and_truncation_same_transition():
    env = make_pz_world(max_steps=2)
    zero = np.array([0.], dtype=np.float64)
    env.reset()
    env.step({"a": zero, "b": zero})
    env.disable_agent("a", reason="test_failure")
    obs, rewards, terminations, truncations, _ = env.step({"a": zero, "b": zero})
    assert set(obs) == {"a", "b"}
    assert terminations == {"a": True, "b": False}
    assert truncations == {"a": False, "b": True}
    assert rewards["a"] == 0 and env.agents == []


def test_heterogeneous_vessels_spaces_sensors_actuators_and_order():
    settings = {"a": ("truth", 10, 0), "b": ("gps", 5, 0.2)}
    kwargs = dict(names=("a", "b"), sensor_by_name=settings,
                  extra_actuator=("a",), mass_by_name={"b": 14},
                  controller_by_name={"b": "high_level"})
    left = make_world(order=("a", "b"), **kwargs)
    right = make_world(order=("b", "a"), **kwargs)
    for env in (left, right):
        obs, _ = env.reset(seed=13)
        assert set(obs["a"]) == {"position"} and set(obs["b"]) == {"velocity"}
        assert env.engine.scheduler.intervals[(0, 1, "truth")] == 1
        assert env.engine.scheduler.intervals[(0, 2, "gps")] == 2
    a_action = DirectAction((("prop", ThrustCommand(20)), ("aux", ThrustCommand(-5))))
    b_action = HighLevelCommand(1, 0)
    result_l = left.step({"a": a_action, "b": b_action})
    result_r = right.step({"b": b_action, "a": a_action})
    for name in ("a", "b"):
        assert torch.equal(left.engine.states[name].position_ned,
                           right.engine.states[name].position_ned)
        assert result_l[1][name] == result_r[1][name]
    assert torch.equal(result_l[0]["a"]["position"], left.engine.states["a"].position_ned)
    assert set(left.engine.held_commands) == {(0, 1, "prop"), (0, 1, "aux"), (0, 2, "prop")}
    assert left.engine.latest_packets[(2, "gps")].sample_step == 0
    assert left.engine.latest_packets[(1, "truth")].sample_step == 1
    assert result_l[4]["b"]["observation_freshness"]["gps"]["age_s"] == pytest.approx(0.1)
    assert result_l[4]["a"]["observation_freshness"]["truth"]["age_s"] == 0
    assert torch.equal(result_l[0]["b"]["velocity"],
                       left.engine.latest_packets[(2, "gps")].values["velocity_ned_mps"])


def test_heterogeneous_declared_gym_spaces_and_bounds():
    native = make_world(("a", "b"), sensor_by_name={
        "a": ("truth", 10, 0), "b": ("gps", 5, 0.1)},
        extra_actuator=("a",), controller_by_name={"b": "high_level"})
    env = PettingZooParallelEnv(native.engine,
        observation_spaces={
            "a": DictSpace({"position": Box(-np.inf, np.inf, (3,), dtype=np.float64)}),
            "b": DictSpace({"velocity": Box(-np.inf, np.inf, (3,), dtype=np.float64)})},
        action_spaces={
            "a": Box(np.array([-100., -50.]), np.array([100., 50.]), dtype=np.float64),
            "b": Box(np.array([-10., -math.pi]), np.array([10., math.pi]), dtype=np.float64)},
        action_decoders={
            "a": lambda x: DirectAction((("prop", ThrustCommand(float(x[0]))),
                                          ("aux", ThrustCommand(float(x[1]))))),
            "b": lambda x: HighLevelCommand(float(x[0]), float(x[1]))},
        observation_encoders={
            "a": lambda x: {"position": x["position"].numpy()},
            "b": lambda x: {"velocity": x["velocity"].numpy()}})
    obs, _ = env.reset(seed=5)
    assert all(env.observation_space(name).contains(obs[name]) for name in env.agents)
    assert env.action_space("a").shape == (2,) and env.action_space("b").shape == (2,)
    actions = {"a": np.array([100., -50.]), "b": np.array([1., 0.])}
    obs, _, _, _, _ = env.step(actions)
    assert all(env.observation_space(name).contains(obs[name]) for name in env.agents)
    assert set(env.engine.held_commands) == {(0, 1, "prop"), (0, 1, "aux"), (0, 2, "prop")}
    for invalid in (np.array([0.]), np.array([0., 51.]),
                    np.array([math.nan, 0.]), np.array(["bad", "0"])):
        with pytest.raises(PhysicalValidationError):
            env.step({"a": invalid, "b": actions["b"]})


def test_duplicate_agent_id_rejected_at_config_boundary():
    with pytest.raises(ConfigSchemaError):
        make_world(("a", "a"))


@pytest.mark.parametrize("change", ("spawn", "sensor_rate", "sensor_noise",
                                     "sensor_seed", "sensor_dropout", "controller",
                                     "vessel_config", "actuator", "reward_event"))
def test_cross_agent_isolation_when_only_a_changes(change):
    common = {"names": ("a", "b"),
              "sensor_by_name": {"a": ("gps", 10, 0), "b": ("gps", 5, 0.2)}}
    modified = dict(common)
    if change == "spawn":
        modified["spawn_by_name"] = {"a": (-20., 0., 0.)}
    elif change == "sensor_rate":
        modified["sensor_by_name"] = {"a": ("gps", 5, 0), "b": ("gps", 5, 0.2)}
    elif change == "sensor_noise":
        modified["sensor_by_name"] = {"a": ("gps", 10, 2), "b": ("gps", 5, 0.2)}
    elif change == "sensor_seed":
        modified["sensor_by_name"] = {"a": ("gps", 10, 0.2, 0, 8),
                                      "b": ("gps", 5, 0.2)}
    elif change == "sensor_dropout":
        modified["sensor_by_name"] = {"a": ("gps", 10, 0, 1),
                                      "b": ("gps", 5, 0.2)}
    elif change == "controller":
        modified["controller_by_name"] = {"a": "high_level"}
    elif change == "vessel_config":
        modified["mass_by_name"] = {"a": 14}
    elif change == "actuator":
        modified["extra_actuator"] = ("a",)
    else:
        modified["goal"] = (0., 0., 0.)
    baseline = make_world(**common)
    changed = make_world(**modified)
    base_obs, _ = baseline.reset(seed=21)
    changed_obs, _ = changed.reset(seed=21)
    assert torch.equal(base_obs["b"]["velocity"], changed_obs["b"]["velocity"])
    a_action = (HighLevelCommand(0, 0) if change == "controller" else
        DirectAction((("prop", ThrustCommand(0)), ("aux", ThrustCommand(0))))
        if change == "actuator" else command(0))
    base_result = baseline.step({"a": command(0), "b": command(0)})
    changed_result = changed.step({"a": a_action, "b": command(0)})
    assert torch.equal(baseline.engine.states["b"].position_ned,
                       changed.engine.states["b"].position_ned)
    assert torch.equal(base_result[0]["b"]["velocity"], changed_result[0]["b"]["velocity"])
    assert base_result[1]["b"] == changed_result[1]["b"] == 0
    base_packet = baseline.engine.latest_packets[(2, "gps")]
    changed_packet = changed.engine.latest_packets[(2, "gps")]
    assert base_packet.sample_step == changed_packet.sample_step
    for key in base_packet.values:
        assert torch.equal(base_packet.values[key], changed_packet.values[key])


@pytest.mark.parametrize("kind", ("obstacle", "vessel", "grounding"))
def test_physics_event_attribution_reward_flags_and_log(kind, tmp_path):
    options = {}
    if kind == "obstacle":
        options["static_entities"] = ({"id": "rock", "position_ned_m": (0.3, 0, 0),
            "shape": {"kind": "sphere", "radius_m": 0.2}},)
    elif kind == "vessel":
        options["spawn_by_name"] = {"b": (0.3, 0, 0)}
    else:
        options.update(bottom=0.1, seabed_collision=True)
    env = make_world(("a", "b"), sensors=True, **options)
    initial = env.engine.reset(seed=77)
    env.agents = env._active()
    recorder = RunRecorder(tmp_path, run_id=kind, resolved=env.engine.resolved,
        scenario=env.engine.scenario, initial_frame=initial,
        project_root=Path(__file__).resolve().parents[2])
    obs, rewards, terminations, truncations, infos = env.step({
        "a": command(0), "b": command(0)})
    frame = env.engine.checkpoint()
    # Use the actual transition frame for recorder event bookkeeping.
    event_ids = {event.contact_id for info in infos.values()
                 for event in info["contact_events"]}
    assert event_ids
    assert all(not flag for flag in (*terminations.values(), *truncations.values()))
    a_position = env.engine.states["a"].position_ned
    expected_progress = 1000 - math.sqrt((1000 - float(a_position[0])) ** 2 +
                                         float(a_position[1]) ** 2 + float(a_position[2]) ** 2)
    assert rewards["a"] == pytest.approx(expected_progress)
    assert rewards["b"] == 0
    assert set(obs) == {"a", "b"}
    if kind == "obstacle":
        assert infos["a"]["contact_events"] and not infos["b"]["contact_events"]
    elif kind == "vessel":
        assert infos["a"]["contact_events"] and infos["b"]["contact_events"]
    else:
        assert infos["a"]["grounding"]["contact_active"]
        assert infos["b"]["grounding"]["contact_active"]
    # Reproduce one transition from the same seed for the recorder's frame API.
    mirror = make_world(("a", "b"), sensors=True, **options).engine
    mirror.reset(seed=77)
    transition = mirror.step({"a": command(0), "b": command(0)})
    recorder.record_frame(transition)
    root = recorder.close(final_frame=transition, external_stop=True)
    logged = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert {row["contact_id"] for row in logged if row["event"] == "collision_contact"} == event_ids
    assert frame.master_step == transition.master_step


def test_goal_completion_reward_and_manifest_log(tmp_path):
    env = make_world(("a", "b"), sensors=True, goal=(0., 0., 0.))
    initial = env.engine.reset(seed=31)
    env.agents = env._active()
    recorder = RunRecorder(tmp_path, run_id="goal", resolved=env.engine.resolved,
        scenario=env.engine.scenario, initial_frame=initial,
        project_root=Path(__file__).resolve().parents[2])
    obs, rewards, terminations, truncations, infos = env.step({
        "a": command(0), "b": command(0)})
    assert set(obs) == {"a", "b"}
    assert rewards == {"a": 10, "b": 0}
    assert infos["a"]["reward_components"]["waypoint_success"] == 10
    assert infos["b"]["reward_components"] == {}
    assert all(terminations.values()) and not any(truncations.values())
    mirror = make_world(("a", "b"), sensors=True, goal=(0., 0., 0.)).engine
    mirror.reset(seed=31)
    transition = mirror.step({"a": command(0), "b": command(0)})
    assert transition.task_evaluation.success
    recorder.record_frame(transition)
    root = recorder.close(final_frame=transition)
    assert json.loads((root / "manifest.json").read_text())["termination_reason"] == "task_success"


def test_seeded_random_spawns_replay_and_change_across_seeds():
    env = make_world(("a", "b"), sensors=True)
    env.engine.template = ScenarioTemplate.model_validate({"id": "random_spawn", "version": "1",
        "spawn_overrides": {"a": {"north_m": {"kind": "uniform", "low": -5,
            "high": 5}}}})
    first, _ = env.reset(seed=111)
    env.step({"a": command(0), "b": command(0)})
    repeated, _ = env.reset(seed=111)
    different, _ = env.reset(seed=112)
    assert torch.equal(first["a"]["position"], repeated["a"]["position"])
    assert not torch.equal(first["a"]["position"], different["a"]["position"])
    assert torch.equal(first["b"]["position"], different["b"]["position"])


def test_sensor_dropout_is_per_agent_and_pending_is_reported():
    env = make_world(("a", "b"), sensor_by_name={
        "a": ("gps", 10, 0.2, 1.0), "b": ("gps", 5, 0.2, 0.0)})
    observations, infos = env.reset(seed=10)
    assert observations["a"] == {} and infos["a"]["pending_observation"]
    assert set(observations["b"]) == {"velocity"}
    assert not infos["b"]["pending_observation"]
    _, _, _, _, infos = env.step({"a": command(0), "b": command(0)})
    assert infos["a"]["pending_observation"] and not infos["b"]["pending_observation"]
    assert set(env.engine.latest_packets) == {(2, "gps")}


def test_dropout_with_explicit_space_keeps_shape_and_ownership():
    native = make_world(("a", "b"), sensor_by_name={
        "a": ("gps", 10, 0.2, 1.0), "b": ("gps", 5, 0.2, 0.0)})
    obs_space = DictSpace({"velocity": Box(-np.inf, np.inf, (3,), dtype=np.float64),
                           "valid": Discrete(2)})
    def encode(raw):
        return {"velocity": raw["velocity"].numpy() if "velocity" in raw else
                np.zeros(3, dtype=np.float64), "valid": int("velocity" in raw)}
    env = PettingZooParallelEnv(native.engine,
        observation_spaces={"a": obs_space, "b": obs_space},
        action_spaces={name: Box(-100, 100, (1,), dtype=np.float64)
                       for name in ("a", "b")},
        action_decoders={name: lambda value: command(float(value[0]))
                         for name in ("a", "b")},
        observation_encoders={"a": encode, "b": encode})
    obs, _ = env.reset(seed=3)
    assert obs["a"]["valid"] == 0 and obs["b"]["valid"] == 1
    for _ in range(3):
        obs, _, _, _, infos = env.step({name: np.array([0.], dtype=np.float64)
                                       for name in env.agents})
        assert all(obs_space.contains(value) for value in obs.values())
        assert obs["a"]["valid"] == 0 and obs["b"]["valid"] == 1
        assert infos["a"]["pending_observation"]


def test_reset_matches_fresh_environment():
    env = make_world(("a", "b"), sensors=True)
    first, _ = env.reset(seed=41)
    env.step({"a": command(20), "b": command(-10)})
    reset, _ = env.reset(seed=41)
    fresh = make_world(("a", "b"), sensors=True)
    new, _ = fresh.reset(seed=41)
    for name in ("a", "b"):
        assert torch.equal(first[name]["position"], reset[name]["position"])
        assert torch.equal(reset[name]["position"], new[name]["position"])
    assert env.engine.master_step == 0 and not env.engine.held_commands


def test_agent_order_action_ownership_and_determinism():
    left = make_world(("a", "b"), order=("a", "b"), sensors=True)
    right = make_world(("a", "b"), order=("b", "a"), sensors=True)
    left.reset(seed=9)
    right.reset(seed=9)
    a = left.step({"a": command(30), "b": command(-10)})
    b = right.step({"b": command(-10), "a": command(30)})
    for name in ("a", "b"):
        assert torch.equal(a[0][name]["position"], b[0][name]["position"])
        assert a[1][name] == b[1][name]
        assert torch.equal(a[0][name]["position"], left.engine.states[name].position_ned)
    assert a[0]["a"]["position"][0] > 0
    assert a[0]["b"]["position"][0] < 20
    assert left.engine.last_propulsion["a"][0] > 0
    assert left.engine.last_propulsion["b"][0] < 0


def test_reward_components_and_agent_attribution():
    env = make_world(("a", "b"))
    env.reset()
    _, rewards, _, _, _ = env.step({"b": command(-10), "a": command(30)})
    report = env.engine.task
    assert rewards["a"] > 0 and rewards["b"] == 0
    assert math.isfinite(rewards["a"])
    assert report.previous_distance < 1000


def test_team_and_individual_reward_component_sum():
    options = {"names": ("a", "b"),
        "task_payload": {"kind": "waypoint_team", "target_ned_m": (100, 0, 0),
                         "radius_m": 0.01},
        "reward_weights": {"individual_weight": 0.7, "team_weight": 0.3}}
    env = make_world(**options)
    mirror = make_world(**options).engine
    env.reset(seed=7)
    mirror.reset(seed=7)
    actions = {"a": command(20), "b": command(-10)}
    _, rewards, _, _, _ = env.step(actions)
    frame = mirror.step(actions)
    report = frame.reward
    assert report is not None
    team_sum = sum(report.team_components.values())
    for name in ("a", "b"):
        expected = 0.7 * sum(report.individual_components[name].values()) + 0.3 * team_sum
        assert rewards[name] == pytest.approx(expected)


def test_vector_environment_identity_and_order_independence():
    left = VectorEnvironment({2: make_world().engine, 8: make_world().engine})
    right = VectorEnvironment({8: make_world().engine, 2: make_world().engine})
    for env in (left, right):
        env.reset(seeds={2: 17, 8: 17})
    first = left.step({2: {"a": command(10)}, 8: {"a": command(-10)}})
    second = right.step({8: {"a": command(-10)}, 2: {"a": command(10)}})
    for env_id in (2, 8):
        assert torch.equal(first.frames[env_id].states["a"].position_ned,
                           second.frames[env_id].states["a"].position_ned)
        assert first.rewards[env_id] == second.rewards[env_id]
    assert first.frames[2].states["a"].position_ned[0] > 0
    assert first.frames[8].states["a"].position_ned[0] < 0


def test_terminal_observation_and_truncation():
    env = make_world(("a", "b"), sensors=True, max_steps=1)
    env.reset()
    obs, rewards, terminations, truncations, _ = env.step(
        {"a": command(0), "b": command(0)})
    assert set(obs) == {"a", "b"}
    assert terminations == {"a": False, "b": False}
    assert truncations == {"a": True, "b": True}
    assert env.agents == []
    with pytest.raises(PhysicalValidationError):
        env.step({})


def test_missing_extra_and_unknown_actions_fail():
    env = make_world(("a", "b"))
    env.reset()
    for actions in ({"a": command(0)}, {"a": command(0), "b": command(0),
                     "z": command(0)}, {"a": command(0), "z": command(0)}):
        with pytest.raises(PhysicalValidationError):
            env.step(actions)
    assert env.engine.master_step == 0


def test_sensor_owner_and_reset_packet_isolation():
    env = make_world(("a", "b"), sensors=True)
    obs, _ = env.reset()
    assert obs["a"]["position"].tolist() == [0, 0, 0]
    assert obs["b"]["position"].tolist() == [20, 0, 0]
    assert set(env.engine.latest_packets) == {(1, "truth"), (2, "truth")}
    env.step({"a": command(10), "b": command(0)})
    assert torch.equal(env.engine.states["b"].position_ned,
                       torch.tensor([20., 0., 0.], dtype=torch.float64))


@pytest.mark.parametrize("value", [-101, 101, float("nan"), float("inf")])
def test_invalid_thrust_rejected(value):
    env = make_world()
    env.reset()
    with pytest.raises(CommandBoundsError):
        env.step({"a": command(value)})


def test_long_rollout_deterministic_replay():
    env = make_world(("a", "b", "c", "d"), sensors=True, max_steps=10000,
                     goal=(1e9, 0., 0.))
    checksums = []
    for _ in range(2):
        env.reset(seed=73)
        checksum = 0.0
        for step in range(10000):
            values = (0, 1, -1, 0) if step % 2 else (1, 0, 0, -1)
            observations, rewards, terms, truncs, _ = env.step({
                name: command(value) for name, value in zip(env.agents, values)})
            assert set(observations) == {"a", "b", "c", "d"}
            assert not any(terms.values())
            assert all(math.isfinite(value) for value in rewards.values())
            for name in env.possible_agents:
                position = observations[name]["position"]
                assert position.shape == (3,) and torch.isfinite(position).all()
                checksum += float(position[0])
            if step < 9999:
                assert not any(truncs.values())
            else:
                assert all(truncs.values())
        checksums.append(checksum)
    assert checksums[0] == checksums[1]


STAGE5A_CASES = (
    "test_pettingzoo_parallel_api_and_spaces",
    "test_pettingzoo_seed_compliance",
    "test_one_agent_terminates_others_continue_and_mixed_time_limit",
    "test_all_agents_terminate_on_goal_once",
    "test_mixed_termination_and_truncation_same_transition",
    "test_heterogeneous_vessels_spaces_sensors_actuators_and_order",
    "test_heterogeneous_declared_gym_spaces_and_bounds",
    "test_duplicate_agent_id_rejected_at_config_boundary",
    "test_cross_agent_isolation_when_only_a_changes",
    "test_physics_event_attribution_reward_flags_and_log",
    "test_goal_completion_reward_and_manifest_log",
    "test_seeded_random_spawns_replay_and_change_across_seeds",
    "test_sensor_dropout_is_per_agent_and_pending_is_reported",
    "test_dropout_with_explicit_space_keeps_shape_and_ownership",
    "test_reset_matches_fresh_environment",
    "test_agent_order_action_ownership_and_determinism",
    "test_reward_components_and_agent_attribution",
    "test_team_and_individual_reward_component_sum",
    "test_vector_environment_identity_and_order_independence",
    "test_terminal_observation_and_truncation",
    "test_missing_extra_and_unknown_actions_fail",
    "test_sensor_owner_and_reset_packet_isolation",
    "test_invalid_thrust_rejected",
    "test_long_rollout_deterministic_replay",
)


def test_registry_complete():
    assert set(STAGE5A_CASES) == {name for name in globals()
                                  if name.startswith("test_") and name != "test_registry_complete"}
