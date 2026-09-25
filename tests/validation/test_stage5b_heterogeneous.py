"""Stage 5B heterogeneous multi-vessel correctness gates."""

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_stage5a_marl import command, make_world

from bcod_sim.core.environment_loads import LinearEnvironmentLoads
from bcod_sim.core.lifecycle import DirectAction
from bcod_sim.actuators.thruster import ThrustCommand
from bcod_sim.core.errors import CommandBoundsError
from bcod_sim.scenario.generator import ScenarioTemplate
from bcod_sim.world.wake import GaussianWakeEmitter, WakeParameters
from bcod_sim.core.engine import EpisodeEngine
from bcod_sim.rl.observation import ObservationContract, ObservationField
from bcod_sim.sensors.base import SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.lidar import LiDAR
from bcod_sim.sensors.sonar import Sonar
from bcod_sim.collision.broadphase import CollisionBody
from bcod_sim.collision.shapes import Box, Sphere
from bcod_sim.collision.solver import ContactMaterial, resolve_contacts
from bcod_sim.state.vessel_state import VesselState


NAMES = ("a", "b", "c")
SPAWNS = {"a": (0., 0., 0.), "b": (200., 0., 0.), "c": (400., 0., 0.)}
MASSES = {"a": 10, "b": 14, "c": 18, "d": 22}


def fleet(*, names=NAMES, order=None, max_steps=30, environment=None,
          loads=None, spawn=None, sensor=True, sensor_by_name=None, **kwargs):
    sensor_map = sensor_by_name if sensor_by_name is not None else (
        {name: (("truth", 10, 0) if name == "a" else
                ("gps", 5 if name == "b" else 10, 0)) for name in names} if sensor else {})
    return make_world(names, order=order, max_steps=max_steps,
        spawn_by_name={name: (spawn or SPAWNS)[name] for name in names},
        mass_by_name={name: MASSES[name] for name in names},
        inertia_by_name={name: tuple(axis * MASSES[name] / 10 for axis in (4, 5, 6))
                         for name in names},
        sensor_by_name=sensor_map, environment_config=environment,
        environment_loads_by_name=loads or {}, goal=(1e9, 0, 0), **kwargs)


def state_vector(env, name):
    state = env.engine.states[name]
    return torch.cat((state.position_ned, state.q_body_to_ned, state.nu_body))


@pytest.mark.parametrize("target", NAMES)
@pytest.mark.parametrize("forcing", (False, True))
def test_solo_multi_invariance_state_actuator_and_sensor_timing(target, forcing):
    environment = ({"current": {"kind": "uniform", "ned_mps": [0.1, 0.05, 0]},
                    "wind": {"kind": "uniform", "ned_mps": [-0.1, 0.03, 0]},
                    "waves": {"kind": "regular", "height_m": 0.05,
                        "period_s": 5., "direction_rad": 0.},
                    "visibility_m": 1000} if forcing else None)
    loads = ({name: LinearEnvironmentLoads(0.1, 0.1, 0.1, 0.)
              for name in NAMES} if forcing else None)
    solo = fleet(names=(target,), max_steps=12, environment=environment, loads=loads)
    multi = fleet(max_steps=12, environment=environment, loads=loads)
    solo.reset(seed=17)
    multi.reset(seed=17)
    for step in range(8):
        thrust = (10, -5, 3)[step % 3]
        solo.step({target: command(thrust)})
        multi.step({name: command(thrust if name == target else 0)
                    for name in NAMES})
        assert torch.equal(state_vector(solo, target), state_vector(multi, target))
        solo_id = solo.engine.vessels[target].vessel_id
        multi_id = multi.engine.vessels[target].vessel_id
        assert solo.engine.actuator_states[(0, solo_id, "prop")] == \
               multi.engine.actuator_states[(0, multi_id, "prop")]
        assert all(torch.equal(solo.engine._load_terms(solo.engine.vessels[target],
                      solo.engine.states[target], solo.engine.master_step * 0.1)[term],
                  multi.engine._load_terms(multi.engine.vessels[target],
                      multi.engine.states[target], multi.engine.master_step * 0.1)[term])
                   for term in ("current", "wind", "wave", "wake"))
        if target in ("a", "b", "c"):
            sensor_id = "truth" if target == "a" else "gps"
            left = solo.engine.latest_packets[(solo_id, sensor_id)]
            right = multi.engine.latest_packets[(multi_id, sensor_id)]
            assert (left.sample_step, left.delivery_step) == (right.sample_step, right.delivery_step)
            if target == "a":
                assert torch.equal(left.values["position_ned_m"], right.values["position_ned_m"])


def test_parameter_and_actuator_ownership():
    env = fleet(sensor=False, extra_actuator=("a",))
    env.reset()
    actions = {"a": DirectAction((("prop", ThrustCommand(10)),
                                  ("aux", ThrustCommand(-5)))),
               "b": command(10), "c": command(10)}
    env.step(actions)
    assert env.engine.actuator_states[(0, 1, "prop")].thrust_n == 10
    assert env.engine.actuator_states[(0, 1, "aux")].thrust_n == -5
    assert env.engine.actuator_states[(0, 2, "prop")].thrust_n == 10
    assert env.engine.actuator_states[(0, 3, "prop")].thrust_n == 10
    assert env.engine.last_propulsion["a"][0] == 5
    assert env.engine.last_propulsion["a"][5] == 5
    assert env.engine.last_propulsion["b"][0] == env.engine.last_propulsion["c"][0] == 10
    assert env.engine.states["b"].nu_body[0] > env.engine.states["c"].nu_body[0] > 0
    assert env.engine.states["a"].nu_body[5] > 0


def test_thrust_limits_rate_and_reversal_remain_per_vessel():
    env = fleet(sensor=False,
        thrust_bounds_by_name={"a": (-5, 5), "b": (-20, 20), "c": (-50, 50)},
        actuator_rate_by_name={"a": 10., "b": 20.})
    env.reset()
    for value in (0., 5., -5.):
        env.step({"a": command(value), "b": command(20.), "c": command(-50.)})
        assert abs(env.engine.actuator_states[(0, 1, "prop")].thrust_n) <= 5
        assert abs(env.engine.actuator_states[(0, 2, "prop")].thrust_n) <= 20
        assert env.engine.actuator_states[(0, 3, "prop")].thrust_n == -50
    assert env.engine.actuator_states[(0, 1, "prop")].thrust_n == 0.
    assert env.engine.actuator_states[(0, 2, "prop")].thrust_n == 6.
    assert env.engine.last_propulsion["c"][0] == -50
    with pytest.raises(CommandBoundsError):
        env.step({"a": command(6.), "b": command(0.), "c": command(0.)})


@pytest.mark.parametrize("component", range(6))
def test_six_axis_parameter_ownership(component):
    env = fleet(sensor=False)
    env.reset()
    force = torch.zeros(6, dtype=torch.float64)
    force[component] = 1.
    accelerations = {}
    for name in NAMES:
        vessel = env.engine.vessels[name]
        state = env.engine.states[name]
        loads = {key: torch.zeros(6, dtype=torch.float64) for key in
                 ("propulsion", "current", "wind", "wave", "wake", "contact", "manual")}
        loads["manual"] = force
        observed = vessel.plant.acceleration(state, loads)
        expected = torch.linalg.solve(vessel.plant.total_mass, force)
        assert torch.allclose(observed, expected, rtol=0, atol=1e-12)
        accelerations[name] = observed[component].item()
    if component < 3:
        assert accelerations["a"] > accelerations["b"] > accelerations["c"] > 0


@pytest.mark.parametrize("wave", (
    {"kind": "calm"},
    {"kind": "regular", "height_m": 0.1, "period_s": 5., "direction_rad": 0.},
    {"kind": "irregular", "spectrum": "jonswap", "significant_height_m": 0.1,
     "peak_period_s": 5., "direction_rad": 0., "component_count": 8, "seed": 37},
))
def test_environmental_coexistence_and_order(wave):
    environment = {"current": {"kind": "uniform", "ned_mps": [0.2, 0.1, 0]},
                   "wind": {"kind": "uniform", "ned_mps": [-0.3, 0.2, 0]},
                   "waves": wave, "visibility_m": 1000}
    loads = {name: LinearEnvironmentLoads(1., 0.5, 2., 0.) for name in NAMES}
    left = fleet(order=NAMES, environment=environment, loads=loads)
    right = fleet(order=("c", "b", "a"), environment=environment, loads=loads)
    for env in (left, right):
        env.reset(seed=13)
    for step in range(4):
        actions = {name: command((5, 0, -5)[index])
                   for index, name in enumerate(NAMES)}
        left.step(actions)
        right.step(dict(reversed(tuple(actions.items()))))
        for name in NAMES:
            assert torch.equal(state_vector(left, name), state_vector(right, name))
            wrench = left.engine._load_terms(left.engine.vessels[name],
                left.engine.states[name], left.engine.master_step * 0.1)
            assert set(wrench) == {"current", "wind", "wave", "wake"}
            assert all(torch.isfinite(value).all() for value in wrench.values())
            assert torch.linalg.vector_norm(wrench["current"]) > 0
            assert torch.linalg.vector_norm(wrench["wind"]) > 0


@pytest.mark.parametrize("forcing", ("current", "wind", "regular", "irregular",
                                      "current_wind", "current_wave", "wind_wave", "combined"))
def test_environment_forcing_matrix_has_owned_wrenches(forcing):
    current_on = "current" in forcing or forcing == "combined"
    wind_on = "wind" in forcing or forcing == "combined"
    wave_on = "wave" in forcing or forcing in ("regular", "irregular", "combined")
    wave = ({"kind": "irregular", "spectrum": "jonswap", "significant_height_m": 0.1,
             "peak_period_s": 5., "direction_rad": 0.2, "component_count": 8, "seed": 41}
            if forcing == "irregular" else
            {"kind": "regular", "height_m": 0.1, "period_s": 5., "direction_rad": 0.2}
            if wave_on else {"kind": "calm"})
    environment = {"current": {"kind": "uniform", "ned_mps":
        [0.2, 0.1, 0] if current_on else [0, 0, 0]},
                   "wind": {"kind": "uniform", "ned_mps":
        [-0.3, 0.2, 0] if wind_on else [0, 0, 0]},
                   "waves": wave, "visibility_m": 1000}
    # Off-axis field components are intentionally present for heading-frame checks.
    loads = {name: LinearEnvironmentLoads(1., 1., 2., 0.) for name in NAMES}
    env = fleet(environment=environment, loads=loads,
        heading_by_name={"a": 0, "b": math.pi / 2, "c": math.pi / 4}, sensor=False)
    env.reset(seed=11)
    for name in NAMES:
        vessel = env.engine.vessels[name]
        state = env.engine.states[name]
        sample = env.engine.world.sample(state.position_ned[None, :], sim_time_s=0.,
            env_id=0, receiver_vessel_id=vessel.vessel_id)
        wrench = env.engine._load_terms(vessel, state, 0.)
        assert torch.isfinite(torch.cat(tuple(wrench.values()))).all()
        if current_on:
            assert torch.linalg.vector_norm(wrench["current"]) > 0
        else:
            assert torch.equal(wrench["current"], torch.zeros(6, dtype=torch.float64))
        if wind_on:
            assert torch.linalg.vector_norm(wrench["wind"]) > 0
        else:
            assert torch.equal(wrench["wind"], torch.zeros(6, dtype=torch.float64))
        if wave_on:
            assert wrench["wave"][2] == pytest.approx(2. *
                (sample.wave_surface_ned_z_m[0] - state.position_ned[2]).item())
        else:
            assert torch.equal(wrench["wave"], torch.zeros(6, dtype=torch.float64))
    env.step({name: command(0) for name in NAMES})
    assert all(torch.isfinite(state_vector(env, name)).all() for name in NAMES)


def test_per_vessel_wrench_ledgers_balance_under_combined_forcing():
    environment = {"current": {"kind": "uniform", "ned_mps": [0.2, 0.1, 0]},
                   "wind": {"kind": "uniform", "ned_mps": [-0.3, 0.2, 0]},
                   "waves": {"kind": "regular", "height_m": 0.1,
                       "period_s": 5., "direction_rad": 0.2},
                   "visibility_m": 1000}
    env = fleet(environment=environment,
        loads={name: LinearEnvironmentLoads(1., 0.5, 2., 0.) for name in NAMES},
        sensor=False)
    env.reset(seed=9)
    env.step({"a": command(5), "b": command(0), "c": command(-5)})
    for name in NAMES:
        vessel = env.engine.vessels[name]
        state = env.engine.states[name]
        external = env.engine._external(vessel, state, 0.1,
                                       env.engine.last_propulsion[name])
        ledger = vessel.plant.diagnostics(state, external)
        ledger.assert_balanced()
        assert torch.equal(ledger.terms["propulsion"], env.engine.last_propulsion[name])
        assert torch.equal(ledger.terms["current"], external["current"])
        assert torch.equal(ledger.terms["wind"], external["wind"])
        assert torch.equal(ledger.terms["wave"], external["wave"])
        assert torch.isfinite(ledger.total).all()


@pytest.mark.parametrize("pair", (("a", "b"), ("a", "c"), ("b", "c")))
def test_unequal_mass_pair_contact_attribution(pair):
    spawn = dict(SPAWNS)
    spawn[pair[0]] = (0., 0., 0.)
    spawn[pair[1]] = (0.3, 0., 0.)
    remaining = (set(NAMES) - set(pair)).pop()
    spawn[remaining] = (400., 0., 0.)
    env = fleet(spawn=spawn, sensor=False)
    env.reset()
    _, _, _, _, infos = env.step({name: command(0) for name in NAMES})
    assert infos[pair[0]]["contact_events"]
    assert infos[pair[1]]["contact_events"]
    assert not infos[remaining]["contact_events"]
    assert state_vector(env, remaining)[0] == spawn[remaining][0]


def test_unequal_mass_contact_conserves_linear_momentum():
    spawn = {"a": (0., 0., 0.), "b": (0.3, 0., 0.), "c": (400., 0., 0.)}
    env = fleet(spawn=spawn, sensor=False)
    env.engine.template = ScenarioTemplate.model_validate({"id": "impact", "version": "1",
        "spawn_overrides": {"a": {"u_mps": {"kind": "fixed", "value": 1.}}}})
    env.reset(seed=4)
    before = sum(MASSES[name] * env.engine.states[name].nu_body[0].item() for name in NAMES)
    _, _, _, _, infos = env.step({name: command(0) for name in NAMES})
    after = sum(MASSES[name] * env.engine.states[name].nu_body[0].item() for name in NAMES)
    event = infos["a"]["contact_events"][0]
    assert event.contact_id in {x.contact_id for x in infos["b"]["contact_events"]}
    assert event.normal_impulse_ns > 0
    assert torch.allclose(event.body_a_impulse_frd[:3], -event.body_b_impulse_frd[:3], atol=1e-12)
    assert after == pytest.approx(before, abs=1e-10)
    assert not infos["c"]["contact_events"]


@pytest.mark.parametrize("pair,speed_a,speed_b,offset_y", (
    (("a", "b"), 1., -1., 0.),
    (("a", "c"), 1., 0., 0.),
    (("b", "c"), 2., -0.5, 0.),
    (("a", "b"), 1., 0., 0.1),
    (("a", "c"), 1., -0.5, 0.15),
))
def test_pairwise_contact_action_reaction_and_geometry(pair, speed_a, speed_b, offset_y):
    env = fleet(sensor=False)
    vessel_a, vessel_b = (env.engine.vessels[name] for name in pair)
    def body(vessel, position, speed):
        state = VesselState(torch.tensor(position, dtype=torch.float64),
            torch.tensor((1., 0., 0., 0.), dtype=torch.float64),
            torch.tensor((speed, 0., 0., 0., 0., 0.), dtype=torch.float64))
        return CollisionBody(0, f"vessel:{vessel.instance_id}", vessel.collision_shape,
                             state, vessel.plant)
    a = body(vessel_a, (0., 0., 0.), speed_a)
    b = body(vessel_b, (0.3, offset_y, 0.), speed_b)
    before = MASSES[pair[0]] * speed_a + MASSES[pair[1]] * speed_b
    result = resolve_contacts((a, b), material=ContactMaterial(), dt_s=0.1)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.normal_impulse_ns > 0
    assert torch.equal(event.body_a_impulse_frd[:3], -event.body_b_impulse_frd[:3])
    assert torch.isfinite(event.point_ned_m).all()
    assert torch.linalg.vector_norm(event.normal_a_to_b_ned) == pytest.approx(1.)
    after = (MASSES[pair[0]] * result.states[(0, a.id)].nu_body[0].item() +
             MASSES[pair[1]] * result.states[(0, b.id)].nu_body[0].item())
    assert after == pytest.approx(before, abs=1e-10)
    if offset_y:
        assert abs(result.states[(0, a.id)].nu_body[5].item()) > 0


@pytest.mark.parametrize("target,offset_y,speed", (
    ("a", 0., 1.), ("b", 0.1, 0.5), ("c", -0.1, 2.),
))
def test_common_obstacle_contact_is_owned_by_correct_vessel(target, offset_y, speed):
    spawn = dict(SPAWNS)
    spawn[target] = (0., 0., 0.)
    for name in NAMES:
        if name != target and spawn[name] == (0., 0., 0.):
            spawn[name] = (600., 0., 0.)
    env = fleet(spawn=spawn, sensor=False, static_entities=({
        "id": "rock", "position_ned_m": (0.3, offset_y, 0.),
        "shape": {"kind": "sphere", "radius_m": 0.2}},))
    env.engine.template = ScenarioTemplate.model_validate({"id": "obstacle_start", "version": "1",
        "spawn_overrides": {target: {"u_mps": {"kind": "fixed", "value": speed}}}})
    env.reset(seed=6)
    _, _, _, _, infos = env.step({name: command(0) for name in NAMES})
    assert len(infos[target]["contact_events"]) == 1
    event = infos[target]["contact_events"][0]
    assert "rock" in event.contact_id and f"vessel:{target}" in event.contact_id
    assert event.normal_impulse_ns > 0
    assert torch.linalg.vector_norm(event.normal_a_to_b_ned) == pytest.approx(1.)
    assert all(not infos[name]["contact_events"] for name in NAMES if name != target)
    if offset_y:
        assert abs(env.engine.states[target].nu_body[5].item()) > 0


def test_sustained_obstacle_pushing_keeps_contact_local():
    env = fleet(sensor=False, static_entities=({"id": "rock",
        "position_ned_m": (0.3, 0., 0.),
        "shape": {"kind": "sphere", "radius_m": 0.2}},))
    env.reset(seed=8)
    for _ in range(10):
        _, _, _, _, infos = env.step({"a": command(10), "b": command(0),
                                      "c": command(0)})
        assert len(infos["a"]["contact_events"]) == 1
        assert not infos["b"]["contact_events"]
        assert not infos["c"]["contact_events"]
        assert infos["a"]["contact_events"][0].normal_impulse_ns > 0


def test_geometry_specific_grounding():
    spawn = dict(SPAWNS)
    spawn["a"] = (0., 0., 0.)
    spawn["b"] = (20., 0., -0.5)
    spawn["c"] = (40., 0., -0.5)
    env = fleet(spawn=spawn, bottom=0.1, seabed_collision=True, sensor=False)
    env.reset()
    _, _, _, _, infos = env.step({name: command(0) for name in NAMES})
    assert infos["a"]["grounding"]["contact_active"]
    assert not infos["b"]["grounding"]["contact_active"]
    assert not infos["c"]["grounding"]["contact_active"]
    assert infos["a"]["grounding"]["under_keel_clearance_m"] < \
           infos["b"]["grounding"]["under_keel_clearance_m"]


def test_different_hull_geometry_has_independent_clearance():
    spawn = {"a": (0., 0., 0.), "b": (20., 0., 0.), "c": (40., 0., 0.)}
    shapes = {"a": Sphere(0.2), "b": Box((0.3, 0.2, 0.5)), "c": Sphere(0.1)}
    env = fleet(spawn=spawn, bottom=0.3, seabed_collision=True,
                collision_shape_by_name=shapes, sensor=False)
    env.reset()
    _, _, _, _, infos = env.step({name: command(0) for name in NAMES})
    assert not infos["a"]["grounding"]["contact_active"]
    assert infos["b"]["grounding"]["contact_active"]
    assert not infos["c"]["grounding"]["contact_active"]
    assert infos["a"]["grounding"]["under_keel_clearance_m"] == pytest.approx(0.1)
    assert infos["c"]["grounding"]["under_keel_clearance_m"] == pytest.approx(0.2)


@pytest.mark.parametrize("heading", (0., math.pi / 4))
def test_sustained_grounded_thrust_has_no_false_neighbor_contact(heading):
    spawn = {"a": (0., 0., 0.), "b": (20., 0., -0.5), "c": (40., 0., -0.5)}
    environment = {"current": {"kind": "uniform", "ned_mps": [0, 0, 0.2]},
                   "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                   "waves": {"kind": "calm"}, "visibility_m": 1000}
    env = fleet(spawn=spawn, bottom=0.1, seabed_collision=True,
        heading_by_name={"a": heading}, sensor=False,
        environment=environment,
        loads={name: LinearEnvironmentLoads(10, 0, 0, 0) for name in NAMES})
    env.reset()
    for _ in range(10):
        _, _, _, _, infos = env.step({"a": command(10),
                                      "b": command(0), "c": command(0)})
        assert infos["a"]["grounding"]["contact_active"]
        assert not infos["b"]["grounding"]["contact_active"]
        assert not infos["c"]["grounding"]["contact_active"]
        assert all("world:seabed" in event.contact_id
                   for event in infos["a"]["contact_events"])


def test_wake_exposure_is_owner_specific():
    spawn = {"a": (0., 0., 0.), "b": (-2., 0., 0.), "c": (100., 0., 0.)}
    loads = {name: LinearEnvironmentLoads(0, 0, 0, 1.) for name in NAMES}
    exposed = fleet(spawn=spawn, loads=loads, sensor=False)
    baseline = fleet(spawn=spawn, loads=loads, sensor=False)
    parameters = WakeParameters(0.2, 2., 10., "stage5b", "1")
    original = exposed.engine.vessels["a"]
    exposed.engine.vessels["a"] = replace(original,
        wake_emitter=GaussianWakeEmitter(0, original.vessel_id, parameters))
    for env in (exposed, baseline):
        env.engine.template = ScenarioTemplate.model_validate({"id": "wake_start", "version": "1",
            "spawn_overrides": {"a": {"u_mps": {"kind": "fixed", "value": 1.}}}})
        env.reset(seed=23)
    for _ in range(2):
        exposed.step({name: command(0) for name in NAMES})
        baseline.step({name: command(0) for name in NAMES})
    assert not torch.equal(state_vector(exposed, "b"), state_vector(baseline, "b"))
    assert torch.equal(state_vector(exposed, "c"), state_vector(baseline, "c"))
    assert torch.equal(state_vector(exposed, "a"), state_vector(baseline, "a"))


def test_mixed_sensor_suites_rates_and_observation_ownership():
    base = fleet(sensor=False, static_entities=({"id": "target",
        "position_ned_m": (5., 0., 0.),
        "shape": {"kind": "sphere", "radius_m": 0.5}},))
    def config(name, sensor_type, rate, latency=0):
        return SensorConfig(name, sensor_type, 0, vessel_id, (0, 0, 0),
            (1, 0, 0, 0), rate, latency, 0., 9, "1", "stage5b")
    vessels = []
    contracts = {}
    for vessel_id, name in enumerate(NAMES, 1):
        sensors = []
        if name in ("a", "b"):
            sensors.append(GPS(config("gps", "gps", 10 if name == "a" else 5),
                origin_wgs84_rad_m=(0, 0, 0)))
        sensors.append(IMU(config("imu", "imu", 5 if name == "c" else 10)))
        if name == "b":
            sensors.append(LiDAR(config("lidar", "lidar", 2, 1),
                min_range_m=0, max_range_m=10, fov_rad=0, ray_count=1))
        if name == "c":
            sensors.append(Sonar(config("sonar", "sonar", 2, 0),
                min_range_m=0, max_range_m=150, fov_rad=0, beam_count=1))
        vessels.append(replace(base.engine.vessels[name], sensors=tuple(sensors)))
        def field(label, sid, stype, key, shape, dtype, units, frame):
            return ObservationField(label, sid, stype, "1", "stage5b", "physical",
                key, shape, dtype, units, frame)
        fields = [field("angular", "imu", "imu", "angular_rate_mount_radps",
                        (3,), "float64", "rad/s", "mount")]
        if name in ("a", "b"):
            fields.append(field("velocity", "gps", "gps", "velocity_ned_mps",
                                (3,), "float64", "m/s", "NED"))
        if name == "b":
            fields.append(field("lidar_range", "lidar", "lidar", "range_m",
                                (1,), "float64", "m", "mount"))
        if name == "c":
            fields.append(field("sonar_range", "sonar", "sonar", "range_m",
                                (1,), "float64", "m", "mount"))
        contracts[name] = ObservationContract(tuple(fields))
    engine = EpisodeEngine(base.engine.resolved, tuple(vessels), observation_contracts=contracts)
    initial = engine.reset(seed=19)
    assert "b" in initial.pending_observations and "c" not in initial.pending_observations
    for step in range(1, 7):
        frame = engine.step({name: command(0) for name in NAMES})
        for packet in frame.delivered_packets:
            assert packet.owner_vessel_id in (1, 2, 3)
            assert packet.sensor_id in {sensor.config.instance_id
                                        for sensor in vessels[packet.owner_vessel_id - 1].sensors}
        if step == 6:
            assert set(frame.observations["a"]) == {"angular", "velocity"}
            assert set(frame.observations["b"]) == {"angular", "velocity", "lidar_range"}
            assert set(frame.observations["c"]) == {"angular", "sonar_range"}
            assert engine.latest_packets[(2, "lidar")].sample_step == 5
            assert engine.latest_packets[(3, "sonar")].sample_step == 5
            assert engine.latest_packets[(1, "gps")].sample_step == 6
            assert engine.latest_packets[(2, "gps")].sample_step == 6


def test_sensor_noise_dropout_and_rng_are_independent():
    settings = {"a": ("gps", 10, 0.2, 0., 5),
                "b": ("gps", 5, 0.2, 0., 5),
                "c": ("gps", 10, 0.2, 1., 5)}
    base = fleet(sensor=False, sensor_by_name=settings)
    changed = fleet(sensor=False, sensor_by_name={
        **settings, "a": ("gps", 5, 2., 0.5, 9)})
    first, _ = base.reset(seed=31)
    second, _ = changed.reset(seed=31)
    assert torch.equal(first["b"]["velocity"], second["b"]["velocity"])
    assert first["c"] == {} and second["c"] == {}
    for _ in range(4):
        base.step({name: command(0) for name in NAMES})
        changed.step({name: command(0) for name in NAMES})
        left = base.engine.latest_packets[(2, "gps")]
        right = changed.engine.latest_packets[(2, "gps")]
        assert left.sample_step == right.sample_step
        assert torch.equal(left.values["velocity_ned_mps"],
                           right.values["velocity_ned_mps"])
        assert (3, "gps") not in base.engine.latest_packets
        assert (3, "gps") not in changed.engine.latest_packets


def test_local_observations_hide_other_vessel_state():
    env = fleet()
    observations, _ = env.reset(seed=12)
    assert set(observations["a"]) == {"position"}
    assert set(observations["b"]) == {"velocity"}
    assert set(observations["c"]) == {"velocity"}
    before = observations["b"]["velocity"].clone()
    changed = fleet(spawn={"a": (-10., 0., 0.), "b": SPAWNS["b"],
                           "c": SPAWNS["c"]})
    changed_observations, _ = changed.reset(seed=12)
    assert torch.equal(before, changed_observations["b"]["velocity"])
    assert "position" not in observations["b"]
    assert "position" not in observations["c"]


def test_three_creation_orders_are_physically_equivalent():
    orders = (NAMES, ("c", "b", "a"), ("b", "a", "c"))
    environments = [fleet(order=order, max_steps=12) for order in orders]
    for env in environments:
        env.reset(seed=8)
    for step in range(10):
        actions = {"a": command(5 if step % 2 else -5),
                   "b": command(2), "c": command(-3)}
        for env in environments:
            env.step({name: actions[name] for name in env.engine.vessels})
        for name in NAMES:
            assert all(torch.equal(state_vector(environments[0], name),
                                   state_vector(env, name)) for env in environments[1:])


def test_long_heterogeneous_rollout_replays_exactly():
    names = ("a", "b", "c", "d")
    environment = {"current": {"kind": "uniform", "ned_mps": [0.01, 0, 0]},
                   "wind": {"kind": "uniform", "ned_mps": [0.01, 0, 0]},
                   "waves": {"kind": "irregular", "spectrum": "jonswap",
                       "significant_height_m": 0.01, "peak_period_s": 5.,
                       "direction_rad": 0., "component_count": 8, "seed": 7},
                   "visibility_m": 1000}
    env = fleet(names=names, max_steps=20000, sensor=True,
        spawn={"a": (0., 0., 0.), "b": (200., 0., 0.),
               "c": (400., 0., 0.), "d": (600., 0., 0.)},
        environment=environment,
        loads={name: LinearEnvironmentLoads(0.01, 0.01, 0.01, 0.) for name in names},
        collision_shape_by_name={"a": Sphere(0.2), "b": Box((0.3, 0.2, 0.2)),
                                 "c": Sphere(0.1), "d": Sphere(0.3)},
        static_entities=({"id": "marker", "position_ned_m": (2., 0., 0.),
            "shape": {"kind": "sphere", "radius_m": 0.2}},),
        bottom=2.)
    digests = []
    for _ in range(2):
        env.reset(seed=29)
        total = torch.zeros(13 * 4, dtype=torch.float64)
        for step in range(20000):
            actions = {name: command((1, 0, -1, 0)[i] if step % 2 == 0 else 0)
                       for i, name in enumerate(names)}
            observations, rewards, terms, truncs, infos = env.step(actions)
            assert set(observations) == set(names)
            assert set(rewards) == set(names)
            assert all(math.isfinite(value) for value in rewards.values())
            assert all(torch.isfinite(state_vector(env, name)).all() for name in names)
            total += torch.cat([state_vector(env, name) for name in names])
            if step < 19999:
                assert not any(terms.values()) and not any(truncs.values())
            else:
                assert all(truncs.values())
        digests.append(total)
    assert torch.equal(digests[0], digests[1])


STAGE5B_CASES = (
    "test_solo_multi_invariance_state_actuator_and_sensor_timing",
    "test_parameter_and_actuator_ownership",
    "test_thrust_limits_rate_and_reversal_remain_per_vessel",
    "test_six_axis_parameter_ownership",
    "test_environmental_coexistence_and_order",
    "test_environment_forcing_matrix_has_owned_wrenches",
    "test_per_vessel_wrench_ledgers_balance_under_combined_forcing",
    "test_unequal_mass_pair_contact_attribution",
    "test_unequal_mass_contact_conserves_linear_momentum",
    "test_pairwise_contact_action_reaction_and_geometry",
    "test_common_obstacle_contact_is_owned_by_correct_vessel",
    "test_sustained_obstacle_pushing_keeps_contact_local",
    "test_geometry_specific_grounding",
    "test_different_hull_geometry_has_independent_clearance",
    "test_sustained_grounded_thrust_has_no_false_neighbor_contact",
    "test_wake_exposure_is_owner_specific",
    "test_mixed_sensor_suites_rates_and_observation_ownership",
    "test_sensor_noise_dropout_and_rng_are_independent",
    "test_local_observations_hide_other_vessel_state",
    "test_three_creation_orders_are_physically_equivalent",
    "test_long_heterogeneous_rollout_replays_exactly",
)


def test_registry_complete():
    assert set(STAGE5B_CASES) == {name for name in globals()
                                   if name.startswith("test_") and name != "test_registry_complete"}
