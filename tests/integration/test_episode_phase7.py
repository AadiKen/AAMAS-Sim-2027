import pytest
import torch

from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.collision.shapes import Sphere
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.core.errors import PhysicalValidationError, UnknownReferenceError
from bcod_sim.core.lifecycle import DirectAction, TerminationReason
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.scenario.generator import ScenarioTemplate, generate
from bcod_sim.sensors.base import SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.tasks.primitives import CoverageParams, CoverageTask, FormationParams, FormationTask, TeamWaypointParams, TeamWaypointTask
from bcod_sim.state.vessel_state import VesselState


def raw(*, behavior="deactivate_keep_physical", max_steps=8, policy_every=2):
    return {
        "schema_version": 1,
        "experiment": {"id": "episode-test", "seed": 19},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.1, "dynamics_substeps": 2,
                       "policy_every_n_master_steps": policy_every, "max_master_steps": max_steps},
        "world": {"source": {"kind": "parametric"}, "environment": {
            "current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]}, "waves": {"kind": "calm"},
            "visibility_m": 1000},
            "bathymetry": {"kind": "flat", "bottom_ned_z_m": 50, "vertical_datum": "MSL"}},
        "vessels": [{"instance_id": "agent", "definition": "vessel@1",
                     "controller": {"mode": "direct_actuator"},
                     "spawn": {"ned_m": [0, 0, 0], "rpy_rad": [0, 0, 0]}}],
        "task": {"type": "waypoint@1", "reward": {"individual_weight": 1, "team_weight": 0},
                 "disabled_agent_behavior": behavior},
    }


def resolved(raw_config=None, task_payload=None):
    registry = Registry()
    registry.register("vessel", "vessel", "1", {}, "test")
    registry.register("task", "waypoint", "1", task_payload or {
        "kind": "waypoint", "agent_id": "agent", "target_ned_m": [100, 0, 0],
        "radius_m": 0.01}, "test")
    return resolve(raw_config or raw(), registry)


def plant():
    v = lambda x: torch.tensor(x, dtype=torch.float64)
    mass = MassProperties(10, v((0, 0, 0)), torch.diag(v((4, 5, 6))),
                          torch.zeros((6, 6), dtype=torch.float64))
    return Plant6(mass, Damping(v((0,)*6), v((0,)*6)),
                  Hydrostatics(10*9.80665, v((0, 0, 0))),
                  OperatingEnvelope(v((1e4,)*6), max_substep_s=0.2))


def engine(config=None, task_payload=None, *, with_sensor=False):
    thruster = FixedThruster(ActuatorConfig("prop", 0, 1, (0, 0, 0), (1, 0, 0, 0), Bounds(-100, 100)))
    sensors = (GPS(SensorConfig("gps", "gps", 0, 1, (0, 0, 0), (1, 0, 0, 0),
                                10, 2, 0, 5, "1", "test"), origin_wgs84_rad_m=(0, 0, 0)),) if with_sensor else ()
    vessel = EpisodeVessel("agent", 1, plant(), Sphere(0.2), (thruster,), sensors, ExplicitZeroLoads())
    return EpisodeEngine(resolved(config, task_payload), (vessel,))


def action(thrust=10):
    return DirectAction((("prop", ThrustCommand(thrust)),))


def state(north=0, east=0):
    return VesselState(torch.tensor([north, east, 0.], dtype=torch.float64),
                       torch.tensor([1., 0, 0, 0], dtype=torch.float64),
                       torch.zeros(6, dtype=torch.float64))


def test_episode_order_reward_checkpoint_replay_and_reset_isolation():
    sim = engine()
    first = sim.reset()
    assert first.master_step == 0 and first.reward is None
    frame1 = sim.step({"agent": action()})
    assert frame1.states["agent"].position_ned[0].item() > 0
    assert frame1.reward.individual_components["agent"]["waypoint_progress"] > 0
    checkpoint = sim.checkpoint()
    frame2 = sim.step({})
    sim.restore(checkpoint)
    repeated = sim.step({})
    assert torch.equal(frame2.states["agent"].position_ned, repeated.states["agent"].position_ned)
    assert frame2.reward == repeated.reward
    assert sim.last_propulsion["agent"].tolist() == checkpoint.last_propulsion["agent"].tolist()
    reset = sim.reset()
    assert reset.states["agent"].position_ned.tolist() == [0, 0, 0]
    assert sim.master_step == 0 and not sim.held_commands and not sim.terminated
    again = sim.step({"agent": action()})
    assert torch.equal(frame1.states["agent"].position_ned, again.states["agent"].position_ned)


def test_policy_tick_contract_and_time_limit():
    sim = engine(raw(max_steps=2))
    sim.reset()
    with pytest.raises(PhysicalValidationError):
        sim.step({})
    assert sim.termination_reason == TerminationReason.CONTRACT_FAILURE
    sim.reset()
    sim.step({"agent": action(0)})
    with pytest.raises(PhysicalValidationError):
        sim.step({"agent": action(0)})
    sim.reset()
    sim.step({"agent": action(0)})
    final = sim.step({})
    assert final.terminated and final.termination_reason == TerminationReason.TIME_LIMIT


def test_task_success_and_disabled_agent_behaviors():
    payload = {"kind": "waypoint", "agent_id": "agent", "target_ned_m": [0, 0, 0],
               "radius_m": 1, "success_bonus": 4}
    sim = engine(task_payload=payload)
    sim.reset()
    frame = sim.step({"agent": action(0)})
    assert frame.termination_reason == TerminationReason.TASK_SUCCESS
    assert frame.reward.per_agent_total["agent"] == 4
    for behavior in ("deactivate_keep_physical", "deactivate_remove_physical", "terminate_episode"):
        sim = engine(raw(behavior=behavior))
        sim.reset()
        sim.step({"agent": action()})
        previous = sim.states["agent"].position_ned.clone()
        sim.disable_agent("agent", reason="damaged")
        assert not sim.statuses["agent"].rl_active
        if behavior == "terminate_episode":
            assert sim.termination_reason == TerminationReason.AGENT_DISABLED
        else:
            frame = sim.step({})
            if behavior == "deactivate_remove_physical":
                assert torch.equal(frame.states["agent"].position_ned, previous)
            else:
                assert frame.states["agent"].position_ned[0].item() > previous[0].item()


def test_seeded_scenario_is_resolved_and_reproducible():
    config = resolved()
    template = ScenarioTemplate.model_validate({"id": "spawn", "version": "1", "spawn_overrides": {
        "agent": {"north_m": {"kind": "uniform", "low": -5, "high": 5},
                  "east_m": {"kind": "categorical", "values": [1, 2], "weights": [1, 3]},
                  "down_m": {"kind": "fixed", "value": 3},
                  "yaw_rad": {"kind": "normal_bounded", "mean": 0, "std": 1, "low": -1, "high": 1},
                  "u_mps": {"kind": "sampled_list", "values": [0, 0.5]}}}})
    a = generate(config, template, seed=9)
    b = generate(config, template, seed=9)
    c = generate(config, template, seed=10)
    assert a == b and a.content_hash != c.content_hash
    assert -5 <= a.spawns[0].position_ned_m[0] <= 5
    assert a.spawns[0].position_ned_m[1] in (1, 2)
    assert a.spawns[0].position_ned_m[2] == 3
    assert -1 <= a.spawns[0].rpy_rad[2] <= 1
    assert a.spawns[0].nu_body[0] in (0, 0.5)
    assert a.payload()["max_master_steps"] == 8


def test_pending_sensor_delivery_replays_from_checkpoint():
    sim = engine(with_sensor=True)
    sim.reset()
    sim.step({"agent": action()})
    checkpoint = sim.checkpoint()
    first = sim.step({})
    sim.restore(checkpoint)
    repeated = sim.step({})
    assert len(first.delivered_packets) == 1
    assert first.delivered_packets[0].sample_step == 0
    assert torch.equal(first.delivered_packets[0].values["velocity_ned_mps"],
                       repeated.delivered_packets[0].values["velocity_ned_mps"])


def test_team_formation_and_coverage_component_accounting():
    team = TeamWaypointTask(TeamWaypointParams(kind="waypoint_team", target_ned_m=(1, 0, 0), radius_m=0.1))
    team.reset({"a": state(), "b": state()})
    result = team.evaluate({"a": state(1), "b": state(1)}, ("a", "b"))
    assert result.success and result.team_components["team_progress"] == 1
    formation = FormationTask(FormationParams(kind="formation", leader_id="a",
        offsets_ned_m={"b": (0, 1, 0)}, tolerance_m=0.1))
    formation.reset({"a": state(), "b": state()})
    result = formation.evaluate({"a": state(), "b": state(0, 1)}, ("a", "b"))
    assert result.success and result.individual_components["b"]["formation_progress"] == 1
    coverage = CoverageTask(CoverageParams(kind="coverage", min_north_m=0, max_north_m=2,
        min_east_m=0, max_east_m=2, cell_size_m=1, required_fraction=0.5))
    coverage.reset({"a": state(), "b": state()})
    first = coverage.evaluate({"a": state(), "b": state()}, ("a", "b"))
    second = coverage.evaluate({"a": state(), "b": state(1, 1)}, ("a", "b"))
    assert first.team_components["new_cells"] == 1
    assert second.success and second.team_components["new_cells"] == 1
    coverage.reset({"a": state(), "b": state()})
    assert coverage.evaluate({"a": state(), "b": state()}, ("a", "b")).team_components["new_cells"] == 1


def test_unknown_task_primitive_does_not_fall_back():
    with pytest.raises(UnknownReferenceError):
        engine(task_payload={"kind": "legacy_unknown"})
