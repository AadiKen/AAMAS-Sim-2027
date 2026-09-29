import math

import pytest

from bcod_sim.config.models import ExperimentConfig
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.actuators.physical import ActuatorPipeline, ActuatorSet, VesselMotion
from bcod_sim.collision.shapes import Sphere
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.core.lifecycle import PhysicalAction, common_differential_to_thruster_commands
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.web.runtime_factory import build_engine
import torch


def test_logged_common_differential_maps_to_normalized_saturated_commands():
    action = common_differential_to_thruster_commands(40, 10)
    assert action == PhysicalAction((("port", (0.5,)), ("starboard", (0.3,))))
    reverse = common_differential_to_thruster_commands(0, 25)
    assert reverse.commands == (("port", (0.25,)), ("starboard", (-0.25,)))
    assert common_differential_to_thruster_commands(90, 30).commands == (
        ("port", (1.,)), ("starboard", (0.6,)))
    mirrored = common_differential_to_thruster_commands(40, -10)
    assert mirrored.commands == (("port", (0.3,)), ("starboard", (0.5,)))


@pytest.mark.parametrize("thrust,difference,limit", [
    (math.nan, 0, 100), (0, math.inf, 100), (0, 0, 0),
])
def test_command_mapping_rejects_invalid_scalars(thrust, difference, limit):
    with pytest.raises(PhysicalValidationError):
        common_differential_to_thruster_commands(thrust, difference,
                                                 command_limit_percent=limit)


def _config(*, backend=None, mode="direct_actuator", actuator_system=None):
    return {
        "schema_version": 1,
        "experiment": {"id": "physical-v1-contract", "seed": 1},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": .1,
                       "dynamics_substeps": 1, "policy_every_n_master_steps": 1},
        "world": {"source": {"kind": "parametric"}, "environment": {
            "current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "waves": {"kind": "calm"}, "visibility_m": 1000},
            "bathymetry": {"kind": "flat", "bottom_ned_z_m": 20,
                           "vertical_datum": "MSL"}},
        "vessels": [{"instance_id": "surveyor", "definition": "vessel@1",
                     "controller": {"mode": mode}, "spawn": {"ned_m": [0, 0, 0],
                     "rpy_rad": [0, 0, 0]}, **({"actuator_backend": backend} if backend else {}),
                     **({"actuator_system": actuator_system} if actuator_system else {})}],
        "task": {"type": "waypoint@1", "reward": {"individual_weight": 1,
                 "team_weight": 0}, "disabled_agent_behavior": "deactivate_keep_physical"},
    }


def test_physical_v1_requires_direct_mode_and_actuator_system():
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(_config(backend="physical_v1"))
    system = {"schema_version": "manta-actuator-v1", "actuators": []}
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(_config(backend="physical_v1", mode="high_level",
                                                actuator_system=system))
    # Configuration validation establishes the requested integration switch.
    assert ExperimentConfig.model_validate(_config(backend="physical_v1",
                                                     actuator_system=system)).vessels[0].actuator_backend == "physical_v1"


def _physical_set(*, exponent=1., tau=.1):
    def device(name, y):
        return {"id": name, "type": "fixed_thruster", "pose": {
            "position_frd": [0., y, 0.], "direction_frd": [1., 0., 0.]},
            "propulsion": {"model": "power_law", "max_forward_n": 10.,
                           "max_reverse_n": 10., "exponent": exponent,
                           "command_bounds": [-1., 1.]},
            "dynamics": {"thrust_tau_s": tau}}
    return ActuatorSet([device("port", -.5), device("starboard", .5)])


def test_actuator_law_lag_mirroring_saturation_and_yaml_roundtrip(tmp_path):
    motion = VesselMotion()
    for exponent, expected in ((1., 5.), (2., 2.5)):
        pair = _physical_set(exponent=exponent, tau=.1)
        step = pair.predict_wrench({"port": (.5,), "starboard": (.5,)}, motion,
                                   dt=.1, commit=True)
        target = expected
        lagged = target * (1. - math.exp(-1.))
        assert [d.thrust_n for d in step.devices] == pytest.approx([lagged, lagged])
        assert step.wrench_frd[0] == pytest.approx(2 * lagged)
        assert step.wrench_frd[5] == pytest.approx(0., abs=1e-12)
    pair = _physical_set()
    left = pair.predict_wrench({"port": (1.,), "starboard": (0.,)}, motion)
    right = pair.predict_wrench({"port": (0.,), "starboard": (1.,)}, motion)
    assert left.wrench_frd[5] > 0 and right.wrench_frd[5] < 0
    assert left.wrench_frd[5] == pytest.approx(-right.wrench_frd[5])
    saturated = pair.predict_wrench({"port": (2.,), "starboard": (0.,)}, motion)
    assert saturated.devices[0].saturated and saturated.devices[0].target_command == (1.,)
    import yaml
    package = {"actuator_system": {"schema_version": "manta-actuator-v1",
        "actuators": [_physical_set().devices[0].spec, _physical_set().devices[1].spec]}}
    path = tmp_path / "actuator.yaml"
    path.write_text(yaml.safe_dump(package))
    restored = ActuatorSet.from_yaml(path)
    assert restored.predict_wrench({"port": (.5,), "starboard": (.5,)}, motion).wrench_frd == \
        pytest.approx(_physical_set().predict_wrench(
            {"port": (.5,), "starboard": (.5,)}, motion).wrench_frd)


def test_direct_physical_action_wrench_is_delivered_to_plant6():
    v = lambda values: torch.tensor(values, dtype=torch.float64)
    plant = Plant6(MassProperties(10., v((0, 0, 0)), torch.diag(v((4, 5, 6))),
                                  torch.zeros((6, 6), dtype=torch.float64)),
                   Damping(v((0,) * 6), v((0,) * 6)),
                   Hydrostatics(10 * 9.80665, v((0, 0, 0))),
                   OperatingEnvelope(v((1e4,) * 6), max_substep_s=.2))
    config = _config(backend="physical_v1", actuator_system={"schema_version": "manta-actuator-v1",
        "actuators": [{"id": "port"}, {"id": "starboard"}]})
    registry = Registry()
    registry.register("vessel", "vessel", "1", {}, "test")
    registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": "surveyor",
        "target_ned_m": [100, 0, 0], "radius_m": .1}, "test")
    engine = EpisodeEngine(resolve(config, registry), (EpisodeVessel("surveyor", 1, plant,
        Sphere(.2), (), (), ExplicitZeroLoads(), physical_pipeline=ActuatorPipeline(_physical_set())),))
    engine.reset()
    action = common_differential_to_thruster_commands(50, 0)
    frame = engine.step({"surveyor": action})
    assert engine.last_propulsion["surveyor"].tolist() == pytest.approx(
        engine.last_actuator_diagnostics["surveyor"]["achieved_wrench"])
    assert engine.last_propulsion["surveyor"][0].item() > 0
    assert frame.states["surveyor"].nu_body[0].item() > 0


def test_physical_v1_runtime_factory_routes_direct_commands_to_plant6():
    registry = Registry()
    registry.register("vessel", "vessel", "1", {
        "mass_kg": 10., "cg_frd_m": [0, 0, 0],
        "inertia_cg_kg_m2": [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
        "added_mass_kg": [[0] * 6 for _ in range(6)],
        "linear_damping": [0] * 6, "quadratic_damping": [0] * 6,
        "buoyancy_n": 98.0665, "center_buoyancy_frd_m": [0, 0, 0],
        "max_abs_nu": [1e4] * 6, "max_substep_s": .2,
        "collision": {"kind": "sphere", "radius_m": .2},
        "environment_loads": [0] * 4,
    }, "test")
    registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": "surveyor",
        "target_ned_m": [100, 0, 0], "radius_m": .1}, "test")
    system = {"schema_version": "manta-actuator-v1", "actuators": [
        {"id": "port", "type": "fixed_thruster", "pose": {"position_frd": [0, -.5, 0],
         "direction_frd": [1, 0, 0]}, "propulsion": {"model": "power_law",
         "max_forward_n": 10., "max_reverse_n": 10., "exponent": 1.,
         "command_bounds": [-1, 1]}, "dynamics": {"thrust_tau_s": .1}},
        {"id": "starboard", "type": "fixed_thruster", "pose": {"position_frd": [0, .5, 0],
         "direction_frd": [1, 0, 0]}, "propulsion": {"model": "power_law",
         "max_forward_n": 10., "max_reverse_n": 10., "exponent": 1.,
         "command_bounds": [-1, 1]}, "dynamics": {"thrust_tau_s": .1}},
    ]}
    engine = build_engine(resolve(_config(backend="physical_v1", actuator_system=system), registry))
    engine.reset()
    frame = engine.step({"surveyor": common_differential_to_thruster_commands(50, 0)})
    # Each thruster target is 5 N and the .1 s lag step reaches (1-exp(-1))*5 N.
    expected_force = 2 * 5 * (1 - math.exp(-1))
    assert engine.last_propulsion["surveyor"].tolist() == pytest.approx(
        engine.last_actuator_diagnostics["surveyor"]["achieved_wrench"])
    assert engine.last_propulsion["surveyor"][0].item() == pytest.approx(expected_force)
    assert frame.states["surveyor"].nu_body[0].item() > 0
