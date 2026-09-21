import copy
import math

import pytest
import torch

from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.errors import ConfigSchemaError, ExternalDataCoverageError, ExternalDataUnavailableError, PhysicalValidationError
from bcod_sim.world.world import ParametricWorld


def experiment():
    return {
        "schema_version": 1,
        "experiment": {"id": "world-test", "seed": 1},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.02, "dynamics_substeps": 1,
                       "policy_every_n_master_steps": 1},
        "world": {
            "source": {"kind": "parametric"},
            "environment": {
                "current": {"kind": "uniform", "ned_mps": [1, 2, 3]},
                "wind": {"kind": "uniform", "ned_mps": [4, 5, 6]},
                "waves": {"kind": "calm"},
                "visibility_m": 5000,
                "rain_rate_mps": 0.001,
                "fog_extinction_per_m": 0.002,
            },
            "bathymetry": {"kind": "flat", "bottom_ned_z_m": 12, "vertical_datum": "MSL"},
            "boundary": {"id": "bounds", "min_ned_m": [-100, -100, -20], "max_ned_m": [100, 100, 50]},
            "spawn_regions": [{"id": "center", "min_ned_m": [-10, -10, -2], "max_ned_m": [10, 10, 2]}],
            "static_entities": [{"id": "buoy", "position_ned_m": [2, 3, 0],
                                  "shape": {"kind": "sphere", "radius_m": 1}}],
            "scripted_entities": [{"id": "traffic", "position_ned_m": [0, 0, 0],
                                    "shape": {"kind": "box", "half_extents_m": [1, 2, 3]},
                                    "velocity_ned_mps": [1, -2, 0], "start_time_s": 5}],
        },
        "vessels": [{"instance_id": "test", "definition": "vessel@1", "controller": {"mode": "direct_actuator"},
                     "spawn": {"ned_m": [0, 0, 0], "rpy_rad": [0, 0, 0]}}],
        "task": {"type": "task@1", "reward": {"individual_weight": 1, "team_weight": 0},
                 "disabled_agent_behavior": "deactivate_keep_physical"},
    }


def world(raw=None, env_id=0):
    registry = Registry()
    registry.register("vessel", "vessel", "1", {"test": True}, "fixture")
    registry.register("task", "task", "1", {"test": True}, "fixture")
    return ParametricWorld.from_resolved(resolve(raw or experiment(), registry), env_id=env_id)


def positions(rows=((0, 0, 0), (10, 20, 5))):
    return torch.tensor(rows, dtype=torch.float64)


def test_uniform_world_samples_ned_si_and_determinism():
    w = world()
    a = w.sample(positions(), sim_time_s=1, env_id=0)
    b = w.sample(positions(), sim_time_s=1, env_id=0)
    assert torch.equal(a.current_ned_mps, b.current_ned_mps)
    assert a.current_ned_mps.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert a.wind_ned_mps.tolist() == [[4, 5, 6], [4, 5, 6]]
    assert a.wake_ned_mps.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert a.wave_surface_ned_z_m.tolist() == [0, 0]
    assert a.wave_orbital_ned_mps.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert a.visibility_m.tolist() == [5000, 5000]
    assert a.rain_rate_mps.tolist() == [0.001, 0.001]
    assert a.fog_extinction_per_m.tolist() == [0.002, 0.002]
    assert a.bottom_ned_z_m.tolist() == [12, 12]
    assert a.bathymetry_vertical_datum == "MSL"


def test_linear_fields_and_batch_order_independence():
    raw = experiment()
    raw["world"]["environment"]["current"] = {
        "kind": "linear", "origin_ned_m": [0, 0, 0], "base_ned_mps": [1, 2, 3],
        "gradient_per_s": [[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]]}
    w = world(raw)
    p = positions()
    sample = w.sample(p, sim_time_s=1, env_id=0)
    reverse = w.sample(p.flip(0), sim_time_s=1, env_id=0)
    assert torch.allclose(sample.current_ned_mps, torch.tensor([[1, 2, 3], [2, 6, 4.5]], dtype=torch.float64))
    assert torch.equal(sample.current_ned_mps, reverse.current_ned_mps.flip(0))


def test_wave_phase_moves_crest_north_and_ned_surface_sign():
    raw = experiment()
    raw["world"]["environment"]["waves"] = {"kind": "regular", "height_m": 2,
                                                   "period_s": 4, "direction_rad": 0, "phase_rad": 0}
    w = world(raw)
    initial = w.sample(positions(((0, 0, 0),)), sim_time_s=0, env_id=0)
    omega = 2 * math.pi / 4
    wavelength = 2 * math.pi * 9.80665 / (omega * omega)
    quarter = w.sample(positions(((wavelength / 4, 0, 0),)), sim_time_s=1, env_id=0)
    assert initial.wave_surface_ned_z_m[0].item() == pytest.approx(-1)
    assert quarter.wave_surface_ned_z_m[0].item() == pytest.approx(-1)
    assert initial.wave_orbital_ned_mps[0, 0].item() > 0


def test_sloped_bathymetry_and_missing_bathymetry_fail_closed():
    raw = experiment()
    raw["world"]["bathymetry"] = {"kind": "plane", "origin_ned_m": [0, 0, 0],
                                   "bottom_at_origin_ned_z_m": 10, "north_slope": 0.1,
                                   "east_slope": -0.2, "vertical_datum": "LAT"}
    w = world(raw)
    assert w.bottom_ned_z_m(positions(((10, 5, 0),)), sim_time_s=0, env_id=0)[0].item() == pytest.approx(10)
    assert w.sample(positions(), sim_time_s=0, env_id=0).bathymetry_vertical_datum == "LAT"
    del raw["world"]["bathymetry"]
    missing = world(raw)
    assert missing.sample(positions(), sim_time_s=0, env_id=0).bottom_ned_z_m is None
    with pytest.raises(ExternalDataUnavailableError):
        missing.bottom_ned_z_m(positions(), sim_time_s=0, env_id=0)


def test_entities_scripted_time_and_sorted_identity():
    w = world()
    before = w.entities(sim_time_s=4, env_id=0)
    after = w.entities(sim_time_s=7, env_id=0)
    assert [e.id for e in before] == ["buoy"]
    assert [e.id for e in after] == ["buoy", "traffic"]
    assert after[1].position_ned_m == (2, -4, 0)
    assert after[1].scripted is True
    assert after == w.entities(sim_time_s=7, env_id=0)


def test_boundary_spawn_and_environment_isolation():
    w = world(env_id=3)
    assert w.spawn_region("center").id == "center"
    with pytest.raises(ExternalDataCoverageError):
        w.sample(positions(((101, 0, 0),)), sim_time_s=0, env_id=3)
    with pytest.raises(PhysicalValidationError):
        w.sample(positions(), sim_time_s=0, env_id=2)
    with pytest.raises(PhysicalValidationError):
        w.sample(positions(), sim_time_s=-1, env_id=3)
    with pytest.raises(ExternalDataCoverageError):
        w.spawn_region("unknown")


def test_invalid_world_config_and_unresolved_assets_rejected():
    raw = experiment()
    raw["world"]["bathymetry"]["vertical_datum"] = ""
    with pytest.raises(ConfigSchemaError):
        world(raw)
    raw = experiment()
    raw["world"]["spawn_regions"][0]["max_ned_m"] = [1000, 10, 2]
    with pytest.raises(PhysicalValidationError):
        world(raw)
    raw = experiment()
    raw["world"]["static_entities"].append(copy.deepcopy(raw["world"]["static_entities"][0]))
    with pytest.raises(ConfigSchemaError):
        world(raw)
    raw = experiment()
    raw["world"]["obstacles"] = ["asset@1"]
    registry = Registry()
    registry.register("vessel", "vessel", "1", {}, "fixture")
    registry.register("task", "task", "1", {}, "fixture")
    registry.register("asset", "asset", "1", {}, "fixture")
    with pytest.raises(PhysicalValidationError):
        ParametricWorld.from_resolved(resolve(raw, registry), env_id=0)


def test_derived_nonfinite_world_values_fail_visibly():
    raw = experiment()
    raw["world"]["environment"]["current"] = {
        "kind": "linear", "origin_ned_m": [0, 0, 0], "base_ned_mps": [0, 0, 0],
        "gradient_per_s": [[1e308, 0, 0], [0, 0, 0], [0, 0, 0]]}
    w = world(raw)
    with pytest.raises(PhysicalValidationError):
        w.sample(positions(((10, 0, 0),)), sim_time_s=0, env_id=0)
