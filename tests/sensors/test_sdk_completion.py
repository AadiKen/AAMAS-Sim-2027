import math

import pytest
import torch

from bcod_sim.config.models import World
from bcod_sim.config.registry import Definition
from bcod_sim.config.hashing import content_hash
from bcod_sim.core.errors import InvalidMediumError, MissingCapabilityError
from bcod_sim.sensors.base import SensorConfig, SensorContext, packet
from bcod_sim.sensors.kinematics import mount_kinematics
from bcod_sim.sensors.sonar import Sonar
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.web.runtime_factory import _sensor
from bcod_sim.sensors.registry import register_sensor_type
from bcod_sim.world.bathymetry import RasterBathymetry
from bcod_sim.world.environment import CapabilityDescriptor, Environment, Medium
from bcod_sim.world.world import ParametricWorld
from bcod_sim.logging.recorder import _encode_sensor_values


DTYPE = torch.float64


def tensor(value): return torch.tensor(value, dtype=DTYPE)


def make_world(*, bathymetry=None, waves=None, current=(1, 2, 0)):
    raw = {"source": {"kind": "parametric"}, "environment": {
        "current": {"kind": "uniform", "ned_mps": current},
        "wind": {"kind": "uniform", "ned_mps": [3, 0, 0]},
        "waves": waves or {"kind": "calm"}, "visibility_m": 100}}
    if bathymetry is not None: raw["bathymetry"] = bathymetry
    return ParametricWorld(World.model_validate(raw), 0)


def context(world, position=(0, 0, 1)):
    return SensorContext(VesselState(tensor(position), tensor((1, 0, 0, 0)), tensor((0,)*6)),
                         world, 0, tensor((0, 0, 0)), tensor((0, 0, 0)))


class CurrentMagnitudeSensor:
    kind = "physical"
    requirements = frozenset({"water.current"})

    def __init__(self, config, *, definition, parameters):
        self.config = config
        self.scale = parameters.get("scale", 1.0)

    def sample(self, ctx, *, sample_step):
        mount = mount_kinematics(self.config, ctx)
        ctx.environment.require(self.requirements)
        current = ctx.environment.sample("water.current", mount.position_ned_m, ctx.sim_time_s).value[0]
        value = float(torch.linalg.vector_norm(current))*self.scale
        return packet(self.config, self.kind, sample_step, {"speed": value},
                      {"speed": "m/s"}, {"speed": "scalar"}, sample_time_s=ctx.sim_time_s)


class WaveCurrentSensor:
    kind = "physical"
    requirements = frozenset({"water.current", "waves.surface"})

    def __init__(self, config): self.config = config

    def sample(self, ctx, *, sample_step):
        mount = mount_kinematics(self.config, ctx); ctx.environment.require(self.requirements)
        current = ctx.environment.sample("water.current", mount.position_ned_m, ctx.sim_time_s).value[0]
        surface = ctx.environment.sample("waves.surface", mount.position_ned_m, ctx.sim_time_s).value[0]
        return packet(self.config, self.kind, sample_step, {"record": {"current": current, "surface": surface}},
                      {"record": "structured"}, {"record": "NED"})


def base_config(sensor_type="custom", name="sensor"):
    return SensorConfig(name, sensor_type, 0, 1, (0, 0, 0), (1, 0, 0, 0),
                        10, 0, 0, 7, "1", "external-test")


def test_environment_capabilities_metadata_medium_and_depth_semantics():
    world = make_world(bathymetry={"kind": "flat", "bottom_ned_z_m": 10, "vertical_datum": "MSL"},
        waves={"kind": "regular", "height_m": 2, "period_s": 4, "direction_rad": 0})
    environment = Environment(world)
    assert environment.has("water.current")
    descriptor = environment.describe("water.current")
    assert descriptor.units == "m/s" and descriptor.frame == "NED" and descriptor.valid_media == {Medium.WATER}
    with pytest.raises(InvalidMediumError):
        environment.sample("water.current", tensor((0, 0, -2)), 0)
    invalid = environment.sample("atmosphere.wind", tensor((0, 0, 1)), 0, invalid="status")
    assert not invalid.valid and invalid.medium == Medium.WATER
    # At t=0 the NED surface z is -1, therefore water depth is 10 - (-1) = 11 m.
    assert environment.sample("bathymetry.depth", tensor((0, 0, 0)), 0).value.item() == pytest.approx(11)
    with pytest.raises(MissingCapabilityError): environment.require({"acoustic.propagation"})


def test_future_domain_provider_registers_without_worldsample_change():
    world = make_world()
    descriptor = CapabilityDescriptor("optical.attenuation", "1/m", "scalar",
        frozenset({Medium.WATER}), "point scalar field", "static", "test provider")
    world.register_capability("optical.attenuation",
        lambda points, time_s, **kwargs: points.new_full((len(points),), .2), descriptor)
    environment = Environment(world)
    assert environment.sample("optical.attenuation", tensor((0, 0, 1)), 0).value.item() == pytest.approx(.2)


def test_external_sensor_registered_and_constructed_entirely_from_definition():
    sensor_type = "test_current_magnitude"
    register_sensor_type(sensor_type, CurrentMagnitudeSensor)
    payload = {"kind": sensor_type, "rate_hz": 10, "parameters": {"scale": 2.0}}
    definition = Definition("sensor", "speed", "1", payload, "external-package", content_hash(payload))
    sensor = _sensor(definition, 1)
    result = sensor.sample(context(make_world()), sample_step=0)
    assert result.values["speed"] == pytest.approx(2*math.sqrt(5))
    assert result.schema.fields[0].dtype == "float"
    assert sensor.config.config_fingerprint == content_hash({"id": "speed", "version": "1",
        "source": "external-package", "payload": payload, "env_id": 0, "owner_vessel_id": 1})


def test_multi_domain_custom_sensor_and_multiple_instances():
    world = make_world(bathymetry={"kind": "flat", "bottom_ned_z_m": 10, "vertical_datum": "MSL"})
    first, second = WaveCurrentSensor(base_config(name="a")), WaveCurrentSensor(base_config(name="b"))
    a = first.sample(context(world), sample_step=0)
    b = second.sample(context(world), sample_step=0)
    assert a.values["record"]["current"].tolist() == [1, 2, 0]
    assert a.sensor_id != b.sensor_id and a.schema.fields[0].dtype == "record"


@pytest.mark.parametrize("bathymetry", [
    {"kind": "flat", "bottom_ned_z_m": 10, "vertical_datum": "MSL"},
    {"kind": "plane", "origin_ned_m": [0, 0, 0], "bottom_at_origin_ned_z_m": 10,
     "north_slope": .1, "east_slope": 0, "vertical_datum": "MSL"},
])
def test_sonar_uses_canonical_flat_and_slope_surfaces(bathymetry):
    world = make_world(bathymetry=bathymetry)
    sonar = Sonar(base_config("sonar"), min_range_m=0, max_range_m=30, fov_rad=0, beam_count=1)
    assert sonar.sample(context(world), sample_step=0).values["range_m"].item() == pytest.approx(9)


def test_sonar_uses_raster_without_sensor_changes():
    world = make_world(bathymetry={"kind": "flat", "bottom_ned_z_m": 10, "vertical_datum": "MSL"})
    world.bathymetry = RasterBathymetry(tensor(((10, 10), (10, 10))), (-5, -5), (10, 10), "MSL")
    sonar = Sonar(base_config("sonar"), min_range_m=0, max_range_m=30, fov_rad=0, beam_count=1)
    assert sonar.sample(context(world), sample_step=0).values["range_m"].item() == pytest.approx(9)


def test_oriented_entity_changes_raycast():
    raw = {"source": {"kind": "parametric"}, "environment": {
        "current": {"kind": "uniform", "ned_mps": [0, 0, 0]}, "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
        "waves": {"kind": "calm"}, "visibility_m": 100}, "static_entities": [{"id": "box",
        "position_ned_m": [5, 0, 0], "shape": {"kind": "box", "half_extents_m": [2, .5, .5]},
        "orientation_q_to_ned": [math.sqrt(.5), 0, 0, math.sqrt(.5)]}]}
    world = ParametricWorld(World.model_validate(raw), 0)
    hit = Environment(world).sample("geometry.raycast", tensor((0, 0, 0)), 0,
        direction_ned=tensor((1, 0, 0)), include_bathymetry=False).value
    assert hit["distance_m"] == pytest.approx(4.5)


def test_large_array_uses_binary_artifact_and_structured_output_stays_structured(tmp_path):
    values = {"image": torch.arange(5000, dtype=torch.float32).reshape(50, 100),
              "detections": [{"id": "target", "range_m": 3.0}]}
    encoded = _encode_sensor_values(tmp_path, "camera", 4, values)
    artifact = tmp_path/encoded["image"]["array_artifact"]
    assert artifact.suffix == ".npy" and artifact.is_file()
    assert encoded["image"]["shape"] == [50, 100]
    assert encoded["detections"] == [{"id": "target", "range_m": 3.0}]
