import copy
import math

import pytest
import torch

from bcod_sim.config.models import World
from bcod_sim.core.errors import ExternalDataUnavailableError, PhysicalValidationError, UnknownReferenceError
from bcod_sim.frames.geodesy import geodetic_to_ned
from bcod_sim.rl.observation import ObservationContract, ObservationField
from bcod_sim.sensors.abstract import AbstractEntitySensor
from bcod_sim.sensors.base import SensorConfig, SensorContext
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.ground_truth import GroundTruthState
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.kinematics import mount_kinematics
from bcod_sim.sensors.lidar import LiDAR
from bcod_sim.sensors.registry import SensorRegistry
from bcod_sim.sensors.scheduler import SensorScheduler
from bcod_sim.sensors.sonar import Sonar
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.world.world import ParametricWorld


DTYPE = torch.float64


def vector(x):
    return torch.tensor(x, dtype=DTYPE)


def sensor_config(sensor_type, *, name=None, mount=(0, 0, 0), q=(1, 0, 0, 0), rate=10,
                  latency=0, noise=0, env=0, owner=0):
    return SensorConfig(name or sensor_type, sensor_type, env, owner, mount, q, rate, latency,
                        noise, 17, "1", "fixture")


def world_spec(*, entities=None, bottom=10, visibility=1000):
    raw = {"source": {"kind": "parametric"}, "environment": {
        "current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
        "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
        "waves": {"kind": "calm"}, "visibility_m": visibility},
        "static_entities": entities or [],
    }
    if bottom is not None:
        raw["bathymetry"] = {"kind": "flat", "bottom_ned_z_m": bottom, "vertical_datum": "MSL"}
    return World.model_validate(raw)


def entity(id, x, y=0, z=0, radius=0.5):
    return {"id": id, "position_ned_m": [x, y, z], "shape": {"kind": "sphere", "radius_m": radius}}


def context(*, position=(0, 0, 0), q=(1, 0, 0, 0), nu=(0, 0, 0, 0, 0, 0),
            entities=None, bottom=10, visibility=1000, time=0, linear_acc=(0, 0, 0), angular_acc=(0, 0, 0)):
    state = VesselState(vector(position), vector(q), vector(nu))
    return SensorContext(state, ParametricWorld(world_spec(entities=entities, bottom=bottom, visibility=visibility), 0), time,
                         vector(linear_acc), vector(angular_acc))


def test_mount_pose_and_lever_arm_velocity():
    config = sensor_config("gps", mount=(0, 2, 0))
    ctx = context(nu=(1, 0, 0, 0, 0, 3))
    kin = mount_kinematics(config, ctx)
    assert kin.position_ned_m.tolist() == pytest.approx([0, 2, 0])
    assert kin.velocity_ned_mps.tolist() == pytest.approx([-5, 0, 0])
    assert kin.acceleration_body_mps2.tolist() == pytest.approx([0, -15, 0])


def test_imu_rotating_frame_and_gravity():
    imu = IMU(sensor_config("imu", mount=(0, 1, 0)))
    ctx = context(nu=(2, 0, 0, 0, 0, 3), linear_acc=(1, 0, 0))
    packet = imu.sample(ctx, sample_step=0)
    # ν_dot + ω×v + ω×(ω×r) = (1,6,0) + (0,-9,0).
    assert packet.values["specific_force_mount_mps2"].tolist() == pytest.approx([1, -3, -9.80665])
    assert packet.values["angular_rate_mount_radps"].tolist() == pytest.approx([0, 0, 3])


def test_gps_wgs84_and_lever_velocity():
    origin = (0.5, -1.0, 12.0)
    gps = GPS(sensor_config("gps", mount=(0, 1, 0)), origin_wgs84_rad_m=origin)
    packet = gps.sample(context(nu=(0, 0, 0, 0, 0, 2)), sample_step=0)
    geo = packet.values["wgs84_lat_lon_alt"]
    assert geodetic_to_ned(*geo.tolist(), origin) == pytest.approx((0, 1, 0), abs=1e-6)
    assert packet.values["velocity_ned_mps"].tolist() == pytest.approx([-2, 0, 0])
    assert packet.frames["wgs84_lat_lon_alt"] == "WGS84"


def test_lidar_mount_fov_resolution_range_and_no_return():
    lidar = LiDAR(sensor_config("lidar", mount=(1, 0, 0)), min_range_m=0, max_range_m=10,
                  fov_rad=math.pi/2, ray_count=3)
    packet = lidar.sample(context(entities=[entity("ahead", 5), entity("side", 1, 5)]), sample_step=0)
    assert packet.values["range_m"].shape == (3,)
    assert packet.values["hit"].tolist() == [False, True, False]
    assert packet.values["range_m"][1].item() == pytest.approx(3.5)
    obscured = lidar.sample(context(entities=[entity("ahead", 5)], visibility=3), sample_step=0)
    assert obscured.values["hit"].tolist() == [False, False, False]
    yaw90 = (math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    rotated = LiDAR(sensor_config("lidar", q=yaw90), min_range_m=0, max_range_m=10,
                    fov_rad=0, ray_count=1)
    rotated_packet = rotated.sample(context(entities=[entity("east", 0, 5)]), sample_step=0)
    assert rotated_packet.values["hit"].tolist() == [True]
    with pytest.raises(PhysicalValidationError):
        LiDAR(sensor_config("lidar"), min_range_m=0, max_range_m=10, fov_rad=1, ray_count=1)


def test_sonar_mount_bottom_range_and_missing_bathymetry():
    sonar = Sonar(sensor_config("sonar", mount=(0, 0, 1)), min_range_m=0, max_range_m=20,
                  fov_rad=math.pi/2, beam_count=3)
    packet = sonar.sample(context(bottom=10), sample_step=0)
    assert packet.values["hit"].tolist() == [True, True, True]
    assert packet.values["range_m"][1].item() == pytest.approx(9)
    assert packet.values["range_m"][0].item() == pytest.approx(9 / math.cos(math.pi/4))
    with pytest.raises(ExternalDataUnavailableError):
        sonar.sample(context(bottom=None), sample_step=0)


def test_abstract_sensor_hidden_state_independence():
    abstract = AbstractEntitySensor(sensor_config("abstract_entities"), max_range_m=10,
                                    horizontal_fov_rad=math.pi/2)
    visible = entity("visible", 5)
    hidden = entity("hidden", -100)
    first = abstract.sample(context(entities=[visible, hidden]), sample_step=0)
    changed = abstract.sample(context(entities=[visible, entity("hidden", -200, 50)]), sample_step=0)
    assert [d.entity_id for d in first.values["detections"]] == ["visible"]
    assert [d.entity_id for d in changed.values["detections"]] == ["visible"]
    assert torch.equal(first.values["detections"][0].relative_mount_m,
                       changed.values["detections"][0].relative_mount_m)


def test_noise_repeatable_by_seed_and_sample_step():
    gps = GPS(sensor_config("gps", noise=0.1), origin_wgs84_rad_m=(0, 0, 0))
    ctx = context()
    a = gps.sample(ctx, sample_step=4)
    b = gps.sample(ctx, sample_step=4)
    c = gps.sample(ctx, sample_step=5)
    assert torch.equal(a.values["velocity_ned_mps"], b.values["velocity_ned_mps"])
    assert not torch.equal(a.values["velocity_ned_mps"], c.values["velocity_ned_mps"])


def test_scheduler_rate_latency_and_missing_context_fails():
    gps = GPS(sensor_config("gps", rate=5, latency=2), origin_wgs84_rad_m=(0, 0, 0))
    scheduler = SensorScheduler((gps,), master_dt_s=0.1)
    delivered = []
    for step in range(5):
        delivered.extend(scheduler.tick(step, {(0, 0): context(time=step * 0.1)}))
    assert [(p.sample_step, p.delivery_step) for p in delivered] == [(0, 2), (2, 4)]
    with pytest.raises(PhysicalValidationError):
        SensorScheduler((gps,), master_dt_s=0.09)
    scheduler = SensorScheduler((gps,), master_dt_s=0.1)
    with pytest.raises(PhysicalValidationError):
        scheduler.tick(0, {})


def test_registry_unknown_sensor_and_ground_truth_separation():
    registry = SensorRegistry()
    with pytest.raises(UnknownReferenceError):
        registry.create(sensor_config("unknown"))
    debug = registry.create(sensor_config("ground_truth_state"))
    packet = debug.sample(context(), sample_step=0)
    assert packet.kind == "ground_truth"
    field = ObservationField("position", debug.config.instance_id, "ground_truth_state", "1", "fixture",
                             "ground_truth", "position_ned_m",
                             (3,), "float64", "m", "NED")
    with pytest.raises(PhysicalValidationError):
        ObservationContract((field,))
    contract = ObservationContract((field,), allow_ground_truth=True)
    assert contract.assemble({debug.config.instance_id: packet})["position"].shape == (3,)


def test_observation_hash_fields_order_and_metadata_validation():
    imu = IMU(sensor_config("imu"))
    packet = imu.sample(context(), sample_step=0)
    force = ObservationField("force", "imu", "imu", "1", "fixture", "physical", "specific_force_mount_mps2",
                             (3,), "float64", "m/s^2", "mount")
    rate = ObservationField("rate", "imu", "imu", "1", "fixture", "physical", "angular_rate_mount_radps",
                            (3,), "float64", "rad/s", "mount")
    a, b = ObservationContract((force, rate)), ObservationContract((rate, force))
    assert a.content_hash != b.content_hash
    newer_force = ObservationField("force", "imu", "imu", "2", "fixture", "physical",
                                   "specific_force_mount_mps2", (3,), "float64", "m/s^2", "mount")
    assert ObservationContract((newer_force, rate)).content_hash != a.content_hash
    assert list(a.assemble({"imu": packet})) == ["force", "rate"]
    with pytest.raises(PhysicalValidationError):
        ObservationContract((ObservationField("force", "imu", "imu", "1", "fixture", "physical", "specific_force_mount_mps2",
                                              (3,), "float64", "N", "mount"),)).assemble({"imu": packet})
    with pytest.raises(PhysicalValidationError):
        a.assemble({})
