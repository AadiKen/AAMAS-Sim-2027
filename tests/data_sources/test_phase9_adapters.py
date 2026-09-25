from datetime import datetime, timezone
import hashlib
import json
import math

import pytest
import torch

from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.collision.shapes import Sphere
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.config.models import World
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.core.lifecycle import DirectAction
from bcod_sim.core.errors import ExternalDataCoverageError, ExternalDataUnavailableError, PhysicalValidationError
from bcod_sim.data_sources.ais import AIS
from bcod_sim.data_sources.base import GeoCoverage
from bcod_sim.data_sources.coops import COOPS
from bcod_sim.data_sources.enc import NOAAENC
from bcod_sim.data_sources.gebco import GEBCO
from bcod_sim.data_sources.ndbc import NDBC
from bcod_sim.data_sources.nws import NWS
from bcod_sim.data_sources.rtofs import RTOFS3D
from bcod_sim.data_sources.world_bundle import RealWorldBundle, RealWorldSources
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.sensors.base import SensorConfig, SensorContext
from bcod_sim.sensors.sonar import Sonar
from bcod_sim.state.vessel_state import VesselState


BUILD_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
COVERAGE = GeoCoverage(-1, 1, -1, 1)
REQUEST = GeoCoverage(-0.1, 0.1, -0.1, 0.1)
ORIGIN = (0.0, 0.0, 0.0)


def encoded(source, units, body, *, frame="WGS84", datum="MSL", extra=None,
            valid_from="2025-01-01T00:00:00Z", valid_until="2027-01-01T00:00:00Z"):
    metadata = {"source": source, "product": f"{source}-fixture", "version": "1",
        "valid_time": "2026-01-01T00:00:00Z", "valid_from": valid_from,
        "valid_until": valid_until, "units": units, "frame": frame, "datum": datum,
        "coverage": vars(COVERAGE)}
    metadata.update(extra or {})
    payload = json.dumps({"metadata": metadata, **body}, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(payload).hexdigest()


def fixtures():
    return {
        "gebco": encoded("GEBCO", {"elevation": "m"},
            {"samples": [{"lat_deg": 0, "lon_deg": 0, "elevation_m": -20}]},
            datum="MSL", extra={"vertical_convention": "elevation_positive_up", "max_sample_radius_m": 20000}),
        "enc": encoded("NOAA ENC", {"geometry": "m"}, {"features": [
            {"id": "buoy-1", "feature_class": "BOYLAT", "lat_deg": 0, "lon_deg": 0,
             "shape": {"kind": "sphere", "radius_m": 1.5}, "collision_enabled": True}]}),
        "rtofs": encoded("NOAA RTOFS", {"horizontal_velocity": "m/s", "vertical_velocity": "m/s", "depth": "m"},
            {"samples": [{"lat_deg": 0, "lon_deg": 0, "depth_m": 0,
                           "u_east_mps": 1, "v_north_mps": 2, "w_up_mps": 0.1}]},
            frame="geographic_ENU", datum=None, extra={"variable": "3d", "max_sample_radius_m": 20000}),
        "ndbc": encoded("NOAA NDBC", {"wave_height": "m", "wave_period": "s", "wave_direction": "deg"},
            {"stations": [{"station_id": "41001", "lat_deg": 0, "lon_deg": 0,
                            "significant_height_m": 2, "period_s": 8, "direction_deg": 90}]}),
        "nws": encoded("NWS", {"wind_speed": "km/h", "wind_direction_from": "deg",
                                  "visibility": "km", "rain_rate": "mm/h", "fog_extinction": "1/m"},
            {"samples": [{"lat_deg": 0, "lon_deg": 0, "wind_speed_kmh": 36,
                           "wind_direction_from_deg": 0, "visibility_km": 5, "rain_mm_per_h": 3.6,
                           "fog_extinction_per_m": 0.002}]}, extra={"max_sample_radius_m": 20000}),
        "coops": encoded("NOAA CO-OPS", {"water_level": "m"},
            {"stations": [{"station_id": "872", "lat_deg": 0, "lon_deg": 0, "water_level_m": 1.2}]},
            datum="MSL"),
        "ais": encoded("USCG AIS", {"speed_over_ground": "kn", "course_over_ground": "deg", "dimensions": "m"},
            {"reports": [{"mmsi": "123456789", "lat_deg": 0, "lon_deg": 0, "sog_kn": 10,
                           "cog_deg": 90, "length_m": 20, "width_m": 6}]}, datum=None),
    }


def build_all(data=None):
    data = data or fixtures()
    return {
        "gebco": GEBCO(*data["gebco"], request_coverage=REQUEST, build_time=BUILD_TIME),
        "enc": NOAAENC(*data["enc"], request_coverage=REQUEST, build_time=BUILD_TIME,
                       origin_wgs84_rad_m=ORIGIN),
        "rtofs": RTOFS3D(*data["rtofs"], request_coverage=REQUEST, build_time=BUILD_TIME),
        "ndbc": NDBC(*data["ndbc"], request_coverage=REQUEST, build_time=BUILD_TIME, station_radius_m=200000),
        "nws": NWS(*data["nws"], request_coverage=REQUEST, build_time=BUILD_TIME),
        "coops": COOPS(*data["coops"], request_coverage=REQUEST, build_time=BUILD_TIME, station_radius_m=200000),
        "ais": AIS(*data["ais"], request_coverage=REQUEST, build_time=BUILD_TIME,
                   origin_wgs84_rad_m=ORIGIN),
    }


def test_gebco_checksum_datum_coverage_and_known_fixture():
    payload, checksum = fixtures()["gebco"]
    adapter = GEBCO(payload, checksum, request_coverage=REQUEST, build_time=BUILD_TIME)
    assert adapter.bottom_ned_z_m(0, 0) == 20
    assert adapter.provenance.payload_sha256 == checksum and adapter.provenance.datum == "MSL"
    with pytest.raises(ExternalDataCoverageError):
        adapter.bottom_ned_z_m(2, 0)
    with pytest.raises(ExternalDataCoverageError):
        adapter.bottom_ned_z_m(0.5, 0)
    bad = json.loads(payload); bad["metadata"]["vertical_convention"] = "unknown"
    bad_payload = json.dumps(bad).encode()
    with pytest.raises(ExternalDataUnavailableError):
        GEBCO(bad_payload, hashlib.sha256(bad_payload).hexdigest(), request_coverage=REQUEST, build_time=BUILD_TIME)


def test_enc_entities_have_collision_geometry_identity_and_provenance():
    adapter = build_all()["enc"]
    row = adapter.entities[0]
    assert row.entity.id == "buoy-1" and row.entity.collision_enabled
    assert row.entity.shape.radius_m == 1.5 and row.feature_class == "BOYLAT"
    assert row.entity.semantic_class == "BOYLAT" and row.entity.source == "NOAA ENC"
    assert row.entity.provenance["payload_sha256"] == row.provenance.payload_sha256
    assert row.provenance.source == "NOAA ENC"


def test_rtofs_uses_requested_3d_depth_and_enu_to_ned_conversion():
    payload, checksum = fixtures()["rtofs"]
    adapter = RTOFS3D(payload, checksum, request_coverage=REQUEST, build_time=BUILD_TIME)
    assert adapter.current_ned_mps(0, 0, 0) == (2, 1, -0.1)
    with pytest.raises(ExternalDataCoverageError):
        adapter.current_ned_mps(0, 0, 5)
    document = json.loads(payload); document["metadata"]["variable"] = "2ds"
    invalid = json.dumps(document).encode()
    with pytest.raises(ExternalDataUnavailableError):
        RTOFS3D(invalid, hashlib.sha256(invalid).hexdigest(), request_coverage=REQUEST, build_time=BUILD_TIME)


def test_rtofs_interpolates_between_depth_levels():
    payload, checksum = encoded("NOAA RTOFS",
        {"horizontal_velocity": "m/s", "vertical_velocity": "m/s", "depth": "m"},
        {"samples": [
            {"lat_deg": 0, "lon_deg": 0, "depth_m": 0, "u_east_mps": 0, "v_north_mps": 2, "w_up_mps": 0},
            {"lat_deg": 0, "lon_deg": 0, "depth_m": 10, "u_east_mps": 10, "v_north_mps": 4, "w_up_mps": 2}]},
        frame="geographic_ENU", datum=None, extra={"variable": "3d", "max_sample_radius_m": 20000})
    adapter = RTOFS3D(payload, checksum, request_coverage=REQUEST, build_time=BUILD_TIME)
    assert adapter.current_ned_mps(0, 0, 5) == pytest.approx((3, 5, -1))


def test_ndbc_station_radius_units_and_known_fixture():
    payload, checksum = fixtures()["ndbc"]
    adapter = NDBC(payload, checksum, request_coverage=REQUEST, build_time=BUILD_TIME, station_radius_m=1000)
    wave = adapter.waves(0, 0)
    assert (wave.significant_height_m, wave.period_s, wave.direction_rad) == pytest.approx((2, 8, math.pi/2))
    with pytest.raises(ExternalDataCoverageError):
        adapter.waves(0.1, 0.1)


def test_nws_kmh_regression_and_all_si_conversions():
    weather = build_all()["nws"].weather(0, 0)
    assert weather.wind_ned_mps == pytest.approx((-10, 0, 0), abs=1e-12)
    assert weather.visibility_m == 5000
    assert weather.rain_rate_mps == pytest.approx(1e-6)
    assert weather.fog_extinction_per_m == 0.002


def test_coops_datum_station_radius_and_ned_surface_sign():
    payload, checksum = encoded("NOAA CO-OPS", {"water_level": "m"},
        {"stations": [{"station_id": "872", "lat_deg": 0, "lon_deg": 0, "water_level_m": 1.2}]}, datum="MLLW")
    adapter = COOPS(payload, checksum, request_coverage=REQUEST, build_time=BUILD_TIME, station_radius_m=1000)
    water = adapter.water_level(0, 0)
    assert water.surface_ned_z_m == -1.2 and water.datum == "MLLW"
    with pytest.raises(ExternalDataCoverageError):
        adapter.water_level(0.1, 0.1)


def test_ais_known_fixture_velocity_geometry_identity_and_provenance():
    traffic = build_all()["ais"].traffic[0]
    assert traffic.mmsi == "123456789" and traffic.entity.collision_enabled
    assert traffic.entity.shape.half_extents_m[:2] == (10, 3)
    assert traffic.entity.velocity_ned_mps == pytest.approx((0, 10*1852/3600, 0), abs=1e-12)
    assert traffic.entity.orientation_q_to_ned == pytest.approx((math.sqrt(.5), 0, 0, math.sqrt(.5)))
    assert traffic.entity.semantic_class == "vessel" and traffic.entity.source == "USCG AIS"
    assert traffic.provenance.source == "USCG AIS"


@pytest.mark.parametrize("name", ["gebco", "enc", "rtofs", "ndbc", "nws", "coops", "ais"])
def test_every_adapter_rejects_checksum_time_coverage_and_units(name):
    data = fixtures()
    payload, checksum = data[name]
    data[name] = (payload, "0"*64)
    with pytest.raises(ExternalDataUnavailableError):
        build_all(data)
    document = json.loads(payload); document["metadata"]["valid_until"] = "2025-06-01T00:00:00Z"
    changed = json.dumps(document).encode(); data = fixtures(); data[name] = (changed, hashlib.sha256(changed).hexdigest())
    with pytest.raises(ExternalDataUnavailableError):
        build_all(data)
    document = json.loads(payload); first_unit = next(iter(document["metadata"]["units"])); document["metadata"]["units"][first_unit] = "wrong"
    changed = json.dumps(document).encode(); data = fixtures(); data[name] = (changed, hashlib.sha256(changed).hexdigest())
    with pytest.raises(ExternalDataUnavailableError):
        build_all(data)
    with pytest.raises(ExternalDataCoverageError):
        constructors_outside(name, payload, checksum)


def constructors_outside(name, payload, checksum):
    outside = GeoCoverage(-2, 2, -2, 2)
    common = {"request_coverage": outside, "build_time": BUILD_TIME}
    if name == "gebco": return GEBCO(payload, checksum, **common)
    if name == "enc": return NOAAENC(payload, checksum, origin_wgs84_rad_m=ORIGIN, **common)
    if name == "rtofs": return RTOFS3D(payload, checksum, **common)
    if name == "ndbc": return NDBC(payload, checksum, station_radius_m=1000, **common)
    if name == "nws": return NWS(payload, checksum, **common)
    if name == "coops": return COOPS(payload, checksum, station_radius_m=1000, **common)
    return AIS(payload, checksum, origin_wgs84_rad_m=ORIGIN, **common)


def test_frozen_bundle_exposes_canonical_world_without_source_access():
    adapters = build_all()
    spec = World.model_validate({"source": {"kind": "real_world", "data_product": "bundle@1"},
        "boundary": {"id": "local", "min_ned_m": [-100, -100, -10], "max_ned_m": [100, 100, 50]}})
    bundle = RealWorldBundle(spec, 0, origin_wgs84_rad_m=ORIGIN, sources=RealWorldSources(**adapters))
    sample = bundle.sample(torch.tensor([[0., 0, 0]], dtype=torch.float64), sim_time_s=0, env_id=0)
    assert sample.current_ned_mps.tolist() == [[2, 1, -0.1]]
    assert sample.wind_ned_mps[0].tolist() == pytest.approx([-10, 0, 0])
    assert sample.bottom_ned_z_m.tolist() == [20]
    assert sample.bathymetry_vertical_datum == "MSL"
    assert [entity.id for entity in bundle.entities(sim_time_s=0, env_id=0)] == ["ais:123456789", "buoy-1"]
    assert len(bundle.content_hash) == 64 and len(bundle.provenance) == 7
    assert bundle.manifest()["bundle_hash"] == bundle.content_hash


def test_real_world_bundle_rejects_incompatible_vertical_datums():
    data = fixtures()
    data["coops"] = encoded("NOAA CO-OPS", {"water_level": "m"},
        {"stations": [{"station_id": "872", "lat_deg": 0, "lon_deg": 0, "water_level_m": 1.2}]}, datum="MLLW")
    adapters = build_all(data)
    spec = World.model_validate({"source": {"kind": "real_world", "data_product": "bundle@1"},
        "boundary": {"id": "local", "min_ned_m": [-100, -100, -10], "max_ned_m": [100, 100, 50]}})
    with pytest.raises(PhysicalValidationError, match="datum"):
        RealWorldBundle(spec, 0, origin_wgs84_rad_m=ORIGIN, sources=RealWorldSources(**adapters))


def test_sonar_uses_real_world_bundle_canonical_bathymetry():
    adapters = build_all()
    spec = World.model_validate({"source": {"kind": "real_world", "data_product": "bundle@1"},
        "boundary": {"id": "local", "min_ned_m": [-100, -100, -10], "max_ned_m": [100, 100, 50]}})
    bundle = RealWorldBundle(spec, 0, origin_wgs84_rad_m=ORIGIN, sources=RealWorldSources(**adapters))
    config = SensorConfig("depth", "sonar", 0, 1, (0, 0, 0), (1, 0, 0, 0), 10, 0, 0, 1, "1", "test")
    sonar = Sonar(config, min_range_m=0, max_range_m=30, fov_rad=0, beam_count=1)
    state = VesselState(torch.tensor((0., 0, 1), dtype=torch.float64),
                        torch.tensor((1., 0, 0, 0), dtype=torch.float64), torch.zeros(6, dtype=torch.float64))
    result = sonar.sample(SensorContext(state, bundle, 0), sample_step=0)
    assert result.values["range_m"].item() == pytest.approx(19)


def test_resolved_real_world_requires_exact_bundle_hash_and_runs_authoritative_engine():
    adapters = build_all()
    world_raw = {"source": {"kind": "real_world", "data_product": "bundle@1"},
        "boundary": {"id": "local", "min_ned_m": [-100, -100, -10], "max_ned_m": [100, 100, 50]}}
    spec = World.model_validate(world_raw)
    bundle = RealWorldBundle(spec, 0, origin_wgs84_rad_m=ORIGIN, sources=RealWorldSources(**adapters))
    registry = Registry()
    registry.register("vessel", "vessel", "1", {}, "test")
    registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": "agent",
                      "target_ned_m": [90, 90, 0], "radius_m": 0.1}, "test")
    registry.register("data_product", "bundle", "1", {"bundle_hash": bundle.content_hash}, "test")
    resolved = resolve({"schema_version": 1, "experiment": {"id": "real", "seed": 1},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.1, "dynamics_substeps": 1,
                       "policy_every_n_master_steps": 1, "max_master_steps": 2},
        "world": world_raw,
        "vessels": [{"instance_id": "agent", "definition": "vessel@1",
                     "controller": {"mode": "direct_actuator"},
                     "spawn": {"ned_m": [50, 50, 0], "rpy_rad": [0, 0, 0]}}],
        "task": {"type": "waypoint@1", "reward": {"individual_weight": 1, "team_weight": 0},
                 "disabled_agent_behavior": "deactivate_keep_physical"}}, registry)
    v = lambda values: torch.tensor(values, dtype=torch.float64)
    mass = MassProperties(10, v((0, 0, 0)), torch.diag(v((4, 5, 6))), torch.zeros((6, 6), dtype=torch.float64))
    plant = Plant6(mass, Damping(v((0,)*6), v((0,)*6)), Hydrostatics(10*9.80665, v((0, 0, 0))),
                   OperatingEnvelope(v((1e4,)*6), max_substep_s=0.2))
    prop = FixedThruster(ActuatorConfig("prop", 0, 1, (0, 0, 0), (1, 0, 0, 0), Bounds(-100, 100)))
    vessel = EpisodeVessel("agent", 1, plant, Sphere(0.2), (prop,), (), ExplicitZeroLoads())
    with pytest.raises(ExternalDataUnavailableError):
        EpisodeEngine(resolved, (vessel,))
    sim = EpisodeEngine(resolved, (vessel,), world=bundle)
    sim.reset()
    frame = sim.step({"agent": DirectAction((("prop", ThrustCommand(0)),))})
    assert frame.master_step == 1

    wrong_registry = Registry()
    wrong_registry.register("vessel", "vessel", "1", {}, "test")
    wrong_registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": "agent",
                            "target_ned_m": [90, 90, 0], "radius_m": 0.1}, "test")
    wrong_registry.register("data_product", "bundle", "1", {"bundle_hash": "0"*64}, "test")
    wrong = resolve(resolved.config.model_dump(mode="json"), wrong_registry)
    with pytest.raises(PhysicalValidationError, match="bundle hash"):
        EpisodeEngine(wrong, (vessel,), world=bundle)
