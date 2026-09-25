"""Strict runtime construction from resolved versioned definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from bcod_sim.actuators.autopilot import HeadingSpeedAutopilot
from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster
from bcod_sim.collision.shapes import Box, Sphere
from bcod_sim.config.resolver import ResolvedExperiment
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import LinearEnvironmentLoads
from bcod_sim.core.errors import ConfigSchemaError
from bcod_sim.dynamics.damping import Damping, CoupledDampingTerm
from bcod_sim.dynamics.crossflow import StripTheoryCrossflow
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.mesh_buoyancy import MeshBuoyancy, TriangleMesh
from bcod_sim.dynamics.environmental import KinematicWaveLoads,RelativeWindLoads,WindCoefficientPoint
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import PlanarEquilibrium, Plant6
from bcod_sim.dynamics.restoring import Hydrostatics, LinearHydrostatics, reference_point_transform
from bcod_sim.sensors.base import SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.ground_truth import GroundTruthState
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.lidar import LiDAR
from bcod_sim.sensors.sonar import Sonar
from bcod_sim.sensors.configs import GPSErrorModel, IMUErrorModel
from bcod_sim.sensors.registry import register_sensor_type, runtime_sensor_registry
from bcod_sim.config.hashing import content_hash


class Strict(BaseModel): model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class VesselRuntime(Strict):
    mass_kg: float = Field(gt=0, allow_inf_nan=False)
    cg_frd_m: tuple[float, float, float]
    inertia_cg_kg_m2: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    added_mass_kg: tuple[tuple[float, float, float, float, float, float], ...]
    linear_damping: tuple[float, float, float, float, float, float]
    quadratic_damping: tuple[float, float, float, float, float, float]
    linear_damping_matrix: tuple[tuple[float, float, float, float, float, float], ...] | None = None
    coupled_damping_terms: tuple[dict, ...] = ()
    buoyancy_n: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    center_buoyancy_frd_m: tuple[float, float, float] | None = None
    hydrostatics: dict | None = None
    crossflow: dict | None = None
    wind_loads: dict | None = None
    wave_loads: dict | None = None
    max_abs_nu: tuple[float, float, float, float, float, float]
    min_substep_s: float = Field(default=1e-6, gt=0, allow_inf_nan=False)
    max_substep_s: float = Field(gt=0, allow_inf_nan=False)
    planar_equilibrium: tuple[float, float, float] | None = None
    collision: dict
    environment_loads: tuple[float, float, float, float]
    autopilot_gains: tuple[float, float] | None = None
    geometry: dict | None = None
    equilibrium_heave_roll_pitch: tuple[float, float, float] | None = None
    provenance: dict | None = None
    validation_claim: Literal["synthetic/reference validation only"] | None = None
    validation_state: Literal["unvalidated"] | None = None

    @model_validator(mode="after")
    def hydrostatic_authority(self):
        legacy = self.buoyancy_n is not None or self.center_buoyancy_frd_m is not None
        if legacy and (self.buoyancy_n is None or self.center_buoyancy_frd_m is None):
            raise ValueError("Legacy buoyancy force and center must be provided together")
        if legacy == (self.hydrostatics is not None):
            raise ValueError("Select exactly one legacy or explicit hydrostatic model")
        return self


class FixedThrusterRuntime(Strict):
    kind: Literal["fixed_thruster"]
    mount_frd_m: tuple[float, float, float]
    mount_q_to_frd: tuple[float, float, float, float]
    thrust_bounds_n: tuple[float, float]
    command_bounds_policy: Literal["error", "clamp_with_event"] = "error"
    thrust_rate_limit_nps: float | None = None
    thrust_time_constant_s: float | None = None
    deadband_n: float = 0
    power_coefficient_w_per_n: float | None = None


class SensorRuntime(Strict):
    kind: str = Field(min_length=1)
    mount_frd_m: tuple[float, float, float] = (0, 0, 0)
    mount_q_to_frd: tuple[float, float, float, float] = (1, 0, 0, 0)
    rate_hz: float = Field(gt=0, allow_inf_nan=False)
    latency_steps: int = Field(default=0, ge=0)
    noise_std: float = Field(default=0, ge=0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)
    origin_wgs84_rad_m: tuple[float, float, float] | None = None
    min_range_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_range_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fov_rad: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    ray_count: int | None = Field(default=None, ge=1)
    beam_count: int | None = Field(default=None, ge=1)
    power_w: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    data_rate_bps: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    warmup_s: float = Field(default=0, ge=0, allow_inf_nan=False)
    dropout_probability: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    max_age_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    parameters: dict = {}


def _definition(resolved: ResolvedExperiment, kind: str, reference: str):
    identity, _, version = reference.partition("@")
    matches = [d for d in resolved.definitions if d.kind == kind and d.id == identity and d.version == version]
    if len(matches) != 1: raise ConfigSchemaError(f"Resolved {kind} definition mismatch: {reference}")
    return matches[0]


def _sensor(definition, vessel_id: int):
    try:
        spec = SensorRuntime.model_validate(dict(definition.payload))
        fingerprint = content_hash({"id": definition.id, "version": definition.version,
                                    "source": definition.source, "payload": dict(definition.payload),
                                    "env_id": 0, "owner_vessel_id": vessel_id})
        config = SensorConfig(definition.id, spec.kind, 0, vessel_id, spec.mount_frd_m,
                              spec.mount_q_to_frd, spec.rate_hz, spec.latency_steps,
                              spec.noise_std, spec.seed, definition.version, definition.source,
                              spec.power_w, spec.data_rate_bps, spec.warmup_s,
                              spec.dropout_probability, fingerprint, spec.max_age_s)
        if spec.kind == "gps":
            if spec.origin_wgs84_rad_m is None: raise ValueError("GPS origin is required")
            model = GPSErrorModel(**spec.parameters) if spec.parameters else None
            return GPS(config, origin_wgs84_rad_m=spec.origin_wgs84_rad_m, error_model=model)
        if spec.kind == "imu":
            model = IMUErrorModel(**spec.parameters) if spec.parameters else None
            return IMU(config, error_model=model)
        if spec.kind == "ground_truth_state": return GroundTruthState(config)
        if spec.kind not in {"lidar", "sonar"}:
            return runtime_sensor_registry.create(config, definition=dict(definition.payload),
                                                  parameters=dict(spec.parameters))
        if spec.min_range_m is None or spec.max_range_m is None or spec.fov_rad is None:
            raise ValueError("Range and FOV are required")
        if spec.kind == "lidar":
            if spec.ray_count is None: raise ValueError("LiDAR ray count is required")
            return LiDAR(config, min_range_m=spec.min_range_m, max_range_m=spec.max_range_m,
                         fov_rad=spec.fov_rad, ray_count=spec.ray_count)
        if spec.beam_count is None: raise ValueError("Sonar beam count is required")
        return Sonar(config, min_range_m=spec.min_range_m, max_range_m=spec.max_range_m,
                     fov_rad=spec.fov_rad, beam_count=spec.beam_count)
    except Exception as exc:
        raise ConfigSchemaError(f"Invalid runtime sensor definition: {definition.id}@{definition.version}") from exc


def build_engine(resolved: ResolvedExperiment) -> EpisodeEngine:
    vessels = []
    dtype = torch.float64 if resolved.config.experiment.numerical_profile == "validation" else torch.float32
    for vessel_id, authored in enumerate(resolved.config.vessels, start=1):
        try: spec = VesselRuntime.model_validate(dict(_definition(resolved, "vessel", authored.definition).payload))
        except Exception as exc: raise ConfigSchemaError(f"Invalid runtime vessel definition: {authored.definition}") from exc
        tensor = lambda value: torch.tensor(value, dtype=dtype)
        properties = MassProperties(spec.mass_kg, tensor(spec.cg_frd_m), tensor(spec.inertia_cg_kg_m2),
                                    tensor(spec.added_mass_kg))
        equilibrium = PlanarEquilibrium(*spec.planar_equilibrium) if spec.planar_equilibrium else None
        if spec.hydrostatics is None:
            assert spec.buoyancy_n is not None and spec.center_buoyancy_frd_m is not None
            hydro = Hydrostatics(spec.buoyancy_n, tensor(spec.center_buoyancy_frd_m))
        elif spec.hydrostatics.get("model") == "linear_matrix":
            item = spec.hydrostatics
            source = tensor(item["stiffness_6x6"]); reference = tensor(item.get("reference_point_frd_m", (0,0,0)))
            resolved_stiffness = reference_point_transform(source, reference)
            hydro_equilibrium = item["equilibrium"]
            hydro = LinearHydrostatics(resolved_stiffness, tensor(hydro_equilibrium["position_ned_m"]),
                tensor(hydro_equilibrium["orientation_rpy_rad"]), item["validity"]["max_abs_roll_rad"],
                item["validity"]["max_abs_pitch_rad"], item.get("allow_unstable", False), source, reference)
        elif spec.hydrostatics.get("model") == "mesh_buoyancy":
            item = spec.hydrostatics
            if item.get("runtime_backend", "exact") != "exact":
                raise ConfigSchemaError("Only exact mesh buoyancy backend is currently implemented")
            mesh = TriangleMesh.from_stl(item["hull_mesh"])
            if mesh.content_hash != item.get("hull_mesh_sha256"):
                raise ConfigSchemaError("Hull mesh content hash mismatch")
            envelope = item["operating_envelope"]
            hydro = MeshBuoyancy(mesh, item["water_density_kg_m3"], item["gravity_mps2"],
                item.get("water_level_ned_m", 0.0), tuple(envelope["heave_m"]),
                envelope["max_abs_roll_rad"], envelope["max_abs_pitch_rad"])
        else:
            raise ConfigSchemaError("Unsupported explicit hydrostatic model")
        crossflow = None
        if spec.crossflow is not None:
            item = spec.crossflow
            if item.get("model") != "strip_theory": raise ConfigSchemaError("Unsupported crossflow model")
            geometry = item["geometry"]
            crossflow = StripTheoryCrossflow.constant_section(geometry["length_m"], geometry["beam_m"],
                geometry["draft_m"], item["integration"]["strips"], water_density_kg_m3=item.get("water_density_kg_m3",1025),
                include_vertical=item.get("include_vertical",False), dtype=dtype)
        linear_matrix=tensor(spec.linear_damping_matrix) if spec.linear_damping_matrix is not None else None
        coupled=tuple(CoupledDampingTerm(**term) for term in spec.coupled_damping_terms)
        plant = Plant6(properties, Damping(tensor(spec.linear_damping), tensor(spec.quadratic_damping),linear_matrix,coupled), hydro,
            OperatingEnvelope(tensor(spec.max_abs_nu), spec.min_substep_s, spec.max_substep_s),
            mode=resolved.config.simulation.dynamics_mode, planar_equilibrium=equilibrium, crossflow=crossflow)
        collision = spec.collision
        if collision.get("kind") == "sphere" and set(collision) == {"kind", "radius_m"}:
            shape = Sphere(collision["radius_m"])
        elif collision.get("kind") == "box" and set(collision) == {"kind", "half_extents_m"}:
            shape = Box(tuple(collision["half_extents_m"]))
        else: raise ConfigSchemaError("Unsupported or malformed runtime collision shape")
        actuators = []
        for reference in authored.actuators:
            definition = _definition(resolved, "actuator", reference)
            try: actuator = FixedThrusterRuntime.model_validate(dict(definition.payload))
            except Exception as exc: raise ConfigSchemaError(f"Invalid runtime actuator definition: {reference}") from exc
            actuators.append(FixedThruster(ActuatorConfig(definition.id, 0, vessel_id, actuator.mount_frd_m,
                actuator.mount_q_to_frd, Bounds(*actuator.thrust_bounds_n), actuator.command_bounds_policy,
                actuator.thrust_rate_limit_nps, actuator.thrust_time_constant_s, actuator.deadband_n,
                actuator.power_coefficient_w_per_n)))
        sensors = tuple(_sensor(_definition(resolved, "sensor", reference), vessel_id)
                        for reference in authored.sensors)
        loads = LinearEnvironmentLoads(*spec.environment_loads)
        wind_loads = None
        if spec.wind_loads is not None:
            item=spec.wind_loads
            if item.get("model")!="coefficient_table": raise ConfigSchemaError("Unsupported wind-load model")
            points=tuple(WindCoefficientPoint(**point) for point in item["coefficients"])
            wind_loads=RelativeWindLoads(item["frontal_area_m2"],item["lateral_area_m2"],item["reference_height_m"],points,item.get("air_density_kg_m3",1.225))
        wave_loads = None
        if spec.wave_loads is not None:
            item=spec.wave_loads
            if item.get("model")!="kinematic_drag": raise ConfigSchemaError("Unsupported wave-load model")
            wave_loads=KinematicWaveLoads(tuple(item["linear_drag"]),tuple(item["quadratic_drag"]),tuple(item["inertia_coefficients"]))
        autopilot = HeadingSpeedAutopilot(*spec.autopilot_gains) if spec.autopilot_gains else None
        vessels.append(EpisodeVessel(authored.instance_id, vessel_id, plant, shape, tuple(actuators), sensors, loads,
                                     wind_loads=wind_loads,wave_loads=wave_loads,autopilot=autopilot))
    return EpisodeEngine(resolved, tuple(vessels))
