"""Strict runtime construction from resolved versioned definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from bcod_sim.actuators.autopilot import HeadingSpeedAutopilot
from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster
from bcod_sim.actuators.physical import ActuatorPipeline, ActuatorSet, SpeedHeadingController
from bcod_sim.collision.shapes import Box, Sphere
from bcod_sim.config.resolver import ResolvedExperiment
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import LinearEnvironmentLoads
from bcod_sim.core.errors import ConfigSchemaError
from bcod_sim.dynamics.damping import Damping, CoupledDampingTerm
from bcod_sim.dynamics.crossflow import StripTheoryCrossflow, SectionalCrossflow
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.mesh_buoyancy import MeshBuoyancy, TriangleMesh
from bcod_sim.dynamics.environmental import KinematicWaveLoads,RelativeWindLoads,WindCoefficientPoint
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import PlanarEquilibrium, Plant6
from bcod_sim.vessel_generation.spec_a.fit import surface_from_payload
from bcod_sim.dynamics.restoring import Hydrostatics, LinearHydrostatics, RestoringLUT, reference_point_transform
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
    speed_dependent_linear_damping_matrix_per_mps: tuple[tuple[float, float, float, float, float, float], ...] | None = None
    coupled_damping_terms: tuple[dict, ...] = ()
    buoyancy_n: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    center_buoyancy_frd_m: tuple[float, float, float] | None = None
    hydrostatics: dict | None = None
    crossflow: dict | None = None
    maneuvering_surface: dict | None = None
    maneuvering_interpretation: Literal["total_steady_hull_load", "residual_viscous"] = "residual_viscous"
    steady_coriolis_owner: Literal["bem", "maneuvering_model"] = "bem"
    added_mass_coriolis_enabled: bool = True
    surge_resistance: dict | None = None
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
        expected_owner = ("maneuvering_model" if self.maneuvering_interpretation == "total_steady_hull_load" else "bem")
        if self.steady_coriolis_owner != expected_owner:
            raise ValueError("maneuvering interpretation conflicts with steady added-mass Coriolis owner")
        expected_enabled = self.steady_coriolis_owner == "bem"
        surface_fallback = self.maneuvering_surface is not None and expected_enabled and not self.added_mass_coriolis_enabled
        if self.added_mass_coriolis_enabled != expected_enabled and not surface_fallback:
            raise ValueError("steady Coriolis ownership conflicts with added_mass_coriolis_enabled")
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
    if not matches or len({definition.content_hash for definition in matches}) != 1:
        raise ConfigSchemaError(f"Resolved {kind} definition mismatch: {reference}")
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


def build_engine(resolved: ResolvedExperiment, *, observation_contracts=None) -> EpisodeEngine:
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
        elif spec.hydrostatics.get("model") == "restoring_lut":
            item=spec.hydrostatics
            if item.get("out_of_domain")!="error" or item.get("interpolation")!="trilinear":
                raise ConfigSchemaError("Unsupported restoring LUT interpolation policy")
            axes=item["axes"]
            hydro=RestoringLUT(tensor(axes["heave_m"]),tensor(axes["roll_rad"]),
                               tensor(axes["pitch_rad"]),tensor(item["wrench_frd"]))
        elif "axes" in spec.hydrostatics and "wrench_frd" in spec.hydrostatics:
            item = spec.hydrostatics
            axes = item["axes"]
            hydro = RestoringLUT(tensor(axes["heave_m"]), tensor(axes["roll_rad"]),
                                 tensor(axes["pitch_rad"]), tensor(item["wrench_frd"]))
        else:
            raise ConfigSchemaError("Unsupported explicit hydrostatic model")
        crossflow = None
        if spec.crossflow is not None:
            item = spec.crossflow
            if item.get("model")=="strip_theory":
                geometry = item["geometry"]
                crossflow = StripTheoryCrossflow.constant_section(geometry["length_m"], geometry["beam_m"],
                    geometry["draft_m"], item["integration"]["strips"], water_density_kg_m3=item.get("water_density_kg_m3",1025),
                    include_vertical=item.get("include_vertical",False), dtype=dtype)
            elif item.get("model")=="sectional_stations":
                crossflow=SectionalCrossflow.from_stations(item["stations"],
                    density=item.get("water_density_kg_m3",1025.),
                    cd_scale=item.get("cd_scale",1.),dtype=dtype)
            else: raise ConfigSchemaError("Unsupported crossflow model")
        resistance_curve=None
        if spec.surge_resistance is not None:
            item=spec.surge_resistance
            resistance_curve=(tensor(item["speed_mps"]),tensor(item["force_x_n"]))
        linear_matrix=tensor(spec.linear_damping_matrix) if spec.linear_damping_matrix is not None else None
        coupled=tuple(CoupledDampingTerm(**term) for term in spec.coupled_damping_terms)
        speed_linear=(tensor(spec.speed_dependent_linear_damping_matrix_per_mps)
                      if spec.speed_dependent_linear_damping_matrix_per_mps is not None else None)
        plant = Plant6(properties, Damping(tensor(spec.linear_damping), tensor(spec.quadratic_damping),linear_matrix,coupled,
                                            surge_resistance_curve=resistance_curve,
                                            speed_dependent_linear_matrix_per_mps=speed_linear), hydro,
            OperatingEnvelope(tensor(spec.max_abs_nu), spec.min_substep_s, spec.max_substep_s),
            mode=resolved.config.simulation.dynamics_mode, planar_equilibrium=equilibrium, crossflow=crossflow,
            maneuvering_surface=(surface_from_payload(spec.maneuvering_surface)
                                 if spec.maneuvering_surface is not None else None),
            added_mass_coriolis_enabled=spec.added_mass_coriolis_enabled,
            steady_coriolis_owner=spec.steady_coriolis_owner,
            surface_min_forward_speed_mps=(spec.maneuvering_surface.get("min_forward_speed_mps",
                                                       .2*spec.maneuvering_surface["coefficients"]["reference_speed_mps"])
                                           if spec.maneuvering_surface is not None else 0.))
        collision = spec.collision
        if collision.get("kind") == "sphere" and set(collision) == {"kind", "radius_m"}:
            shape = Sphere(collision["radius_m"])
        elif collision.get("kind") == "box" and set(collision) == {"kind", "half_extents_m"}:
            shape = Box(tuple(collision["half_extents_m"]))
        else: raise ConfigSchemaError("Unsupported or malformed runtime collision shape")
        actuators = []
        physical_pipeline = None
        if authored.actuator_system is not None:
            try:
                system = authored.actuator_system
                if system.get("schema_version") != "manta-actuator-v1":
                    raise ValueError("Unsupported actuator schema")
                aset = ActuatorSet(list(system["actuators"]), allocation=system.get("control_allocation"))
                parameters = system.get("controller", {})
                effective_mass = float(plant.total_mass[0, 0])
                yaw_inertia = float(plant.total_mass[5, 5])
                controller = SpeedHeadingController(effective_mass, yaw_inertia,
                    float(spec.linear_damping[0]), float(spec.linear_damping[5])) if authored.controller.mode == "high_level" else None
                physical_pipeline = ActuatorPipeline(aset, controller,
                    float(system.get("allocation_period_s", resolved.config.simulation.master_dt_s)))
            except Exception as exc:
                raise ConfigSchemaError(f"Invalid physical actuator system for vessel {authored.instance_id}") from exc
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
                                     wind_loads=wind_loads,wave_loads=wave_loads,autopilot=autopilot,
                                     physical_pipeline=physical_pipeline))
    return EpisodeEngine(resolved, tuple(vessels), observation_contracts=observation_contracts)
