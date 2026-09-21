"""Frozen parametric world and deterministic batched queries."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.config.models import Region, World
from bcod_sim.config.resolver import ResolvedExperiment
from bcod_sim.core.errors import ExternalDataCoverageError, ExternalDataUnavailableError, PhysicalValidationError
from bcod_sim.world.bathymetry import Bathymetry
from bcod_sim.world.entities import EntitySnapshot, snapshot_scripted, snapshot_static
from bcod_sim.world.fields import build_vector_field
from bcod_sim.world.waves import WaveField
from bcod_sim.world.wake import WakeField


@dataclass(frozen=True)
class WorldSample:
    current_ned_mps: torch.Tensor
    wind_ned_mps: torch.Tensor
    wake_ned_mps: torch.Tensor
    wave_surface_ned_z_m: torch.Tensor
    wave_orbital_ned_mps: torch.Tensor
    wave_acceleration_ned_mps2: torch.Tensor
    water_density_kg_m3: torch.Tensor
    air_density_kg_m3: torch.Tensor
    visibility_m: torch.Tensor
    rain_rate_mps: torch.Tensor
    fog_extinction_per_m: torch.Tensor
    bottom_ned_z_m: torch.Tensor | None
    bathymetry_vertical_datum: str | None
    current_valid: torch.Tensor | None = None
    current_depth_semantics: str = "source_2d_vertically_uniform"
    local_water_depth_m: torch.Tensor | None = None
    wave_number_per_m: torch.Tensor | None = None
    wave_phase_velocity_mps: torch.Tensor | None = None
    wave_group_velocity_mps: torch.Tensor | None = None
    wave_breaking_active: torch.Tensor | None = None


class ParametricWorld:
    """Base fields are frozen; shared wake state changes only at buffer swaps."""

    def __init__(self, spec: World, env_id: int) -> None:
        if spec.source.kind != "parametric":
            raise ExternalDataUnavailableError("Real-world world bundles are built in Phase 9")
        if spec.environment is None or env_id < 0:
            raise PhysicalValidationError("Parametric world requires environment and nonnegative env ID")
        if spec.obstacles:
            raise PhysicalValidationError("Referenced obstacle assets require a geometry resolver; use explicit static_entities in Phase 4")
        self.spec = spec
        self.env_id = env_id
        self.current = build_vector_field(spec.environment.current, env_id)
        self.wind = build_vector_field(spec.environment.wind, env_id)
        self.waves = WaveField(spec.environment.waves)
        self.wake = WakeField()
        self.bathymetry = Bathymetry(spec.bathymetry) if spec.bathymetry else None
        self._check_spawn_regions()

    @classmethod
    def from_resolved(cls, resolved: ResolvedExperiment, *, env_id: int) -> "ParametricWorld":
        if not isinstance(resolved, ResolvedExperiment):
            raise TypeError("World construction requires ResolvedExperiment")
        return cls(resolved.config.world, env_id)

    def _check_spawn_regions(self) -> None:
        boundary = self.spec.boundary
        if boundary is None:
            return
        for region in self.spec.spawn_regions:
            if any(low < outer_low or high > outer_high for low, high, outer_low, outer_high in
                   zip(region.min_ned_m, region.max_ned_m, boundary.min_ned_m, boundary.max_ned_m)):
                raise PhysicalValidationError(f"Spawn region {region.id} exceeds world boundary")

    def _validate_query(self, positions_ned_m: torch.Tensor, sim_time_s: float, env_id: int) -> None:
        if env_id != self.env_id:
            raise PhysicalValidationError("World query environment ID mismatch")
        if not math.isfinite(sim_time_s) or sim_time_s < 0:
            raise PhysicalValidationError("World query time must be finite and nonnegative")
        if positions_ned_m.ndim != 2 or positions_ned_m.shape[1] != 3 or not positions_ned_m.is_floating_point():
            raise PhysicalValidationError("World query positions must be a floating [N,3] NED tensor")
        if not torch.isfinite(positions_ned_m).all().item():
            raise PhysicalValidationError("World query positions must be finite")
        boundary = self.spec.boundary
        if boundary is not None:
            lower = positions_ned_m.new_tensor(boundary.min_ned_m)
            upper = positions_ned_m.new_tensor(boundary.max_ned_m)
            if ((positions_ned_m < lower) | (positions_ned_m > upper)).any().item():
                raise ExternalDataCoverageError("World query outside declared boundary")

    def sample(self, positions_ned_m: torch.Tensor, *, sim_time_s: float, env_id: int,
               receiver_vessel_id: int | None = None) -> WorldSample:
        self._validate_query(positions_ned_m, sim_time_s, env_id)
        environment = self.spec.environment
        assert environment is not None
        count = positions_ned_m.shape[0]
        bottom = self.bathymetry.bottom_ned_z_m(positions_ned_m) if self.bathymetry else None
        current = self.current.sample(positions_ned_m, sim_time_s, env_id)
        local_depth = bottom if bottom is not None else None
        current_valid = ((positions_ned_m[:,2] <= bottom) & (bottom > 0)) if bottom is not None else positions_ned_m.new_ones(count,dtype=torch.bool)
        surface, orbital, wave_acceleration, wave_diagnostics = self.waves.sample_kinematics(
            positions_ned_m, sim_time_s, local_depth_m=local_depth, current_ned_mps=current, diagnostics=True)
        result = WorldSample(
            current_ned_mps=current,
            wind_ned_mps=self.wind.sample(positions_ned_m, sim_time_s, env_id),
            wake_ned_mps=self.wake.sample(positions_ned_m, env_id=env_id, receiver_vessel_id=receiver_vessel_id),
            wave_surface_ned_z_m=surface,
            wave_orbital_ned_mps=orbital,
            wave_acceleration_ned_mps2=wave_acceleration,
            water_density_kg_m3=positions_ned_m.new_full((count,),1025.0),
            air_density_kg_m3=positions_ned_m.new_full((count,),1.225),
            visibility_m=positions_ned_m.new_full((count,), environment.visibility_m),
            rain_rate_mps=positions_ned_m.new_full((count,), environment.rain_rate_mps),
            fog_extinction_per_m=positions_ned_m.new_full((count,), environment.fog_extinction_per_m),
            bottom_ned_z_m=bottom,
            bathymetry_vertical_datum=self.bathymetry.vertical_datum if self.bathymetry else None,
            current_valid=current_valid,
            current_depth_semantics="source_2d_vertically_uniform",
            local_water_depth_m=local_depth,
            wave_number_per_m=wave_diagnostics["wave_number_per_m"],
            wave_phase_velocity_mps=wave_diagnostics["phase_velocity_mps"],
            wave_group_velocity_mps=wave_diagnostics["group_velocity_mps"],
            wave_breaking_active=wave_diagnostics["breaking_active"],
        )
        for name, value in vars(result).items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                raise PhysicalValidationError(f"World field {name} produced nonfinite values")
        return result

    def bottom_ned_z_m(self, positions_ned_m: torch.Tensor, *, sim_time_s: float, env_id: int) -> torch.Tensor:
        self._validate_query(positions_ned_m, sim_time_s, env_id)
        if self.bathymetry is None:
            raise ExternalDataUnavailableError("Bathymetry was not configured")
        return self.bathymetry.bottom_ned_z_m(positions_ned_m)

    def entities(self, *, sim_time_s: float, env_id: int) -> tuple[EntitySnapshot, ...]:
        if env_id != self.env_id or not math.isfinite(sim_time_s) or sim_time_s < 0:
            raise PhysicalValidationError("Invalid entity query")
        static = (snapshot_static(entity) for entity in self.spec.static_entities)
        scripted = (snapshot_scripted(entity, sim_time_s) for entity in self.spec.scripted_entities)
        return tuple(sorted((entity for entity in (*static, *scripted) if entity is not None), key=lambda e: e.id))

    def spawn_region(self, region_id: str) -> Region:
        for region in self.spec.spawn_regions:
            if region.id == region_id:
                return region
        raise ExternalDataCoverageError(f"Unknown spawn region: {region_id}")
