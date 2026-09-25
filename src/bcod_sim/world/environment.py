"""Typed, provider-extensible sensor view of the authoritative world."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import torch

from bcod_sim.core.errors import (ExternalDataCoverageError, InvalidMediumError,
                                  MissingCapabilityError, PhysicalValidationError)
from bcod_sim.frames.geodesy import ned_to_geodetic


class Medium(str, Enum):
    AIR = "AIR"
    WATER = "WATER"
    SOLID = "SOLID/SEABED"
    OUT_OF_COVERAGE = "OUT_OF_COVERAGE"


@dataclass(frozen=True)
class CapabilityDescriptor:
    name: str
    units: str
    frame: str
    valid_media: frozenset[Medium]
    spatial_semantics: str
    temporal_semantics: str
    provenance: str


@dataclass(frozen=True)
class DomainValue:
    value: Any
    descriptor: CapabilityDescriptor
    medium: Medium | tuple[Medium, ...]
    valid: bool = True
    status: str = "valid"


Provider = Callable[..., Any]


class CapabilityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[Provider, CapabilityDescriptor]] = {}

    def register(self, name: str, provider: Provider, descriptor: CapabilityDescriptor) -> None:
        if not name or descriptor.name != name:
            raise PhysicalValidationError("Capability name and descriptor must agree")
        if name in self._entries:
            raise PhysicalValidationError(f"Duplicate environment capability: {name}")
        self._entries[name] = (provider, descriptor)

    def has(self, name: str) -> bool:
        return name in self._entries

    def require(self, names: Iterable[str]) -> None:
        missing = sorted(set(names) - self._entries.keys())
        if missing:
            raise MissingCapabilityError(f"Missing environment capabilities: {', '.join(missing)}")

    def describe(self, name: str) -> CapabilityDescriptor:
        try:
            return self._entries[name][1]
        except KeyError as exc:
            raise MissingCapabilityError(f"Unknown environment capability: {name}") from exc

    @property
    def descriptors(self) -> Mapping[str, CapabilityDescriptor]:
        return MappingProxyType({name: row[1] for name, row in self._entries.items()})

    def provider(self, name: str) -> Provider:
        try:
            return self._entries[name][0]
        except KeyError as exc:
            raise MissingCapabilityError(f"Unknown environment capability: {name}") from exc


def _descriptor(name: str, units: str, frame: str, media: Iterable[Medium], spatial: str,
                temporal: str = "sampled at simulation time", provenance: str = "canonical world"):
    return CapabilityDescriptor(name, units, frame, frozenset(media), spatial, temporal, provenance)


class Environment:
    """Public domain API. Providers may be added without changing WorldSample."""

    def __init__(self, world: Any, *, state: Any | None = None,
                 linear_acceleration_body_mps2: torch.Tensor | None = None,
                 actuator_state: Mapping[str, object] | None = None,
                 registry: CapabilityRegistry | None = None) -> None:
        self.world, self.state = world, state
        self.linear_acceleration_body_mps2 = linear_acceleration_body_mps2
        self.actuator_state = MappingProxyType(dict(actuator_state or {}))
        self.registry = registry or CapabilityRegistry()
        if registry is None:
            self._register_world_capabilities()

    def has(self, capability: str) -> bool:
        return self.registry.has(capability)

    def require(self, capabilities: Iterable[str]) -> None:
        self.registry.require(capabilities)

    def describe(self, capability: str) -> CapabilityDescriptor:
        return self.registry.describe(capability)

    def medium(self, position_ned_m: torch.Tensor, time_s: float) -> Medium | tuple[Medium, ...]:
        points = _points(position_ned_m)
        try:
            if hasattr(self.world, "sources"):
                bottoms = self.world.bottom_ned_z_m(points, sim_time_s=time_s, env_id=self.world.env_id)
                surfaces = points.new_tensor([self.world.sources.coops.water_level(
                    *self.world._geographic(point)).surface_ned_z_m for point in points])
            else:
                sample = self.world.sample(points, sim_time_s=time_s, env_id=self.world.env_id)
                bottoms, surfaces = sample.bottom_ned_z_m, sample.wave_surface_ned_z_m
        except ExternalDataCoverageError:
            return Medium.OUT_OF_COVERAGE if points.shape[0] == 1 else tuple(Medium.OUT_OF_COVERAGE for _ in points)
        result = []
        for index, point in enumerate(points):
            surface = float(surfaces[index])
            bottom = None if bottoms is None else float(bottoms[index])
            z = float(point[2])
            result.append(Medium.AIR if z < surface else Medium.SOLID if bottom is not None and z > bottom else Medium.WATER)
        return result[0] if len(result) == 1 else tuple(result)

    def sample(self, capability: str, position_ned_m: torch.Tensor, time_s: float, *,
               invalid: str = "raise", **kwargs) -> DomainValue:
        descriptor = self.describe(capability)
        points = _points(position_ned_m)
        media = self.medium(points, time_s)
        sequence = (media,) if isinstance(media, Medium) else media
        invalid_media = [medium for medium in sequence if medium not in descriptor.valid_media]
        if invalid_media and descriptor.valid_media:
            status = f"{capability} is invalid in {invalid_media[0].value}"
            if invalid == "status":
                return DomainValue(None, descriptor, media, False, status)
            raise InvalidMediumError(status)
        value = self.registry.provider(capability)(points, time_s=time_s, **kwargs)
        return DomainValue(value, descriptor, media)

    def _register_world_capabilities(self) -> None:
        all_media = (Medium.AIR, Medium.WATER, Medium.SOLID)
        water = (Medium.WATER,)
        air = (Medium.AIR,)
        boundary = all_media
        field = lambda name: (lambda p, time_s, **_: getattr(
            self.world.sample(p, sim_time_s=time_s, env_id=self.world.env_id), name))
        registrations = (
            ("water.current", field("current_ned_mps"), _descriptor("water.current", "m/s", "NED", water, "point vector field")),
            ("water.surface", field("wave_surface_ned_z_m"), _descriptor("water.surface", "m", "NED z positive down", boundary, "surface elevation field")),
            ("water.density", field("water_density_kg_m3"), _descriptor("water.density", "kg/m^3", "scalar", water, "point scalar field")),
            ("waves.surface", field("wave_surface_ned_z_m"), _descriptor("waves.surface", "m", "NED z positive down", boundary, "surface elevation field")),
            ("waves.velocity", field("wave_orbital_ned_mps"), _descriptor("waves.velocity", "m/s", "NED", water, "point vector field")),
            ("waves.acceleration", field("wave_acceleration_ned_mps2"), _descriptor("waves.acceleration", "m/s^2", "NED", water, "point vector field")),
            ("atmosphere.wind", field("wind_ned_mps"), _descriptor("atmosphere.wind", "m/s", "NED", air, "point vector field")),
            ("weather.visibility", field("visibility_m"), _descriptor("weather.visibility", "m", "scalar", all_media, "point scalar field")),
            ("weather.rain", field("rain_rate_mps"), _descriptor("weather.rain", "m/s", "scalar", air, "point scalar field")),
            ("weather.fog", field("fog_extinction_per_m"), _descriptor("weather.fog", "1/m", "scalar", air, "point scalar field")),
            ("geometry.entities", lambda p, time_s, **_: self.world.entities(sim_time_s=time_s, env_id=self.world.env_id), _descriptor("geometry.entities", "structured", "NED", all_media, "world entity snapshot")),
            ("geometry.raycast", self._raycast, _descriptor("geometry.raycast", "m", "NED", all_media, "nearest analytic ray intersection")),
            ("bathymetry.bottom_z", self._bottom, _descriptor("bathymetry.bottom_z", "m", "NED z positive down", boundary, "seabed boundary")),
            ("bathymetry.normal", self._normal, _descriptor("bathymetry.normal", "unitless", "NED", boundary, "seabed boundary normal")),
            ("bathymetry.depth", self._depth, _descriptor("bathymetry.depth", "m", "scalar", boundary, "water depth = bottom z - surface z")),
            ("bathymetry.wet", self._wet, _descriptor("bathymetry.wet", "bool", "scalar", boundary, "wet/dry boundary classification")),
            ("positioning.geodetic_transform", self._geodetic, _descriptor("positioning.geodetic_transform", "rad,rad,m", "WGS84", all_media, "NED-to-geodetic transform", "static")),
        )
        for name, provider, descriptor in registrations:
            if name.startswith("bathymetry.") and getattr(self.world, "bathymetry", None) is None:
                continue
            if name == "positioning.geodetic_transform" and not hasattr(self.world, "origin"):
                continue
            self.registry.register(name, provider, descriptor)
        if self.state is not None:
            vehicle = (
                ("vehicle.pose", lambda p, time_s, **_: {"position_ned_m": self.state.position_ned, "q_body_to_ned": self.state.q_body_to_ned}, _descriptor("vehicle.pose", "m,unitless", "NED/FRD", all_media, "owner vehicle", "simulation state")),
                ("vehicle.velocity", lambda p, time_s, **_: self.state.nu_body[:3], _descriptor("vehicle.velocity", "m/s", "FRD", all_media, "owner vehicle", "simulation state")),
                ("vehicle.angular_velocity", lambda p, time_s, **_: self.state.nu_body[3:], _descriptor("vehicle.angular_velocity", "rad/s", "FRD", all_media, "owner vehicle", "simulation state")),
                ("vehicle.acceleration", lambda p, time_s, **_: self.linear_acceleration_body_mps2, _descriptor("vehicle.acceleration", "m/s^2", "FRD", all_media, "owner vehicle", "simulation state")),
                ("vehicle.actuators", lambda p, time_s, **_: self.actuator_state, _descriptor("vehicle.actuators", "structured", "component-defined", all_media, "owner vehicle", "simulation state")),
            )
            for row in vehicle: self.registry.register(*row)
        for name, (provider, descriptor) in getattr(self.world, "capability_extensions", {}).items():
            self.registry.register(name, provider, descriptor)

    def _bottom(self, points: torch.Tensor, *, time_s: float, **_) -> torch.Tensor:
        return self.world.bottom_ned_z_m(points, sim_time_s=time_s, env_id=self.world.env_id)

    def _normal(self, points: torch.Tensor, *, time_s: float, **_) -> torch.Tensor:
        surface = self.world.bathymetry
        return points.new_tensor([surface.seabed_normal(float(p[0]), float(p[1])) for p in points])

    def _depth(self, points: torch.Tensor, *, time_s: float, **_) -> torch.Tensor:
        sample = self.world.sample(points, sim_time_s=time_s, env_id=self.world.env_id)
        assert sample.bottom_ned_z_m is not None
        return sample.bottom_ned_z_m - sample.wave_surface_ned_z_m

    def _wet(self, points: torch.Tensor, *, time_s: float, **_) -> torch.Tensor:
        return self._depth(points, time_s=time_s) > 0

    def _geodetic(self, points: torch.Tensor, *, time_s: float, **_) -> torch.Tensor:
        return points.new_tensor([ned_to_geodetic(tuple(p.tolist()), self.world.origin) for p in points])

    def _raycast(self, points: torch.Tensor, *, time_s: float, direction_ned: torch.Tensor,
                 max_range_m: float = float("inf"), include_entities: bool = True,
                 include_bathymetry: bool = True, **_) -> dict[str, object]:
        if points.shape[0] != 1:
            raise PhysicalValidationError("geometry.raycast currently accepts one origin")
        from bcod_sim.sensors.raycast import environment_distance
        direction = direction_ned / torch.linalg.vector_norm(direction_ned)
        distance, target = environment_distance(tuple(points[0].tolist()), tuple(direction.tolist()), self.world,
            sim_time_s=time_s, include_entities=include_entities, include_bathymetry=include_bathymetry)
        hit = distance is not None and distance <= max_range_m
        return {"distance_m": distance if hit else max_range_m, "hit": hit, "target": target if hit else None}


def _points(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != 3 or not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise PhysicalValidationError("Environment positions must be finite floating [3] or [N,3] NED tensors")
    return value
