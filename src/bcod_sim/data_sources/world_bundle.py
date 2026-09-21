"""Frozen local real-world bundle; all source access ends before construction."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.config.hashing import content_hash
from bcod_sim.config.models import Region, World
from bcod_sim.core.errors import ExternalDataCoverageError, PhysicalValidationError
from bcod_sim.data_sources.ais import AIS
from bcod_sim.data_sources.coops import COOPS
from bcod_sim.data_sources.enc import NOAAENC
from bcod_sim.data_sources.gebco import GEBCO
from bcod_sim.data_sources.ndbc import NDBC
from bcod_sim.data_sources.nws import NWS
from bcod_sim.data_sources.rtofs import RTOFS3D
from bcod_sim.frames.geodesy import ned_to_geodetic
from bcod_sim.world.entities import snapshot_scripted, snapshot_static
from bcod_sim.world.wake import WakeField
from bcod_sim.world.world import WorldSample


@dataclass(frozen=True)
class RealWorldSources:
    gebco: GEBCO
    enc: NOAAENC
    rtofs: RTOFS3D
    ndbc: NDBC
    nws: NWS
    coops: COOPS
    ais: AIS


class RealWorldBundle:
    """Canonical local query interface with no fetch or mutable source adapter."""

    def __init__(self, spec: World, env_id: int, *, origin_wgs84_rad_m: tuple[float, float, float],
                 sources: RealWorldSources) -> None:
        if spec.source.kind != "real_world" or env_id < 0:
            raise PhysicalValidationError("Real-world bundle requires real_world spec and nonnegative env ID")
        self.spec, self.env_id, self.origin, self.sources = spec, env_id, origin_wgs84_rad_m, sources
        self.wake = WakeField()
        self.bathymetry = self
        self._collision_tiles = {}
        self.vertical_datum = sources.gebco.provenance.datum
        self.provenance = tuple(getattr(sources, name).provenance for name in
            ("gebco", "enc", "rtofs", "ndbc", "nws", "coops", "ais"))
        self.content_hash = content_hash({"origin_wgs84_rad_m": origin_wgs84_rad_m,
            "sources": [{"source": p.source, "product": p.product, "version": p.version,
                         "valid_time": p.valid_time, "checksum": p.payload_sha256,
                         "datum": p.datum} for p in self.provenance]})

    def manifest(self) -> dict:
        return {"bundle_hash": self.content_hash, "origin_wgs84_rad_m": self.origin,
                "sources": [{"source": p.source, "product": p.product, "version": p.version,
                             "valid_time": p.valid_time, "units": dict(p.units), "frame": p.frame,
                             "datum": p.datum, "coverage": vars(p.coverage),
                             "payload_sha256": p.payload_sha256} for p in self.provenance]}

    def _validate(self, positions: torch.Tensor, sim_time_s: float, env_id: int) -> None:
        if env_id != self.env_id or not math.isfinite(sim_time_s) or sim_time_s < 0:
            raise PhysicalValidationError("Invalid real-world query identity or time")
        if positions.ndim != 2 or positions.shape[1] != 3 or not positions.is_floating_point() or not torch.isfinite(positions).all().item():
            raise PhysicalValidationError("Real-world positions require finite floating [N,3] NED tensor")
        if self.spec.boundary is not None:
            low, high = positions.new_tensor(self.spec.boundary.min_ned_m), positions.new_tensor(self.spec.boundary.max_ned_m)
            if ((positions < low) | (positions > high)).any().item():
                raise ExternalDataCoverageError("World query outside declared local boundary")

    def _geographic(self, position: torch.Tensor) -> tuple[float, float]:
        lat, lon, _ = ned_to_geodetic(tuple(position.tolist()), self.origin)
        return math.degrees(lat), math.degrees(lon)

    def bottom_ned_z_m(self, positions_ned_m: torch.Tensor, *, sim_time_s: float = 0,
                       env_id: int | None = None) -> torch.Tensor:
        query_env = self.env_id if env_id is None else env_id
        self._validate(positions_ned_m, sim_time_s, query_env)
        return positions_ned_m.new_tensor([self.sources.gebco.bottom_ned_z_m(*self._geographic(row))
                                           for row in positions_ned_m])

    def sample(self, positions_ned_m: torch.Tensor, *, sim_time_s: float, env_id: int,
               receiver_vessel_id: int | None = None) -> WorldSample:
        self._validate(positions_ned_m, sim_time_s, env_id)
        currents, winds, surfaces, heights, periods, directions, visibility, rain, fog, bottoms = ([] for _ in range(10))
        for row in positions_ned_m:
            lat, lon = self._geographic(row)
            bottom_value = self.sources.gebco.bottom_ned_z_m(lat, lon)
            currents.append(self.sources.rtofs.current_ned_mps(lat, lon, max(0.0, row[2].item())))
            weather = self.sources.nws.weather(lat, lon)
            wave = self.sources.ndbc.waves(lat, lon)
            water = self.sources.coops.water_level(lat, lon)
            winds.append(weather.wind_ned_mps); visibility.append(weather.visibility_m); rain.append(weather.rain_rate_mps)
            fog.append(weather.fog_extinction_per_m)
            surfaces.append(water.surface_ned_z_m); heights.append(wave.significant_height_m)
            periods.append(wave.period_s); directions.append(wave.direction_rad)
            bottoms.append(bottom_value)
        # Snapshot wave conditions use a deterministic local deep-water sinusoid.
        surface_values, orbital, accelerations = [], [], []
        for index, row in enumerate(positions_ned_m):
            omega = 2*math.pi/periods[index]
            phase = omega*sim_time_s - omega**2/9.80665 * (
                row[0].item()*math.cos(directions[index]) + row[1].item()*math.sin(directions[index]))
            amplitude = heights[index]/2
            surface_values.append(surfaces[index] - amplitude*math.cos(phase))
            speed = amplitude*omega*math.cos(phase)
            orbital.append((speed*math.cos(directions[index]), speed*math.sin(directions[index]),
                            amplitude*omega*math.sin(phase)))
            acceleration=amplitude*omega**2
            accelerations.append((-acceleration*math.sin(phase)*math.cos(directions[index]),
                                  -acceleration*math.sin(phase)*math.sin(directions[index]),
                                  acceleration*math.cos(phase)))
        count = positions_ned_m.shape[0]
        return WorldSample(positions_ned_m.new_tensor(currents), positions_ned_m.new_tensor(winds),
            self.wake.sample(positions_ned_m, env_id=env_id, receiver_vessel_id=receiver_vessel_id),
            positions_ned_m.new_tensor(surface_values), positions_ned_m.new_tensor(orbital), positions_ned_m.new_tensor(accelerations),
            positions_ned_m.new_full((count,),1025.0), positions_ned_m.new_full((count,),1.225),
            positions_ned_m.new_tensor(visibility), positions_ned_m.new_tensor(rain), positions_ned_m.new_tensor(fog),
            positions_ned_m.new_tensor(bottoms), self.vertical_datum,
            current_valid=positions_ned_m[:,2] <= positions_ned_m.new_tensor(bottoms),
            current_depth_semantics="source_derived_3d",
            local_water_depth_m=positions_ned_m.new_tensor(bottoms))

    def depth_at(self,north_m:float,east_m:float,*,surface_ned_z_m:float=0.)->float:
        p=torch.tensor(((north_m,east_m,0.),),dtype=torch.float64)
        return float(self.bottom_ned_z_m(p)[0])-surface_ned_z_m

    def seabed_position(self,north_m:float,east_m:float): return north_m,east_m,self.depth_at(north_m,east_m)

    def seabed_normal(self,north_m:float,east_m:float):
        epsilon=1.; dzdn=(self.depth_at(north_m+epsilon,east_m)-self.depth_at(north_m-epsilon,east_m))/(2*epsilon); dzde=(self.depth_at(north_m,east_m+epsilon)-self.depth_at(north_m,east_m-epsilon))/(2*epsilon)
        raw=(-dzdn,-dzde,1.); length=math.sqrt(sum(x*x for x in raw)); return tuple(x/length for x in raw)

    def is_wet(self,north_m:float,east_m:float,*,surface_ned_z_m:float=0.): return self.depth_at(north_m,east_m,surface_ned_z_m=surface_ned_z_m)>0

    def bounds(self):
        boundary=self.spec.boundary
        return None if boundary is None else (boundary.min_ned_m[0],boundary.max_ned_m[0],boundary.min_ned_m[1],boundary.max_ned_m[1])

    def collision_tile(self,north_m:float,east_m:float):
        from bcod_sim.collision.shapes import SeabedSurface
        config=self.spec.seabed_collision; key=(math.floor(north_m/config.tile_size_m),math.floor(east_m/config.tile_size_m))
        if key not in self._collision_tiles: self._collision_tiles[key]=SeabedSurface(self,key,config.tile_size_m,config.resolution_m)
        return self._collision_tiles[key]

    def entities(self, *, sim_time_s: float, env_id: int):
        if env_id != self.env_id or not math.isfinite(sim_time_s) or sim_time_s < 0:
            raise PhysicalValidationError("Invalid entity query")
        static = [snapshot_static(row.entity) for row in self.sources.enc.entities]
        traffic = [snapshot_scripted(row.entity, sim_time_s) for row in self.sources.ais.traffic]
        return tuple(sorted((*static, *(row for row in traffic if row is not None)), key=lambda row: row.id))

    def spawn_region(self, region_id: str) -> Region:
        for region in self.spec.spawn_regions:
            if region.id == region_id:
                return region
        raise ExternalDataCoverageError(f"Unknown spawn region: {region_id}")
