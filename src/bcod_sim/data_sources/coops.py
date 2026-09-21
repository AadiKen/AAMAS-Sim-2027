"""CO-OPS water level observations with explicit datum and radius."""

from dataclasses import dataclass
from datetime import datetime
import math

from bcod_sim.core.errors import ExternalDataUnavailableError
from bcod_sim.data_sources.base import GeoCoverage, Provenance, nearest, number, records, verify_json_payload


@dataclass(frozen=True)
class WaterLevelObservation:
    station_id: str
    surface_ned_z_m: float
    datum: str
    provenance: Provenance


class COOPS:
    def __init__(self, payload: bytes, checksum: str, *, request_coverage: GeoCoverage,
                 build_time: datetime, station_radius_m: float) -> None:
        verified = verify_json_payload(payload, checksum, source="NOAA CO-OPS",
            required_units={"water_level": "m"}, required_frame="WGS84",
            request_coverage=request_coverage, build_time=build_time)
        if verified.provenance.datum is None:
            raise ExternalDataUnavailableError("CO-OPS water level datum is required")
        if not math.isfinite(station_radius_m) or station_radius_m <= 0:
            raise ValueError("CO-OPS station radius must be positive")
        self.provenance, self.station_radius_m = verified.provenance, station_radius_m
        self.stations = records(verified.document, "stations", {"station_id", "lat_deg", "lon_deg", "water_level_m"})
        for row in self.stations:
            if not isinstance(row["station_id"], str) or not row["station_id"]:
                raise ExternalDataUnavailableError("CO-OPS station identity is required")
            self.provenance.coverage.require(number(row, "lat_deg"), number(row, "lon_deg"))
            number(row, "water_level_m")

    def water_level(self, lat_deg: float, lon_deg: float) -> WaterLevelObservation:
        self.provenance.coverage.require(lat_deg, lon_deg)
        row = nearest(self.stations, lat_deg, lon_deg, max_radius_m=self.station_radius_m)
        return WaterLevelObservation(str(row["station_id"]), -float(row["water_level_m"]),
                                     self.provenance.datum, self.provenance)
