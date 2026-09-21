"""NDBC station wave observations with explicit radius coverage."""

from dataclasses import dataclass
from datetime import datetime
import math

from bcod_sim.core.errors import ExternalDataUnavailableError
from bcod_sim.data_sources.base import GeoCoverage, Provenance, nearest, number, records, verify_json_payload


@dataclass(frozen=True)
class WaveObservation:
    station_id: str
    significant_height_m: float
    period_s: float
    direction_rad: float
    provenance: Provenance


class NDBC:
    def __init__(self, payload: bytes, checksum: str, *, request_coverage: GeoCoverage,
                 build_time: datetime, station_radius_m: float) -> None:
        verified = verify_json_payload(payload, checksum, source="NOAA NDBC",
            required_units={"wave_height": "m", "wave_period": "s", "wave_direction": "deg"},
            required_frame="WGS84", request_coverage=request_coverage, build_time=build_time)
        if not math.isfinite(station_radius_m) or station_radius_m <= 0:
            raise ValueError("NDBC station radius must be positive")
        self.provenance, self.station_radius_m = verified.provenance, station_radius_m
        self.stations = records(verified.document, "stations", {"station_id", "lat_deg", "lon_deg",
            "significant_height_m", "period_s", "direction_deg"})
        for row in self.stations:
            if not isinstance(row["station_id"], str) or not row["station_id"]:
                raise ExternalDataUnavailableError("NDBC station identity is required")
            self.provenance.coverage.require(number(row, "lat_deg"), number(row, "lon_deg"))
            number(row, "significant_height_m", minimum=0); number(row, "period_s", strictly_positive=True)
            number(row, "direction_deg")

    def waves(self, lat_deg: float, lon_deg: float) -> WaveObservation:
        self.provenance.coverage.require(lat_deg, lon_deg)
        row = nearest(self.stations, lat_deg, lon_deg, max_radius_m=self.station_radius_m)
        return WaveObservation(str(row["station_id"]), float(row["significant_height_m"]),
                               float(row["period_s"]), math.radians(float(row["direction_deg"])),
                               self.provenance)
