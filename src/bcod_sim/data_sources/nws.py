"""NWS weather conversion to canonical SI and NED wind."""

from dataclasses import dataclass
from datetime import datetime
import math

from bcod_sim.data_sources.base import GeoCoverage, Provenance, nearest, number, records, verify_json_payload
from bcod_sim.frames.units import to_si


@dataclass(frozen=True)
class WeatherObservation:
    wind_ned_mps: tuple[float, float, float]
    visibility_m: float
    rain_rate_mps: float
    fog_extinction_per_m: float
    provenance: Provenance


class NWS:
    def __init__(self, payload: bytes, checksum: str, *, request_coverage: GeoCoverage,
                 build_time: datetime) -> None:
        verified = verify_json_payload(payload, checksum, source="NWS",
            required_units={"wind_speed": "km/h", "wind_direction_from": "deg",
                            "visibility": "km", "rain_rate": "mm/h", "fog_extinction": "1/m"},
            required_frame="WGS84", request_coverage=request_coverage, build_time=build_time)
        self.provenance = verified.provenance
        self.max_sample_radius_m = number(verified.document["metadata"], "max_sample_radius_m",
                                          strictly_positive=True)
        self.samples = records(verified.document, "samples", {"lat_deg", "lon_deg", "wind_speed_kmh",
            "wind_direction_from_deg", "visibility_km", "rain_mm_per_h", "fog_extinction_per_m"})
        for row in self.samples:
            self.provenance.coverage.require(number(row, "lat_deg"), number(row, "lon_deg"))
            number(row, "wind_speed_kmh", minimum=0); number(row, "wind_direction_from_deg")
            number(row, "visibility_km", strictly_positive=True); number(row, "rain_mm_per_h", minimum=0)
            number(row, "fog_extinction_per_m", minimum=0)

    def weather(self, lat_deg: float, lon_deg: float) -> WeatherObservation:
        self.provenance.coverage.require(lat_deg, lon_deg)
        row = nearest(self.samples, lat_deg, lon_deg, max_radius_m=self.max_sample_radius_m)
        speed = to_si(float(row["wind_speed_kmh"]), dimension="speed", unit="km/h")
        direction = math.radians(float(row["wind_direction_from_deg"]))
        # Meteorological direction is where wind comes from, clockwise from north.
        wind = (-speed * math.cos(direction), -speed * math.sin(direction), 0.0)
        visibility = to_si(float(row["visibility_km"]), dimension="length", unit="km")
        rain = float(row["rain_mm_per_h"]) / 1000 / 3600
        fog = float(row["fog_extinction_per_m"])
        if not math.isfinite(fog) or fog < 0:
            raise ValueError("NWS fog extinction must be finite and nonnegative")
        return WeatherObservation(wind, visibility, rain, fog, self.provenance)
