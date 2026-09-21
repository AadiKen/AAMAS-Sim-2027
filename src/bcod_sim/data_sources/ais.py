"""Fresh AIS traffic import into collision-capable scripted entities."""

from dataclasses import dataclass
from datetime import datetime
import math

from bcod_sim.config.models import BoxShape, ScriptedEntity
from bcod_sim.core.errors import ExternalDataUnavailableError
from bcod_sim.data_sources.base import GeoCoverage, Provenance, number, records, verify_json_payload
from bcod_sim.frames.geodesy import geodetic_to_ned
from bcod_sim.frames.units import to_si


@dataclass(frozen=True)
class AISTraffic:
    mmsi: str
    entity: ScriptedEntity
    provenance: Provenance


class AIS:
    def __init__(self, payload: bytes, checksum: str, *, request_coverage: GeoCoverage,
                 build_time: datetime, origin_wgs84_rad_m: tuple[float, float, float]) -> None:
        verified = verify_json_payload(payload, checksum, source="USCG AIS",
            required_units={"speed_over_ground": "kn", "course_over_ground": "deg", "dimensions": "m"},
            required_frame="WGS84", request_coverage=request_coverage, build_time=build_time)
        self.provenance = verified.provenance
        traffic = []
        reports = records(verified.document, "reports", {"mmsi", "lat_deg", "lon_deg", "sog_kn",
            "cog_deg", "length_m", "width_m"})
        for report in reports:
            try:
                if not isinstance(report["mmsi"], str) or not report["mmsi"]:
                    raise ValueError("mmsi")
                lat, lon = number(report, "lat_deg"), number(report, "lon_deg")
                self.provenance.coverage.require(lat, lon)
                position = geodetic_to_ned(math.radians(lat), math.radians(lon),
                                           0.0, origin_wgs84_rad_m)
                speed = to_si(number(report, "sog_kn", minimum=0), dimension="speed", unit="kn")
                course = math.radians(number(report, "cog_deg"))
                velocity = (speed * math.cos(course), speed * math.sin(course), 0.0)
                length = number(report, "length_m", strictly_positive=True)
                width = number(report, "width_m", strictly_positive=True)
                entity = ScriptedEntity(id=f"ais:{report['mmsi']}", position_ned_m=position,
                    shape=BoxShape(kind="box", half_extents_m=(length/2, width/2, max(0.5, width/4))),
                    collision_enabled=True, velocity_ned_mps=velocity, start_time_s=0)
            except (KeyError, TypeError, ValueError) as exc:
                raise ExternalDataUnavailableError("Malformed AIS report") from exc
            traffic.append(AISTraffic(str(report["mmsi"]), entity, self.provenance))
        if len({row.mmsi for row in traffic}) != len(traffic):
            raise ExternalDataUnavailableError("Duplicate AIS MMSI in snapshot")
        self.traffic = tuple(sorted(traffic, key=lambda row: row.mmsi))
