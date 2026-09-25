"""NOAA ENC chart objects converted to collision-capable local entities."""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Mapping

from bcod_sim.config.models import BoxShape, SphereShape, StaticEntity
from bcod_sim.core.errors import ExternalDataUnavailableError
from bcod_sim.data_sources.base import GeoCoverage, Provenance, number, records, verify_json_payload
from bcod_sim.frames.geodesy import geodetic_to_ned


@dataclass(frozen=True)
class ChartEntity:
    entity: StaticEntity
    feature_class: str
    provenance: Provenance


class NOAAENC:
    def __init__(self, payload: bytes, checksum: str, *, request_coverage: GeoCoverage,
                 build_time: datetime, origin_wgs84_rad_m: tuple[float, float, float]) -> None:
        verified = verify_json_payload(payload, checksum, source="NOAA ENC", required_units={"geometry": "m"},
            required_frame="WGS84", request_coverage=request_coverage, build_time=build_time)
        self.provenance = verified.provenance
        entities = []
        features = records(verified.document, "features", {"id", "feature_class", "lat_deg", "lon_deg",
            "shape", "collision_enabled"}, {"altitude_m"})
        for feature in features:
            try:
                lat, lon = number(feature, "lat_deg"), number(feature, "lon_deg")
                self.provenance.coverage.require(lat, lon)
                altitude = number(feature, "altitude_m") if "altitude_m" in feature else 0
                position = geodetic_to_ned(math.radians(lat), math.radians(lon), altitude, origin_wgs84_rad_m)
                shape_spec = feature["shape"]
                if not isinstance(shape_spec, Mapping):
                    raise ValueError("shape")
                if shape_spec.get("kind") == "sphere" and set(shape_spec) == {"kind", "radius_m"}:
                    shape = SphereShape(kind="sphere", radius_m=shape_spec["radius_m"])
                elif shape_spec.get("kind") == "box" and set(shape_spec) == {"kind", "half_extents_m"}:
                    shape = BoxShape(kind="box", half_extents_m=tuple(shape_spec["half_extents_m"]))
                else:
                    raise ValueError("shape")
                if not isinstance(feature["collision_enabled"], bool):
                    raise ValueError("collision")
                entity = StaticEntity(id=feature["id"], position_ned_m=position, shape=shape,
                                      collision_enabled=feature["collision_enabled"],
                                      semantic_class=str(feature["feature_class"]), source="NOAA ENC",
                                      provenance={"product": self.provenance.product,
                                                  "version": self.provenance.version,
                                                  "payload_sha256": self.provenance.payload_sha256})
                feature_class = feature["feature_class"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ExternalDataUnavailableError("Malformed ENC feature") from exc
            entities.append(ChartEntity(entity, feature_class, self.provenance))
        if len({row.entity.id for row in entities}) != len(entities):
            raise ExternalDataUnavailableError("Duplicate ENC feature identity")
        self.entities = tuple(sorted(entities, key=lambda row: row.entity.id))
