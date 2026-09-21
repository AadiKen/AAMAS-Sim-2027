"""Shared fail-closed payload, coverage, time, and provenance contracts."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

from bcod_sim.core.errors import ExternalDataCoverageError, ExternalDataUnavailableError


def utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ExternalDataUnavailableError("External valid time must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ExternalDataUnavailableError("External valid time requires timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class GeoCoverage:
    min_lat_deg: float
    max_lat_deg: float
    min_lon_deg: float
    max_lon_deg: float

    def __post_init__(self) -> None:
        values = (self.min_lat_deg, self.max_lat_deg, self.min_lon_deg, self.max_lon_deg)
        if (not all(math.isfinite(x) for x in values) or not -90 <= self.min_lat_deg < self.max_lat_deg <= 90 or
                not -180 <= self.min_lon_deg < self.max_lon_deg <= 180):
            raise ExternalDataUnavailableError("Invalid geographic coverage")

    def require(self, lat_deg: float, lon_deg: float) -> None:
        if not (self.min_lat_deg <= lat_deg <= self.max_lat_deg and
                self.min_lon_deg <= lon_deg <= self.max_lon_deg):
            raise ExternalDataCoverageError("Requested coordinate is outside payload coverage")

    def contains(self, other: "GeoCoverage") -> bool:
        return (self.min_lat_deg <= other.min_lat_deg <= other.max_lat_deg <= self.max_lat_deg and
                self.min_lon_deg <= other.min_lon_deg <= other.max_lon_deg <= self.max_lon_deg)


@dataclass(frozen=True)
class Provenance:
    source: str
    product: str
    version: str
    valid_time: str
    units: Mapping[str, str]
    frame: str
    datum: str | None
    coverage: GeoCoverage
    payload_sha256: str


@dataclass(frozen=True)
class VerifiedPayload:
    document: Mapping
    provenance: Provenance


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def verify_json_payload(payload: bytes, expected_sha256: str, *, source: str,
                        required_units: Mapping[str, str], required_frame: str,
                        request_coverage: GeoCoverage, build_time: datetime) -> VerifiedPayload:
    actual = hashlib.sha256(payload).hexdigest()
    if len(expected_sha256) != 64 or actual != expected_sha256.lower():
        raise ExternalDataUnavailableError("External payload checksum mismatch")
    try:
        document = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON value: {value}")))
        meta = document["metadata"]
        coverage = GeoCoverage(**meta["coverage"])
        valid_from, valid_until = utc(meta["valid_from"]), utc(meta["valid_until"])
        valid_time = utc(meta["valid_time"])
        units = dict(meta["units"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExternalDataUnavailableError("Malformed external payload metadata") from exc
    if meta.get("source") != source or not meta.get("product") or not meta.get("version"):
        raise ExternalDataUnavailableError("External payload source/product/version mismatch")
    when = build_time.astimezone(timezone.utc) if build_time.tzinfo else None
    if when is None or not valid_from <= valid_time <= valid_until or not valid_from <= when <= valid_until:
        raise ExternalDataUnavailableError("External payload does not satisfy current build time")
    if not coverage.contains(request_coverage):
        raise ExternalDataCoverageError("Requested world extent exceeds payload coverage")
    if meta.get("frame") != required_frame or any(units.get(k) != v for k, v in required_units.items()):
        raise ExternalDataUnavailableError("External payload units or frame mismatch")
    datum = meta.get("datum")
    provenance = Provenance(source, meta["product"], meta["version"], valid_time.isoformat(),
                            MappingProxyType(units), required_frame, datum, coverage, actual)
    return VerifiedPayload(_freeze(document), provenance)


def records(document: Mapping, key: str, required: set[str], optional: set[str] | None = None) -> tuple[Mapping, ...]:
    if set(document) != {"metadata", key}:
        raise ExternalDataUnavailableError(f"Unexpected external payload sections for {key}")
    rows = document.get(key)
    allowed = required | (optional or set())
    if not isinstance(rows, tuple) or not rows:
        raise ExternalDataUnavailableError(f"External payload requires nonempty {key}")
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row) or set(row) - allowed:
            raise ExternalDataUnavailableError(f"Malformed external record in {key}")
    return rows


def number(row: Mapping, key: str, *, minimum: float | None = None,
           strictly_positive: bool = False) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExternalDataUnavailableError(f"External field {key} must be finite numeric")
    value = float(value)
    if (minimum is not None and value < minimum) or (strictly_positive and value <= 0):
        raise ExternalDataUnavailableError(f"External field {key} is outside its physical domain")
    return value


def distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat1, lat2 = math.radians(lat_a), math.radians(lat_b)
    dlat, dlon = lat2-lat1, math.radians(lon_b-lon_a)
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371008.8 * 2 * math.asin(min(1.0, math.sqrt(h)))


def nearest(rows: tuple[Mapping, ...], lat_deg: float, lon_deg: float, *, max_radius_m: float | None = None):
    if not rows:
        raise ExternalDataUnavailableError("External payload contains no samples")
    row = min(rows, key=lambda x: distance_m(lat_deg, lon_deg, x["lat_deg"], x["lon_deg"]))
    distance = distance_m(lat_deg, lon_deg, row["lat_deg"], row["lon_deg"])
    if max_radius_m is not None and distance > max_radius_m:
        raise ExternalDataCoverageError("No source station within declared radius")
    return row
