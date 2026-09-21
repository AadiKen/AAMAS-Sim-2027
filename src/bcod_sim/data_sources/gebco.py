"""GEBCO elevation-positive-up to canonical NED bottom conversion."""

from datetime import datetime
from typing import Mapping

from bcod_sim.core.errors import ExternalDataUnavailableError
from bcod_sim.data_sources.base import GeoCoverage, nearest, number, records, verify_json_payload


class GEBCO:
    def __init__(self, payload: bytes, checksum: str, *, request_coverage: GeoCoverage,
                 build_time: datetime) -> None:
        verified = verify_json_payload(payload, checksum, source="GEBCO",
            required_units={"elevation": "m"}, required_frame="WGS84",
            request_coverage=request_coverage, build_time=build_time)
        if verified.provenance.datum is None or verified.document["metadata"].get("vertical_convention") != "elevation_positive_up":
            raise ExternalDataUnavailableError("GEBCO vertical datum/convention is ambiguous")
        self.provenance = verified.provenance
        self.max_sample_radius_m = number(verified.document["metadata"], "max_sample_radius_m",
                                          strictly_positive=True)
        self.samples = records(verified.document, "samples", {"lat_deg", "lon_deg", "elevation_m"})
        for row in self.samples:
            self.provenance.coverage.require(number(row, "lat_deg"), number(row, "lon_deg"))
            number(row, "elevation_m")

    def bottom_ned_z_m(self, lat_deg: float, lon_deg: float) -> float:
        self.provenance.coverage.require(lat_deg, lon_deg)
        return -float(nearest(self.samples, lat_deg, lon_deg,
                              max_radius_m=self.max_sample_radius_m)["elevation_m"])
