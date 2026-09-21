"""RTOFS depth-specific 3D currents in canonical NED."""

from datetime import datetime

from bcod_sim.core.errors import ExternalDataCoverageError, ExternalDataUnavailableError
from bcod_sim.data_sources.base import GeoCoverage, distance_m, number, records, verify_json_payload


class RTOFS3D:
    def __init__(self, payload: bytes, checksum: str, *, request_coverage: GeoCoverage,
                 build_time: datetime) -> None:
        verified = verify_json_payload(payload, checksum, source="NOAA RTOFS",
            required_units={"horizontal_velocity": "m/s", "vertical_velocity": "m/s", "depth": "m"},
            required_frame="geographic_ENU", request_coverage=request_coverage, build_time=build_time)
        if verified.document["metadata"].get("variable") == "2ds":
            raise ExternalDataUnavailableError("Depth-averaged RTOFS 2ds cannot substitute for 3D current")
        self.provenance = verified.provenance
        self.max_sample_radius_m = number(verified.document["metadata"], "max_sample_radius_m",
                                          strictly_positive=True)
        self.samples = records(verified.document, "samples", {"lat_deg", "lon_deg", "depth_m",
            "u_east_mps", "v_north_mps", "w_up_mps"})
        for row in self.samples:
            self.provenance.coverage.require(number(row, "lat_deg"), number(row, "lon_deg"))
            number(row, "depth_m", minimum=0)
            for field in ("u_east_mps", "v_north_mps", "w_up_mps"): number(row, field)

    def current_ned_mps(self, lat_deg: float, lon_deg: float, depth_m: float) -> tuple[float, float, float]:
        self.provenance.coverage.require(lat_deg, lon_deg)
        candidates = [row for row in self.samples if float(row["depth_m"]) == depth_m]
        if not candidates:
            raise ExternalDataCoverageError("Requested RTOFS depth level is unavailable")
        row = min(candidates, key=lambda x: distance_m(lat_deg, lon_deg, x["lat_deg"], x["lon_deg"]))
        if distance_m(lat_deg, lon_deg, row["lat_deg"], row["lon_deg"]) > self.max_sample_radius_m:
            raise ExternalDataCoverageError("No RTOFS grid sample covers requested coordinate")
        return float(row["v_north_mps"]), float(row["u_east_mps"]), -float(row["w_up_mps"])
