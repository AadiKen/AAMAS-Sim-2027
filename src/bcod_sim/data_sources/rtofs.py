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
        if depth_m < 0:
            raise ExternalDataCoverageError("Requested RTOFS depth must be nonnegative")
        # Select the nearest horizontal column, then interpolate its vertical levels.
        horizontal = min(self.samples, key=lambda x: distance_m(lat_deg, lon_deg, x["lat_deg"], x["lon_deg"]))
        column = [row for row in self.samples if row["lat_deg"] == horizontal["lat_deg"] and
                  row["lon_deg"] == horizontal["lon_deg"]]
        if distance_m(lat_deg, lon_deg, horizontal["lat_deg"], horizontal["lon_deg"]) > self.max_sample_radius_m:
            raise ExternalDataCoverageError("No RTOFS grid sample covers requested coordinate")
        levels = sorted(column, key=lambda row: float(row["depth_m"]))
        if depth_m < float(levels[0]["depth_m"]) or depth_m > float(levels[-1]["depth_m"]):
            raise ExternalDataCoverageError("Requested RTOFS depth lies outside the source column")
        lower = max((row for row in levels if float(row["depth_m"]) <= depth_m),
                    key=lambda row: float(row["depth_m"]))
        upper = min((row for row in levels if float(row["depth_m"]) >= depth_m),
                    key=lambda row: float(row["depth_m"]))
        low_depth, high_depth = float(lower["depth_m"]), float(upper["depth_m"])
        fraction = 0.0 if high_depth == low_depth else (depth_m-low_depth)/(high_depth-low_depth)
        interpolate = lambda key: float(lower[key]) + fraction*(float(upper[key])-float(lower[key]))
        return interpolate("v_north_mps"), interpolate("u_east_mps"), -interpolate("w_up_mps")
