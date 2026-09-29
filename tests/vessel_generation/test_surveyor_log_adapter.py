import csv
import io
import json
import zipfile

import pytest

from bcod_sim.surveyor_validation.logs import commands, import_archive


def test_command_mapping_and_clipping():
    assert commands(20, 0) == pytest.approx((.2, .2))
    assert commands(0, 30) == pytest.approx((.3, -.3))
    assert commands(0, -30) == pytest.approx((-.3, .3))
    assert commands(80, 50) == pytest.approx((1., .3))


def test_importer_flags_invalid_fix_and_keeps_raw(tmp_path):
    header = ["Day", "Time", "Latitude", "Longitude", "Heading (degrees Magnetic)",
              "Yaw rate [degrees/s]", "Thrust (% Thrust)", "Thrust difference (% Thrust)",
              "Acceleration x, forward (G)", "Acceleration y, starboard (G)",
              "Acceleration z, down (G)", "Control Mode"]
    archive = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for vessel in ("USV1", "USV2"):
            stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=header); writer.writeheader()
            for time, lon, heading in ((140000, -80.1, 359), (140001, 0., 0), (140004, -80.10001, 1)):
                writer.writerow({"Day": 20250226, "Time": time, "Latitude": 25.7,
                    "Longitude": lon, "Heading (degrees Magnetic)": heading,
                    "Yaw rate [degrees/s]": 5, "Thrust (% Thrust)": 10,
                    "Thrust difference (% Thrust)": -20,
                    "Acceleration x, forward (G)": 0,
                    "Acceleration y, starboard (G)": 0,
                    "Acceleration z, down (G)": -1, "Control Mode": "Waypoint"})
            bundle.writestr(f"task1/{vessel}_test/state_data.csv", stream.getvalue())
    quality = import_archive(archive, tmp_path / "out")
    assert quality["vessels"]["USV1"]["invalid_gps"] == 1
    assert quality["vessels"]["USV1"]["invalid_heading"] == 1
    assert (tmp_path / "out/raw/USV1_state_data.csv").exists()
    rows = list(csv.DictReader((tmp_path / "out/USV1_normalized.csv").open()))
    assert json.loads(rows[1]["quality_flags"]) == ["INVALID_GPS", "MISSING_OR_ZERO_HEADING"]
    assert float(rows[2]["heading"]) == pytest.approx(2*3.141592653589793+3.141592653589793/180)
