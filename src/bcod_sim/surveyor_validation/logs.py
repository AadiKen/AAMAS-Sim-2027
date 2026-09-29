"""Import the two Surveyor state logs without interpolating measurements."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

EARTH_RADIUS_M = 6371008.8
G_MPS2 = 9.80665


def commands(thrust_percent: float, difference_percent: float) -> tuple[float, float]:
    """Return port and starboard commands normalized to [-1, 1]."""
    if not math.isfinite(thrust_percent) or not math.isfinite(difference_percent):
        raise ValueError("commands must be finite")
    clip = lambda value: min(1., max(-1., value/100.))
    return clip(thrust_percent + difference_percent), clip(thrust_percent - difference_percent)


def _timestamp(row: dict) -> float:
    day = str(row["Day"]).split(".")[0]
    stamp = str(row["Time"]).split(".")[0].zfill(6)
    return datetime.strptime(day+stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()


def _finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _gps(row: dict) -> tuple[float | None, float | None]:
    lat, lon = _finite(row.get("Latitude")), _finite(row.get("Longitude"))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None
    if abs(lat) < 1 or abs(lon) < 1:
        return None, None
    return lat, lon


def _relative_xy(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = origin
    x_east = EARTH_RADIUS_M*math.cos(math.radians(lat0))*math.radians(lon-lon0)
    y_north = EARTH_RADIUS_M*math.radians(lat-lat0)
    return x_east, y_north


def _read_archive(archive: Path) -> tuple[dict[str, bytes], dict[str, list[dict]]]:
    raw, rows = {}, {}
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not name.endswith("state_data.csv"):
                continue
            vessel = "USV1" if "/USV1_" in name else "USV2" if "/USV2_" in name else None
            if vessel is None or vessel in raw:
                raise ValueError("archive must contain one state CSV per known Surveyor")
            content = bundle.read(name)
            raw[vessel] = content
            rows[vessel] = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if set(rows) != {"USV1", "USV2"}:
        raise ValueError("archive does not contain both Surveyor state CSVs")
    return raw, rows


def _label(row: dict, previous: dict | None) -> str:
    t, d = row["thrust"], row["thrust_difference"]
    if abs(t) < .02 and abs(d) < .02:
        return "LOW_THRUST"
    if abs(d) > .15 and abs(d) > abs(t):
        return "LEFT_TURN" if d > 0 else "RIGHT_TURN"
    if abs(d) > .15:
        return "MIXED"
    if previous is not None and abs(t-previous["thrust"]) > .04:
        return "ACCELERATION" if abs(t) > abs(previous["thrust"]) else "DECELERATION"
    return "STRAIGHT"


def import_archive(archive: str | Path, output: str | Path) -> dict:
    archive, root = Path(archive), Path(output)
    raw, source_rows = _read_archive(archive)
    valid_positions = [_gps(row) for rows in source_rows.values() for row in rows]
    valid_positions = [item for item in valid_positions if item[0] is not None]
    if not valid_positions:
        raise ValueError("no valid GPS fixes")
    origin = (statistics.median(p[0] for p in valid_positions),
              statistics.median(p[1] for p in valid_positions))
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "raw"; raw_dir.mkdir(exist_ok=True)
    normalized = {}
    quality = {"archive_sha256": sha256(archive.read_bytes()).hexdigest(),
               "local_frame": "ENU; x east, y north", "gps_origin_deg": origin,
               "timestamp_reference": "Day/Time treated as UTC for relative alignment; source timezone unverified",
               "heading_reference": "magnetic north; declination unavailable",
               "acceleration_note": "IMU values are specific force in body FRD, gravity not removed",
               "vessels": {}}
    for vessel, rows in source_rows.items():
        (raw_dir / f"{vessel}_state_data.csv").write_bytes(raw[vessel])
        ordered = sorted(rows, key=_timestamp)
        output_rows = []
        seen = set(); previous_heading = None; previous = None
        duplicates = 0; invalid_gps = 0; invalid_heading = 0
        for record in ordered:
            stamp = _timestamp(record)
            if stamp in seen:
                duplicates += 1
                continue
            seen.add(stamp)
            flags = []
            lat, lon = _gps(record)
            if lat is None:
                x = y = None; invalid_gps += 1; flags.append("INVALID_GPS")
            else:
                x, y = _relative_xy(lat, lon, origin)
            heading_deg = _finite(record.get("Heading (degrees Magnetic)"))
            heading_valid = heading_deg is not None and 0 < heading_deg <= 360
            if not heading_valid:
                invalid_heading += 1; flags.append("MISSING_OR_ZERO_HEADING")
                heading = None
            else:
                angle = math.radians(heading_deg)
                if previous_heading is not None:
                    angle += round((previous_heading-angle)/(2*math.pi))*2*math.pi
                heading = angle; previous_heading = angle
            yaw_deg = _finite(record.get("Yaw rate [degrees/s]"))
            t = _finite(record.get("Thrust (% Thrust)"))
            d = _finite(record.get("Thrust difference (% Thrust)"))
            if t is None or d is None:
                flags.append("INVALID_COMMAND"); t = t or 0.; d = d or 0.
            port, starboard = commands(t, d)
            if abs(t+d)>100 or abs(t-d)>100:
                flags.append("COMMAND_CLIPPED")
            accelerations = [_finite(record.get(f"Acceleration {axis}, {direction} (G)"))
                             for axis, direction in (("x", "forward"), ("y", "starboard"), ("z", "down"))]
            item = {"timestamp": stamp, "vessel_id": vessel,
                    "x_local": x, "y_local": y, "gps_valid": lat is not None,
                    "heading": heading, "heading_valid": heading_valid,
                    "yaw_rate": None if yaw_deg is None else math.radians(yaw_deg),
                    "ax": None if accelerations[0] is None else accelerations[0]*G_MPS2,
                    "ay": None if accelerations[1] is None else accelerations[1]*G_MPS2,
                    "az": None if accelerations[2] is None else accelerations[2]*G_MPS2,
                    "thrust": t/100., "thrust_difference": d/100.,
                    "port_command": port, "starboard_command": starboard,
                    "inter_vessel_distance": None, "quality_flags": flags,
                    "other_along_m": None, "other_cross_m": None,
                    "control_mode": record.get("Control Mode", "")}
            item["segment_label"] = _label(item, previous)
            if previous is not None and stamp-previous["timestamp"] > 3:
                item["quality_flags"].append("TIME_GAP_GT_3S")
            output_rows.append(item); previous = item
        normalized[vessel] = output_rows
        intervals = [b["timestamp"]-a["timestamp"] for a,b in zip(output_rows,output_rows[1:])]
        quality["vessels"][vessel] = {"raw_rows": len(rows), "normalized_rows": len(output_rows),
            "duplicate_timestamps": duplicates, "invalid_gps": invalid_gps,
            "invalid_heading": invalid_heading,
            "duration_s": output_rows[-1]["timestamp"]-output_rows[0]["timestamp"],
            "sample_interval_s": {"min": min(intervals), "median": statistics.median(intervals),
                                  "max": max(intervals)},
            "command_range": {"thrust": [min(r["thrust"] for r in output_rows), max(r["thrust"] for r in output_rows)],
                              "difference": [min(r["thrust_difference"] for r in output_rows), max(r["thrust_difference"] for r in output_rows)]}}
    for vessel in normalized:
        other = normalized["USV1" if vessel == "USV2" else "USV2"]
        for row in normalized[vessel]:
            if not row["gps_valid"]:
                continue
            candidate = min(other, key=lambda item: abs(item["timestamp"]-row["timestamp"]))
            if abs(candidate["timestamp"]-row["timestamp"]) > 1 or not candidate["gps_valid"]:
                row["quality_flags"].append("NO_SIMULTANEOUS_OTHER_GPS")
                continue
            row["inter_vessel_distance"] = math.hypot(row["x_local"]-candidate["x_local"],
                                                       row["y_local"]-candidate["y_local"])
            if row["heading_valid"]:
                dx, dy = candidate["x_local"]-row["x_local"], candidate["y_local"]-row["y_local"]
                psi = row["heading"]
                row["other_along_m"] = dx*math.sin(psi)+dy*math.cos(psi)
                row["other_cross_m"] = dx*math.cos(psi)-dy*math.sin(psi)
            along, cross = row["other_along_m"], row["other_cross_m"]
            if row["inter_vessel_distance"] >= 9.0 or (
                along is not None and row["inter_vessel_distance"] >= 5.4 and abs(cross) >= 3.0):
                row["quality_flags"].append("LOW_INTERACTION_CANDIDATE")
            elif along is not None and along > 1.8 and abs(cross) < 2.5:
                row["quality_flags"].append("CLOSE_FOLLOWING")
            else:
                row["quality_flags"].append("NEAR_OTHER_VESSEL")
    for vessel, rows in normalized.items():
        fields = list(rows[0])
        with (root / f"{vessel}_normalized.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row in rows:
                writer.writerow({**row, "quality_flags": json.dumps(row["quality_flags"])})
        distances = [r["inter_vessel_distance"] for r in rows if r["inter_vessel_distance"] is not None]
        quality["vessels"][vessel]["distance_m"] = ({"min": min(distances),
            "median": statistics.median(distances), "max": max(distances)} if distances else None)
        quality["vessels"][vessel]["low_interaction_candidate_rows"] = sum(
            "LOW_INTERACTION_CANDIDATE" in r["quality_flags"] for r in rows)
        segments = []
        start = 0
        for index in range(1, len(rows)+1):
            boundary = (index == len(rows) or rows[index]["segment_label"] != rows[start]["segment_label"] or
                        rows[index]["timestamp"]-rows[index-1]["timestamp"] > 3)
            if not boundary:
                continue
            block = rows[start:index]
            distances = [r["inter_vessel_distance"] for r in block if r["inter_vessel_distance"] is not None]
            segments.append({"vessel_id": vessel, "segment_id": f"{vessel}_{len(segments):03d}",
                "label": block[0]["segment_label"], "start_timestamp": block[0]["timestamp"],
                "end_timestamp": block[-1]["timestamp"], "sample_count": len(block),
                "duration_s": block[-1]["timestamp"]-block[0]["timestamp"],
                "median_distance_m": statistics.median(distances) if distances else None,
                "low_interaction_samples": sum("LOW_INTERACTION_CANDIDATE" in r["quality_flags"] for r in block)})
            start = index
        quality["vessels"][vessel]["segment_count"] = len(segments)
        with (root / f"{vessel}_segments.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(segments[0])); writer.writeheader(); writer.writerows(segments)
    (root / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    return quality
