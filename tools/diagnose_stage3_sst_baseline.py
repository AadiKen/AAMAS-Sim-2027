"""Read-only rolling-load diagnostic for a completed Stage 3 surge case."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter


NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def signal_stats(time: np.ndarray, value: np.ndarray) -> dict:
    duration = float(time[-1] - time[0])
    uniform_time = np.linspace(time[0], time[-1], 4097)
    uniform_value = np.interp(uniform_time, time, value)
    slope = float(np.polyfit(uniform_time, uniform_value, 1)[0])
    centered = uniform_value - np.polyval(np.polyfit(uniform_time, uniform_value, 1), uniform_time)
    frequencies = np.fft.rfftfreq(len(centered), duration / 4096)
    amplitudes = 2 * np.abs(np.fft.rfft(centered)) / len(centered)
    eligible = np.flatnonzero((frequencies >= 1 / duration) & (frequencies <= 5))
    peak = int(eligible[np.argmax(amplitudes[eligible])]) if len(eligible) else 0
    return {
        "mean": float(uniform_value.mean()),
        "std": float(uniform_value.std()),
        "slope_per_s": slope,
        "trend_over_window": slope * duration,
        "dominant_frequency_hz": float(frequencies[peak]) if peak else None,
        "dominant_period_s": float(1 / frequencies[peak]) if peak else None,
        "dominant_amplitude": float(amplitudes[peak]) if peak else None,
        "detrended_rms": float(centered.std()),
        "min": float(uniform_value.min()),
        "max": float(uniform_value.max()),
    }


def analyze(root: Path) -> dict:
    history = np.asarray(OpenFOAMAdapter.force_history(root), dtype=float)
    if history.ndim != 2 or history.shape[1] != 7:
        raise ValueError("a complete six-component force history is required")
    time, wrench = history[:, 0], history[:, 1:]
    blocks = []
    for start in (0, 2, 4, 6, 8):
        mask = (time >= start) & (time <= start + 2)
        channels = {name: signal_stats(time[mask], wrench[mask, index])
                    for index, name in enumerate(NAMES)}
        blocks.append({"start_s": start, "end_s": start + 2, "channels": channels})
    rolling = []
    for start in (0, 2, 4, 6):
        mask = (time >= start) & (time <= start + 4)
        channels = {name: signal_stats(time[mask], wrench[mask, index])
                    for index, name in enumerate(NAMES)}
        rolling.append({"start_s": start, "end_s": start + 4, "channels": channels})
    changes = {name: [float(abs(blocks[index]["channels"][name]["mean"]-
                              blocks[index-1]["channels"][name]["mean"]))
                      for index in range(1, len(blocks))] for name in NAMES}
    return {"samples": len(time), "time_range_s": [float(time[0]), float(time[-1])],
            "two_second_blocks": blocks, "four_second_rolling": rolling,
            "adjacent_two_second_mean_changes": changes}


def boundary_stats(root: Path, time_name: str) -> dict:
    """Sample cells touching each far-field patch without modifying CFD state."""
    mesh = root / "constant" / "polyMesh"
    owner_text = (mesh / "owner").read_text()
    owner_match = re.search(r"\n(\d+)\s*\((.*?)\)", owner_text, re.S)
    owner = np.fromstring(owner_match.group(2), sep=" ").astype(int)
    if len(owner) != int(owner_match.group(1)):
        raise ValueError("incomplete mesh owner list")
    boundary_text = (mesh / "boundary").read_text()
    patches = {match.group(1): (int(match.group(2)), int(match.group(3)))
               for match in re.finditer(r"\b(\w+)\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);",
                                        boundary_text, re.S)}

    def field(name: str, vector: bool = False) -> np.ndarray:
        data = (root / time_name / name).read_text()
        kind = "vector" if vector else "scalar"
        match = re.search(rf"internalField\s+nonuniform List<{kind}>\s*(\d+)\s*\((.*?)\)",
                          data, re.S)
        if vector:
            # Stop at the list's final standalone closing parenthesis.
            match = re.search(r"internalField\s+nonuniform List<vector>\s*(\d+)\s*\(\n(.*?)\n\)\s*;",
                              data, re.S)
        if not match:
            raise ValueError(f"nonuniform {name} field not found at {time_name}")
        values = np.fromstring(match.group(2).replace("(", " ").replace(")", " "), sep=" ")
        count = int(match.group(1))
        if vector:
            if len(values) != count * 3: raise ValueError(f"incomplete {name} field")
            return values.reshape((-1, 3))
        if len(values) != count: raise ValueError(f"incomplete {name} field")
        return values

    alpha, velocity = field("alpha.water"), field("U", vector=True)
    pressure, k, omega, nut = (field(name) for name in ("p_rgh", "k", "omega", "nut"))
    report = {}
    for name, (size, start) in patches.items():
        cells = owner[start:start + size]
        water = alpha[cells]
        speed_error = np.linalg.norm(velocity[cells] - (1., 0., 0.), axis=1)
        report[name] = {
            "faces": size,
            "alpha_min": float(water.min()), "alpha_max": float(water.max()),
            "mixed_alpha_fraction": float(np.mean((water > .01) & (water < .99))),
            "velocity_error_p95_mps": float(np.percentile(speed_error, 95)),
            "velocity_error_max_mps": float(speed_error.max()),
            "velocity_x_range_mps": [float(velocity[cells, 0].min()),
                                       float(velocity[cells, 0].max())],
            "reverse_flow_fraction": float(np.mean(velocity[cells, 0] < 0)),
            "wet_velocity_error_p95_mps": (float(np.percentile(speed_error[water > .9], 95))
                                              if np.any(water > .9) else None),
            "dry_velocity_error_p95_mps": (float(np.percentile(speed_error[water < .1], 95))
                                              if np.any(water < .1) else None),
            "p_rgh_range": [float(pressure[cells].min()), float(pressure[cells].max())],
            "wet_p_rgh_range": ([float(pressure[cells][water > .9].min()),
                                   float(pressure[cells][water > .9].max())]
                                  if np.any(water > .9) else None),
            "k_median": float(np.median(k[cells])), "k_p95": float(np.percentile(k[cells], 95)),
            "wet_k_median": (float(np.median(k[cells][water > .9]))
                              if np.any(water > .9) else None),
            "wet_k_p95": (float(np.percentile(k[cells][water > .9], 95))
                           if np.any(water > .9) else None),
            "omega_median": float(np.median(omega[cells])),
            "nut_median": float(np.median(nut[cells])),
            "wet_nut_median": (float(np.median(nut[cells][water > .9]))
                                if np.any(water > .9) else None),
            "wet_nut_p95": (float(np.percentile(nut[cells][water > .9], 95))
                             if np.any(water > .9) else None),
        }
    return report


def hull_pressure_partition(root: Path, time_name: str) -> dict:
    """Separate dry and wetted hull pressure contributions at a saved state."""
    mesh = root / "constant" / "polyMesh"
    boundary = (mesh / "boundary").read_text()
    patch = re.search(r"\bhull\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);", boundary, re.S)
    count, start = (int(value) for value in patch.groups())
    point_text = (mesh / "points").read_text()
    point_match = re.search(r"\n(\d+)\s*\(\n(.*?)\n\)", point_text, re.S)
    points = np.fromstring(point_match.group(2).replace("(", " ").replace(")", " "),
                           sep=" ").reshape((-1, 3))
    if len(points) != int(point_match.group(1)):
        raise ValueError("incomplete point list")
    face_text = (mesh / "faces").read_text()
    face_match = re.search(r"\n(\d+)\s*\(\n(.*?)\n\)", face_text, re.S)
    face_lines = face_match.group(2).splitlines()
    if len(face_lines) != int(face_match.group(1)):
        raise ValueError("incomplete face list")
    owner_text = (mesh / "owner").read_text()
    owner_match = re.search(r"\n(\d+)\s*\((.*?)\)", owner_text, re.S)
    owners = np.fromstring(owner_match.group(2), sep=" ").astype(int)
    alpha_text = (root / time_name / "alpha.water").read_text()
    alpha_match = re.search(r"internalField\s+nonuniform List<scalar>\s*(\d+)\s*\((.*?)\)",
                            alpha_text, re.S)
    alpha = np.fromstring(alpha_match.group(2), sep=" ")
    pressure_text = (root / time_name / "p").read_text().split("    hull\n", 1)[1]
    pressure_match = re.search(r"nonuniform List<scalar>\s*(\d+)\s*\((.*?)\)",
                               pressure_text, re.S)
    pressure = np.fromstring(pressure_match.group(2), sep=" ")
    if len(pressure) != count: raise ValueError("hull pressure and mesh sizes disagree")
    totals = {name: {"force": np.zeros(3), "moment": np.zeros(3), "area_m2": 0.,
                     "faces": 0} for name in ("wet", "dry", "interface")}
    origin = np.asarray((0., 0., -.10601493958702342))
    for index, line in enumerate(face_lines[start:start + count]):
        vertex_ids = np.fromstring(line.split("(", 1)[1].split(")", 1)[0],
                                   sep=" ").astype(int)
        vertices = points[vertex_ids]
        area_vector = .5 * np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)
        center = vertices.mean(axis=0)
        fraction = alpha[owners[start + index]]
        name = "wet" if fraction > .9 else "dry" if fraction < .1 else "interface"
        force = pressure[index] * area_vector
        item = totals[name]
        item["force"] += force
        item["moment"] += np.cross(center - origin, force)
        item["area_m2"] += np.linalg.norm(area_vector)
        item["faces"] += 1
    return {name: {"force": item["force"].tolist(),
                   "moment": item["moment"].tolist(),
                   "area_m2": item["area_m2"], "faces": item["faces"]}
            for name, item in totals.items()}


def side_surface_profile(root: Path, time_name: str, *, include_columns: bool = False) -> dict:
    """Estimate surface displacement from VOF on the two far-field side faces."""
    mesh = root / "constant" / "polyMesh"
    boundary = (mesh / "boundary").read_text()
    patch = re.search(r"\bsides\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);",
                      boundary, re.S)
    count, start = (int(value) for value in patch.groups())
    point_text = (mesh / "points").read_text()
    point_match = re.search(r"\n(\d+)\s*\(\n(.*?)\n\)", point_text, re.S)
    points = np.fromstring(point_match.group(2).replace("(", " ").replace(")", " "),
                           sep=" ").reshape((-1, 3))
    face_text = (mesh / "faces").read_text()
    face_match = re.search(r"\n(\d+)\s*\(\n(.*?)\n\)", face_text, re.S)
    face_lines = face_match.group(2).splitlines()
    owner_text = (mesh / "owner").read_text()
    owner_match = re.search(r"\n(\d+)\s*\((.*?)\)", owner_text, re.S)
    owners = np.fromstring(owner_match.group(2), sep=" ").astype(int)
    alpha_text = (root / time_name / "alpha.water").read_text()
    alpha_match = re.search(r"internalField\s+nonuniform List<scalar>\s*(\d+)\s*\((.*?)\)",
                            alpha_text, re.S)
    alpha = np.fromstring(alpha_match.group(2), sep=" ")
    columns: dict[tuple[str, int], float] = {}
    for index, line in enumerate(face_lines[start:start + count]):
        ids = np.fromstring(line.split("(", 1)[1].split(")", 1)[0], sep=" ").astype(int)
        vertices = points[ids]
        center = vertices.mean(axis=0)
        area = .5 * np.linalg.norm(np.cross(vertices,
            np.roll(vertices,-1,axis=0)).sum(axis=0))
        # Far-field side quads are 0.5 m wide in x; area/width is column dz.
        side = "port" if center[1] > 0 else "starboard"
        x_index = int(round((center[0] + 11.75) / .5))
        key = side, x_index
        columns[key] = columns.get(key, 0.) + alpha[owners[start + index]] * area / .5
    result = {}
    for side in ("port", "starboard"):
        profile = sorted((index, depth - 5.) for (name, index), depth in columns.items()
                         if name == side)
        if not profile:
            continue
        values = [eta for _, eta in profile]
        result[side] = {"rms_eta_m": float(np.sqrt(np.mean(np.square(values)))),
                        "min_eta_m": float(min(values)), "max_eta_m": float(max(values)),
                        "columns": len(values)}
        if include_columns:
            result[side]["eta_by_x_m"] = [((index * .5) - 11.75, eta)
                                           for index, eta in profile]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.case_root), indent=2))


if __name__ == "__main__":
    main()
