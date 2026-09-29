"""Section-constrained Surveyor envelope hypotheses, separate from frozen M1.

The source is a tessellation of selected STEP faces in imported millimetres.
This constructs diagnostic closed envelopes, not a claim that pocket topology is known.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import trimesh


ROOT = Path("docs/surveyor_cad_validation/geometry_recovery/section_reconstruction")
CENTRE_X_MM = -338.0
CENTRE_Y_MM = -190.0
TOP_CAP_Y_MM = -168.3
ORIGIN_LONGITUDINAL_MM = 915.0
STATIONS_MM = np.arange(330.0, 1741.0, 5.0)
ANGLES = np.linspace(0, 2*np.pi, 512, endpoint=False)


def _source_points_at_station(mesh: trimesh.Trimesh, station: float) -> np.ndarray:
    lines, _ = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, station],
                                                 return_faces=True)
    if not len(lines):
        return np.empty((0, 2))
    lengths = np.linalg.norm(lines[:, 1, :2]-lines[:, 0, :2], axis=1)
    pieces = []
    for line, length in zip(lines, lengths):
        count = max(2, int(np.ceil(length/.5))+1)
        weight = np.linspace(0, 1, count)
        pieces.append(line[0, :2][None, :]*(1-weight[:, None])+
                      line[1, :2][None, :]*weight[:, None])
    return np.vstack(pieces)


def _observed_radial_field(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    radii = np.full((len(STATIONS_MM), len(ANGLES)), np.nan)
    observed = np.zeros_like(radii, dtype=bool)
    audits = []
    for station_index, station in enumerate(STATIONS_MM):
        points = _source_points_at_station(mesh, station)
        if not len(points):
            audits.append({"source_Z_mm": float(station), "sample_count": 0,
                           "observed_angular_fraction": 0.0})
            continue
        centred = points-np.array([CENTRE_X_MM, CENTRE_Y_MM])
        theta = np.mod(np.arctan2(centred[:, 1], centred[:, 0]), 2*np.pi)
        radial = np.linalg.norm(centred, axis=1)
        order = np.argsort(theta)
        theta, radial = theta[order], radial[order]
        # Source shell is a single radial intersection at almost every angle.
        # Ignore local duplicate angles from adjacent CAD-face tessellations.
        keep = np.r_[True, np.diff(theta)>1e-6]
        theta, radial = theta[keep], radial[keep]
        augmented_theta = np.r_[theta[-1]-2*np.pi, theta, theta[0]+2*np.pi]
        augmented_radius = np.r_[radial[-1], radial, radial[0]]
        interpolated = np.interp(ANGLES, augmented_theta, augmented_radius)
        distances = np.min(np.abs(np.mod(ANGLES[:, None]-theta[None, :]+np.pi, 2*np.pi)-np.pi), axis=1)
        seen = distances < .0175
        radii[station_index, seen] = interpolated[seen]
        observed[station_index] = seen
        audits.append({"source_Z_mm": float(station), "sample_count": len(points),
                       "observed_angular_fraction": float(np.mean(seen)),
                       "section_bounds_mm": [points.min(axis=0).tolist(),
                                             points.max(axis=0).tolist()]})
    return radii, observed, audits


def _fill_missing(radii: np.ndarray, observed: np.ndarray,
                  source_mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    filled = radii.copy()
    top = np.sin(ANGLES) > 0
    # A flat section lid joins the surviving upper edges. It is explicitly
    # reconstructed and excluded from source-surface deviation claims.
    for i in range(len(STATIONS_MM)):
        top_y = TOP_CAP_Y_MM
        missing_top = ~observed[i] & top
        safe = np.maximum(np.sin(ANGLES[missing_top]), .001)
        cap_r = (top_y-CENTRE_Y_MM)/safe
        # Only use the top cap between original port and starboard top edges.
        section_points = _source_points_at_station(source_mesh, STATIONS_MM[i])
        if len(section_points):
            xlo, xhi = section_points[:, 0].min(), section_points[:, 0].max()
            cap_x = CENTRE_X_MM+cap_r*np.cos(ANGLES[missing_top])
            valid = (cap_x >= xlo-1) & (cap_x <= xhi+1) & (cap_r > 0) & (cap_r < 500)
            indices = np.flatnonzero(missing_top)[valid]
            filled[i, indices] = cap_r[valid]
    # Each unresolved angular ray borrows its shape from intact neighbouring
    # longitudinal sections. Missing end rays fall back to the nearest section.
    for angle_index in range(len(ANGLES)):
        finite = np.flatnonzero(np.isfinite(filled[:, angle_index]))
        if len(finite):
            missing = ~np.isfinite(filled[:, angle_index])
            filled[missing, angle_index] = np.interp(
                STATIONS_MM[missing], STATIONS_MM[finite], filled[finite, angle_index])
    # Rare angles never seen at any station are angularly interpolated.
    for i in range(len(STATIONS_MM)):
        finite = np.flatnonzero(np.isfinite(filled[i]))
        if len(finite) < 3:
            raise ValueError(f"station {STATIONS_MM[i]} lacks constrained section")
        unknown = ~np.isfinite(filled[i])
        filled[i, unknown] = np.interp(ANGLES[unknown], ANGLES[finite], filled[i, finite],
                                       period=2*np.pi)
    # Fair only unsupported lower sectors. Original angular samples and the
    # explicitly reconstructed planar lid remain fixed Dirichlet constraints.
    fair_mask = (~observed) & (np.sin(ANGLES)[None, :] < 0)
    for _ in range(12):
        smoothed = gaussian_filter(filled, sigma=(2.0, 3.0), mode=("nearest", "wrap"))
        filled[fair_mask] = smoothed[fair_mask]
    reconstructed = ~observed
    return filled, reconstructed


def _hypothesis_radii(base: np.ndarray, observed: np.ndarray, name: str) -> np.ndarray:
    if name == "smooth":
        return base.copy()
    result = base.copy()
    lower_missing = (~observed) & (np.sin(ANGLES)[None, :] < -.12)
    # The three hypotheses share every original source-surface angular sample.
    # The 6 mm maximum variation acts only in unsupported lower pocket sectors.
    z_taper = np.maximum(0, 1-np.abs(STATIONS_MM-805)/360)
    theta_taper = np.maximum(0, -np.sin(ANGLES))
    sign = -1 if name == "inset" else 1
    adjustment = sign*6.0*z_taper[:, None]*theta_taper[None, :]
    result[lower_missing] += adjustment[lower_missing]
    return result


def _loft_port(radii: np.ndarray) -> trimesh.Trimesh:
    station_count, angular_count = radii.shape
    xx = CENTRE_X_MM+radii*np.cos(ANGLES)[None, :]
    yy = CENTRE_Y_MM+radii*np.sin(ANGLES)[None, :]
    zz = np.broadcast_to(STATIONS_MM[:, None], radii.shape)
    # Imported source xyz (mm) -> canonical body FRD (m).
    vertices = np.column_stack(((zz.ravel()-ORIGIN_LONGITUDINAL_MM)/1000,
                                -xx.ravel()/1000, -yy.ravel()/1000))
    faces = []
    for i in range(station_count-1):
        a = i*angular_count+np.arange(angular_count)
        b = i*angular_count+(np.arange(angular_count)+1)%angular_count
        c, d = a+angular_count, b+angular_count
        faces.extend(np.column_stack((a, b, c)).tolist())
        faces.extend(np.column_stack((b, d, c)).tolist())
    for end_index in (0, station_count-1):
        centre = vertices[end_index*angular_count:(end_index+1)*angular_count].mean(axis=0)
        centre_index = len(vertices)
        vertices = np.vstack((vertices, centre))
        ring = end_index*angular_count+np.arange(angular_count)
        next_ring = end_index*angular_count+(np.arange(angular_count)+1)%angular_count
        cap = np.column_stack((ring, next_ring, np.full(angular_count, centre_index)))
        if end_index == 0:
            cap = cap[:, ::-1]
        faces.extend(cap.tolist())
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def run(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(source, process=True)
    components = sorted(mesh.split(only_watertight=False), key=lambda part: part.area, reverse=True)
    port_candidates = [part for part in components if part.bounds[1, 0] < 0]
    if not port_candidates:
        raise ValueError("no negative-source-X pontoon candidate")
    port_source = port_candidates[0]
    radii, observed, audit = _observed_radial_field(port_source)
    base, reconstructed = _fill_missing(radii, observed, port_source)
    results = {"schema": "surveyor-constrained-section-hypotheses-v1",
               "input_source_tessellation_sha256": sha256(source.read_bytes()).hexdigest(),
               "source_units": "mm", "station_step_mm": 5,
               "angular_count": len(ANGLES), "port_source_component_face_count": len(port_source.faces),
               "section_centre_source_X_mm": CENTRE_X_MM,
               "section_centre_source_Y_mm": CENTRE_Y_MM,
               "top_cap_source_Y_mm": TOP_CAP_Y_MM,
               "reference_origin_source_Z_mm": ORIGIN_LONGITUDINAL_MM,
               "section_audit": audit, "hypotheses": {}}
    np.savez_compressed(output / "section_constraints.npz", stations_mm=STATIONS_MM,
                        angles_rad=ANGLES, source_radii_mm=radii, source_observed=observed,
                        fair_radii_mm=base, reconstructed_mask=reconstructed,
                        centre_x_mm=CENTRE_X_MM, centre_y_mm=CENTRE_Y_MM,
                        longitudinal_origin_mm=ORIGIN_LONGITUDINAL_MM,
                        top_cap_y_mm=TOP_CAP_Y_MM)
    for name in ("smooth", "inset", "outset"):
        variant = _hypothesis_radii(base, observed, name)
        port = _loft_port(variant)
        starboard = port.copy(); starboard.vertices[:, 1] *= -1; starboard.invert()
        combined = trimesh.util.concatenate((port, starboard))
        combined.fix_normals()
        filename = output / f"surveyor_{name}_FRD_m.stl"
        combined.export(filename)
        results["hypotheses"][name] = {
            "file": filename.name, "sha256": sha256(filename.read_bytes()).hexdigest(),
            "watertight": bool(combined.is_watertight),
            "winding_consistent": bool(combined.is_winding_consistent),
            "positive_volume": bool(port.volume > 0 and starboard.volume > 0),
            "volume_m3": float(combined.volume), "port_volume_m3": float(port.volume),
            "surface_area_m2": float(combined.area),
            "center_of_volume_frd_m": combined.center_mass.tolist(),
            "bounds_frd_m": combined.bounds.tolist(),
            "face_count": len(combined.faces),
            "mirrored_from_port": True,
        }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, station in enumerate((450, 700, 1050)):
        i = int(np.argmin(np.abs(STATIONS_MM-station)))
        for row, name in enumerate(("smooth", "inset")):
            rr = _hypothesis_radii(base, observed, name)[i]
            ax = axes[row, col]
            ax.plot(CENTRE_X_MM+rr*np.cos(ANGLES), CENTRE_Y_MM+rr*np.sin(ANGLES), lw=1)
            theta = ANGLES[observed[i]]
            ax.scatter(CENTRE_X_MM+radii[i, observed[i]]*np.cos(theta),
                       CENTRE_Y_MM+radii[i, observed[i]]*np.sin(theta), s=2)
            ax.set(title=f"{name}, source Z={station} mm", aspect="equal", xlim=(-420, -260),
                   ylim=(-270, -150), xlabel="source X (mm)", ylabel="source Y (mm)")
    fig.tight_layout(); fig.savefig(output / "section_hypotheses.png", dpi=150); plt.close(fig)
    (output / "reconstruction_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/reconstruct_surveyor_envelopes.py SOURCE_STL OUTPUT_DIR")
    result = run(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result["hypotheses"], indent=2))
