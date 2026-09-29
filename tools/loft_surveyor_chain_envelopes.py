"""Loft closed section chains from the intact mixed-group Surveyor STEP faces.

Original section paths are traversed in their measured order. Only gaps between
open chain endpoints and the upper lid are reconstructed. The three pocket
hypotheses change the bridge between inner ends of the two-chain region.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Polygon
import trimesh


STATIONS_MM = np.arange(50.0, 1751.0, 5.0)
CONTOUR_POINTS = 512
LONGITUDINAL_ORIGIN_MM = 915.0


def _chains(mesh: trimesh.Trimesh, z: float) -> list[np.ndarray]:
    segments, _ = trimesh.intersections.mesh_plane(
        mesh, [0, 0, 1], [0, 0, z], return_faces=True)
    graph = nx.Graph()
    for first, second in segments[:, :, :2]:
        graph.add_edge(tuple(np.round(first, 4)), tuple(np.round(second, 4)))
    paths = []
    for vertices in nx.connected_components(graph):
        subgraph = graph.subgraph(vertices)
        ends = [vertex for vertex, degree in subgraph.degree if degree == 1]
        if len(ends) != 2 or max(dict(subgraph.degree).values()) > 2:
            raise ValueError(f"station {z}: section chain has branching or closed overlap")
        path = nx.shortest_path(subgraph, ends[0], ends[1])
        paths.append(np.asarray(path, dtype=float))
    if len(paths) not in (1, 2):
        raise ValueError(f"station {z}: expected one or two source chains, got {len(paths)}")
    return paths


def _top_endpoint(chain: np.ndarray) -> int:
    return 0 if chain[0, 1] > chain[-1, 1] else -1


def _close_section(chains: list[np.ndarray], variant: str, z: float) -> tuple[np.ndarray, dict]:
    if len(chains) == 1:
        chain = chains[0]
        if chain[0, 0] > chain[-1, 0]:
            chain = chain[::-1]
        path = chain
        bridge = np.empty((0, 2))
    else:
        left, right = sorted(chains, key=lambda chain: chain[:, 0].mean())
        if _top_endpoint(left) == -1:
            left = left[::-1]
        if _top_endpoint(right) == 0:
            right = right[::-1]
        if not (left[0, 0] < right[-1, 0] and left[-1, 0] < right[0, 0]):
            raise ValueError(f"station {z}: two source chains cannot be ordered")
        weight = np.linspace(0, 1, 65)
        bridge = left[-1][None, :]*(1-weight[:, None])+right[0][None, :]*weight[:, None]
        # Only the measured midbody pocket break receives an ensemble bulge.
        if 1000 <= z <= 1130:
            bulge = {"smooth": 0., "inset": 15., "outset": -15.}[variant]
            bridge[:, 1] += bulge*np.sin(np.pi*weight)
        path = np.vstack((left, bridge[1:-1], right))
    # Measured top edge heights differ by a few millimetres; cap directly
    # between them without extending or reshaping surviving side surfaces.
    polygon = Polygon(path)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"station {z}: reconstructed section self-intersects")
    return path, {"source_chain_count": len(chains),
                  "source_vertex_count": sum(len(chain) for chain in chains),
                  "bridge_length_mm": float(np.linalg.norm(path[0]-path[-1])) if len(chains)==1
                    else float(np.linalg.norm(bridge[0]-bridge[-1])),
                  "bridge_area_proxy_mm2": float(abs(Polygon(bridge).area)) if len(bridge)>2 else 0.,
                  "section_area_mm2": float(polygon.area)}


def _resample_closed(path: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack((path, path[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.r_[0, np.cumsum(lengths)]
    target = np.linspace(0, distance[-1], count, endpoint=False)
    return np.column_stack((np.interp(target, distance, closed[:, 0]),
                            np.interp(target, distance, closed[:, 1])))


def _loft(contours: np.ndarray) -> trimesh.Trimesh:
    station_count, point_count, _ = contours.shape
    source_x, source_y = contours[:, :, 0], contours[:, :, 1]
    source_z = np.broadcast_to(STATIONS_MM[:, None], source_x.shape)
    vertices = np.column_stack(((source_z.ravel()-LONGITUDINAL_ORIGIN_MM)/1000,
                                -source_x.ravel()/1000, -source_y.ravel()/1000))
    faces = []
    for i in range(station_count-1):
        first = i*point_count+np.arange(point_count)
        next_point = i*point_count+(np.arange(point_count)+1)%point_count
        ahead, ahead_next = first+point_count, next_point+point_count
        faces.extend(np.column_stack((first, next_point, ahead)).tolist())
        faces.extend(np.column_stack((next_point, ahead_next, ahead)).tolist())
    for i in (0, station_count-1):
        ring = i*point_count+np.arange(point_count)
        next_point = i*point_count+(np.arange(point_count)+1)%point_count
        centre = vertices[ring].mean(axis=0)
        centre_index = len(vertices)
        vertices = np.vstack((vertices, centre))
        cap = np.column_stack((ring, next_point, np.full(point_count, centre_index)))
        if i == 0:
            cap = cap[:, ::-1]
        faces.extend(cap.tolist())
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def run(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    imported = trimesh.load_mesh(source, process=True)
    components = imported.split(only_watertight=False)
    port = next(part for part in components if part.bounds[1, 0] < 0)
    starboard = next(part for part in components if part.bounds[0, 0] > 0)
    source_sections = [_chains(port, z) for z in STATIONS_MM]
    try:
        [_chains(starboard, z) for z in STATIONS_MM]
        starboard_issue = None
    except ValueError as exc:
        # The mixed STEP face group branches and has disconnected loops in
        # the starboard pocket. Preserve that conflict in provenance.
        starboard_issue = str(exc)
    result = {"schema": "surveyor-chain-section-hypotheses-v1",
              "source_sha256": sha256(source.read_bytes()).hexdigest(),
              "station_step_mm": float(STATIONS_MM[1]-STATIONS_MM[0]),
              "contour_points": CONTOUR_POINTS,
              "starboard_independent_section_issue": starboard_issue,
              "starboard_source_treatment": "mirrored_from_cleaner_port; pocket-source disagreement unresolved",
              "hypotheses": {}}
    for name in ("smooth", "inset", "outset"):
        sections = [_close_section(chains, name, float(z)) for chains, z in
                    zip(source_sections, STATIONS_MM)]
        contours = np.asarray([_resample_closed(path, CONTOUR_POINTS)
                               for path, _ in sections])
        port_mesh = _loft(contours)
        starboard_mesh = port_mesh.copy()
        starboard_mesh.vertices[:, 1] *= -1
        starboard_mesh.invert()
        combined = trimesh.util.concatenate((port_mesh, starboard_mesh))
        combined.fix_normals()
        path = output / f"surveyor_{name}_FRD_m.stl"
        combined.export(path)
        section_audit = [{"source_Z_mm": float(z), **record} for z, (_, record) in
                         zip(STATIONS_MM, sections)]
        result["hypotheses"][name] = {
            "file": path.name, "sha256": sha256(path.read_bytes()).hexdigest(),
            "watertight": bool(combined.is_watertight),
            "winding_consistent": bool(combined.is_winding_consistent),
            "volume_m3": float(combined.volume),
            "port_volume_m3": float(port_mesh.volume),
            "starboard_volume_m3": float(starboard_mesh.volume),
            "surface_area_m2": float(combined.area),
            "volume_centroid_frd_m": combined.center_mass.tolist(),
            "bounds_frd_m": combined.bounds.tolist(),
            "face_count": len(combined.faces),
            "starboard_mirrored_from_port": True,
            "section_audit": section_audit,
        }
        np.savez_compressed(output / f"{name}_sections.npz", stations_mm=STATIONS_MM,
                            port_contours_source_xy_mm=contours)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, z in zip(axes, (700, 1050, 1200)):
        i = int(np.argmin(abs(STATIONS_MM-z)))
        for name in ("smooth", "inset", "outset"):
            path, _ = _close_section(source_sections[i], name, float(z))
            ax.plot(path[:, 0], path[:, 1], lw=1, label=name)
        ax.set(xlabel="source X (mm)", ylabel="source Y (mm)", title=f"Z={z} mm",
               aspect="equal")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "source_chain_sections.png", dpi=150); plt.close(fig)
    (output / "reconstruction_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print({name: {"volume_m3": row["volume_m3"], "watertight": row["watertight"]}
           for name, row in result["hypotheses"].items()})
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/loft_surveyor_chain_envelopes.py SOURCE_STL OUTPUT_DIR")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
