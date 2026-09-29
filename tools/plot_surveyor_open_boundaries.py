"""Diagnostic mesh-boundary plots from the explicitly invalid Surveyor candidate STL."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import trimesh


def run(candidate: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(candidate, process=True)
    diagnostics = {"schema": "surveyor-mesh-boundary-audit-v1", "source": str(candidate),
                   "source_is_valid_hull": False, "components": []}
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    for component_id, component in enumerate(mesh.split(only_watertight=False)):
        edges, counts = np.unique(component.edges_sorted, axis=0, return_counts=True)
        free = edges[counts == 1]
        graph = nx.Graph(); graph.add_edges_from(map(tuple, free))
        loops = []
        for vertices in sorted(nx.connected_components(graph), key=len, reverse=True):
            subset = graph.subgraph(vertices)
            points = component.vertices[list(vertices)]
            length = sum(np.linalg.norm(component.vertices[a]-component.vertices[b])
                         for a, b in subset.edges)
            loops.append({"edge_count": subset.number_of_edges(), "length_mm": float(length),
                          "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                          "closed_cycle": all(degree == 2 for _, degree in subset.degree)})
        diagnostics["components"].append({"id": component_id, "face_count": len(component.faces),
            "bounds_mm": component.bounds.tolist(), "surface_area_mm2": float(component.area),
            "watertight": bool(component.is_watertight), "free_edge_count": len(free),
            "free_edge_total_length_mm": float(sum(loop["length_mm"] for loop in loops)),
            "boundary_loops": loops})
        if not len(free):
            continue
        color = "tab:blue" if component.centroid[0] < 0 else "tab:orange"
        for a, b in free:
            p, q = component.vertices[[a, b]]
            for ax, ix, iy in ((axes[0], 2, 0), (axes[1], 2, 1), (axes[2], 0, 1)):
                ax.plot([p[ix]/1000, q[ix]/1000], [p[iy]/1000, q[iy]/1000],
                        color=color, linewidth=.65, alpha=.85)
    for ax, title, xlabel, ylabel in zip(axes,
            ("Top: longitudinal vs lateral", "Side: longitudinal vs vertical", "Front: lateral vs vertical"),
            ("source Z / m", "source Z / m", "source X / m"),
            ("source X / m", "source Y / m", "source Y / m")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.25)
    fig.suptitle("Open edges in selected STEP-face tessellation; invalid for coefficients")
    fig.tight_layout(); fig.savefig(output / "free_edges_before.png", dpi=160); plt.close(fig)
    (output / "mesh_topology.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/plot_surveyor_open_boundaries.py CANDIDATE_STL OUTPUT_DIR")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
