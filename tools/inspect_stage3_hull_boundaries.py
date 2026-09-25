#!/usr/bin/env python3
"""Inspect, but never repair, boundary topology in the Stage 3 Otter hull candidates."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage3_openfoam_validation import _parse_obj


def inspect_mesh(vertices, faces):
    welded, coordinates, welded_faces = {}, [], []
    for face in faces:
        converted = []
        for source_index in face:
            key = tuple(round(value, 7) for value in vertices[source_index])
            if key not in welded:
                welded[key] = len(coordinates)
                coordinates.append(np.asarray(key))
            converted.append(welded[key])
        welded_faces.append(tuple(converted))
    edge_counts = Counter(
        tuple(sorted(edge))
        for a, b, c in welded_faces
        for edge in ((a, b), (b, c), (c, a))
    )
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency = defaultdict(set)
    for a, b in boundary_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    unseen, networks = set(adjacency), []
    while unseen:
        seed = next(iter(unseen)); stack = [seed]; members = {seed}; unseen.remove(seed)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in members:
                    members.add(neighbor); unseen.discard(neighbor); stack.append(neighbor)
        network_edges = [edge for edge in boundary_edges if edge[0] in members]
        points = np.asarray([coordinates[index] for index in members])
        center = points.mean(axis=0)
        _, singular_values, vh = np.linalg.svd(points - center, full_matrices=False)
        distances = np.abs((points - center) @ vh[-1])
        degrees = Counter(len(adjacency[index]) for index in members)
        networks.append({
            "vertices": len(members), "edges": len(network_edges),
            "degree_counts": {str(key): value for key, value in sorted(degrees.items())},
            "simple_closed_loop": degrees == {2: len(members)},
            "bbox_m": (points.max(axis=0) - points.min(axis=0)).tolist(),
            "center_m": center.tolist(),
            "edge_length_m": float(sum(np.linalg.norm(coordinates[a] - coordinates[b]) for a, b in network_edges)),
            "best_fit_plane_max_error_m": float(distances.max()),
            "best_fit_plane_rms_error_m": float(np.sqrt(np.mean(distances**2))),
            "singular_values": singular_values.tolist(),
        })
    networks.sort(key=lambda item: item["edge_length_m"], reverse=True)
    return coordinates, boundary_edges, {
        "boundary_edges": len(boundary_edges),
        "boundary_networks": len(networks),
        "simple_closed_loops": sum(item["simple_closed_loop"] for item in networks),
        "branched_or_open_networks": sum(not item["simple_closed_loop"] for item in networks),
        "networks": networks,
    }


def render(results, plot_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, (name, (coordinates, edges, _)) in enumerate(results.items()):
        for column, (horizontal, vertical, labels) in enumerate(((2, 0, ("native Z / length (m)", "native X / beam (m)")), (2, 1, ("native Z / length (m)", "native Y / vertical (m)")))):
            axis = axes[row, column]
            for a, b in edges:
                axis.plot([coordinates[a][horizontal], coordinates[b][horizontal]], [coordinates[a][vertical], coordinates[b][vertical]], color="#c43c39", linewidth=0.8)
            axis.set_title(f"{name}: boundary edges")
            axis.set_xlabel(labels[0]); axis.set_ylabel(labels[1]); axis.grid(alpha=0.25); axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle("Otter visual hull boundary topology (no repair applied)")
    figure.savefig(plot_path, dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    vertices, groups = _parse_obj(args.obj)
    selected = sorted(groups, key=lambda name: len(groups[name]), reverse=True)[:2]
    raw = {name: inspect_mesh(vertices, groups[name]) for name in selected}
    payload = {
        "selection": "two greatest triangle-count components",
        "classification": "IRREGULAR_REQUIRES_GEOMETRY_GUESSING",
        "repair_performed": False,
        "reason": "Multiple branched boundary networks and unmatched port/starboard openings are not deterministic planar closure loops.",
        "hulls": {name: data[2] for name, data in raw.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    render(raw, args.plot)


if __name__ == "__main__":
    main()
