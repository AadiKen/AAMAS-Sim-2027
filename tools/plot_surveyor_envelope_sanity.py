"""Show the reconstructed envelope against the published draft reference."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def run(mesh_path: Path, output: Path) -> None:
    mesh = trimesh.load_mesh(mesh_path, process=True)
    points = mesh.vertices[::max(1, len(mesh.vertices)//10000)]
    waterline_z = float(mesh.bounds[1, 2]-.17)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(points[:, 0], points[:, 1], s=.2, alpha=.2)
    axes[0].set(xlabel="x forward (m)", ylabel="y starboard (m)", title="Top")
    axes[1].scatter(points[:, 0], points[:, 2], s=.2, alpha=.2)
    axes[1].axhline(waterline_z, color="red", lw=2, label="0.17 m draft from keel")
    axes[1].set(xlabel="x forward (m)", ylabel="z down (m)", title="Side")
    axes[1].legend(fontsize=8)
    axes[2].scatter(points[:, 1], points[:, 2], s=.2, alpha=.2)
    axes[2].axhline(waterline_z, color="red", lw=2)
    axes[2].set(xlabel="y starboard (m)", ylabel="z down (m)", title="Front")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Exploratory Surveyor pontoon envelopes: published draft line lies above all modelled volume")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/plot_surveyor_envelope_sanity.py MESH_STL OUTPUT_PNG")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
