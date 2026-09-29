"""Plot the three Surveyor section-envelope equilibrium waterlines."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def run(root: Path, output: Path) -> None:
    mesh = trimesh.load_mesh(root/"surveyor_smooth_FRD_m.stl", process=True)
    points = mesh.vertices[::max(1, len(mesh.vertices)//20000)]
    hydro = json.loads((root/"hydrostatics_52_3kg.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(points[:, 0], points[:, 1], s=.2, alpha=.2)
    axes[0].set(xlabel="x forward (m)", ylabel="y starboard (m)", title="Top")
    axes[1].scatter(points[:, 0], points[:, 2], s=.2, alpha=.2)
    axes[2].scatter(points[:, 1], points[:, 2], s=.2, alpha=.2)
    axes[1].set(xlabel="x forward (m)", ylabel="z down (m)", title="Side")
    axes[2].set(xlabel="y starboard (m)", ylabel="z down (m)", title="Front")
    for name, color in (("smooth", "black"), ("inset", "tab:orange"), ("outset", "tab:green")):
        level = hydro[name]["equilibrium_52_3kg"]["waterline_z_frd_m"]
        axes[1].axhline(level, color=color, linewidth=1.2, label=f"{name}: {hydro[name]['equilibrium_draft_m']:.4f} m")
        axes[2].axhline(level, color=color, linewidth=1.2)
    published_level = float(mesh.bounds[1, 2]-.17)
    axes[1].axhline(published_level, color="red", linestyle="--", linewidth=1.3,
                    label="published draft: 0.1700 m")
    axes[2].axhline(published_level, color="red", linestyle="--", linewidth=1.3)
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Diagnostic section envelopes: 52.3 kg equilibrium and published draft")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170);plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/plot_surveyor_chain_hydrostatics.py HYPOTHESES_DIR OUTPUT_PNG")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
