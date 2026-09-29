"""Plot conditional frozen-M1 coefficient spread across Surveyor pocket variants."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bcod_sim.vessel_generation.coefficient_package import load_coefficient_package, reference_wrench


def run(root: Path, output: Path) -> None:
    packages = {name: load_coefficient_package(root/name/"coefficient_package.yaml")
                for name in ("smooth", "inset", "outset")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    speeds = np.linspace(0, 3, 25)
    sways = np.linspace(-.5, .5, 25)
    yaws = np.linspace(-.4, .4, 25)
    for name, package in packages.items():
        x = [reference_wrench(package, np.array([u, 0, 0, 0, 0, 0]))[0] for u in speeds]
        y = [reference_wrench(package, np.array([0, v, 0, 0, 0, 0]))[1] for v in sways]
        n = [reference_wrench(package, np.array([0, 0, 0, 0, 0, r]))[5] for r in yaws]
        axes[0, 0].plot(speeds, x, label=name)
        axes[0, 1].plot(sways, y, label=name)
        axes[1, 0].plot(yaws, n, label=name)
        added = np.asarray(package["added_mass"]["matrix_6x6"])
        axes[1, 1].plot([0, 1, 2, 3], np.diag(added)[[1, 2, 4, 5]], marker="o", label=name)
    axes[0, 0].set(xlabel="surge u (m/s)", ylabel="X (N)", title="Frozen M1 surge resistance")
    axes[0, 1].set(xlabel="pure sway v (m/s)", ylabel="Y (N)", title="Frozen M1 lateral force")
    axes[1, 0].set(xlabel="pure yaw r (rad/s)", ylabel="N (N m)", title="Frozen M1 yaw moment")
    axes[1, 1].set_xticks([0, 1, 2, 3], ["sway", "heave", "pitch", "yaw"])
    axes[1, 1].set(ylabel="diagonal M_A (mixed SI units)", title="Strip-estimated added mass diagonal")
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Conditional pocket-bridge spread; source classification uncertainty excluded")
    fig.tight_layout();output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170);plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/plot_surveyor_coefficient_spread.py COEFFICIENT_DIR OUTPUT_PNG")
    run(Path(sys.argv[1]), Path(sys.argv[2]))
