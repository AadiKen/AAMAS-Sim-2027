"""Render static time–distance and load figures from the saved diagnostic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_stage3_sst_diagnostic_probes import CASE, force_file, probe_file, interface_elevation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.case.parent / "diagnostic_60_80"
    output.mkdir(parents=True, exist_ok=True)
    probe = args.case / "postProcessing/diagnosticProbes/60"
    time, points, alpha = probe_file(probe / "alpha.water")
    horizontal, eta = interface_elevation(alpha, points)
    # Keep one common signed scale, in millimetres, so side/outlet differences
    # are not exaggerated by independent panel autoscaling.
    anomaly = eta - np.nanmean(eta, axis=0)
    scale = max(float(np.nanpercentile(np.abs(anomaly), 98)) * 1000, .01)
    routes = {
        "Downstream": list(range(6, 15)),
        "Upstream": list(range(5, -1, -1)),
        "Port": list(range(22, 30)),
        "Starboard": list(range(21, 14, -1)),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (name, indices) in zip(axes.flat, routes.items()):
        distance = np.r_[0, np.cumsum(np.linalg.norm(
            np.diff(horizontal[indices], axis=0), axis=1))]
        values = (anomaly[:, indices] * 1000).T
        image = axis.pcolormesh(time, distance, values, shading="nearest",
                                cmap="RdBu_r", vmin=-scale, vmax=scale)
        axis.set_title(name)
        axis.set_xlabel("Simulated time (s)")
        axis.set_ylabel("Distance outward from first probe (m)")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Interface anomaly (mm)",
                 shrink=.85)
    fig.suptitle("WAM-V SST diagnostic: interface time–distance traces")
    fig.savefig(output / "interface_time_distance.png", dpi=180)
    plt.close(fig)

    load_time, pressure, viscous = force_file(
        args.case / "postProcessing/forces/60/forces.dat")
    total = pressure + viscous
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for axis, name, index, unit in zip(axes, ("Fx", "Fz", "My"), (0, 2, 4),
                                       ("N", "N", "N m")):
        axis.plot(load_time, total[:, index], color="#17548a", lw=.8)
        axis.plot(load_time, pressure[:, index], color="#c05728", lw=.6,
                  alpha=.75, label="Pressure")
        axis.set_ylabel(f"{name} ({unit})")
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Simulated time (s)")
    fig.suptitle("WAM-V SST diagnostic: 60–80 s load histories")
    fig.savefig(output / "load_histories.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
