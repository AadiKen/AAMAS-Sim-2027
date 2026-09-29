"""Reproducible manufactured CAD sweep; no external benchmark data.

Usage: python tools/run_manta_milestone1.py OUTPUT_DIRECTORY
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import trimesh

from bcod_sim.vessel_generation.coefficient_package import load_coefficient_package, reference_wrench
from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel


def _generate(root: Path, name: str, mesh: trimesh.Trimesh, mass: float) -> dict:
    source = root / f"{name}.stl"
    mesh.export(source)
    package_root = generate_simple_vessel(geometry=source, output=root / name,
        mass_kg=mass, cg_frd_m=(0., 0., 0.), units="m", disable_bem=True,
        lut_samples=3)
    package = load_coefficient_package(package_root / "coefficient_package.yaml")
    surge = reference_wrench(package, np.array([1., 0., 0., 0., 0., 0.]))[0]
    sway = reference_wrench(package, np.array([.5, .2, 0., 0., 0., 0.]))[1]
    yaw = reference_wrench(package, np.array([0., 0., 0., 0., 0., .2]))[5]
    return {"name": name, "package_sha256": package["canonical_sha256"],
            "volume_m3": package["hydrostatics"]["volume_m3"],
            "surge_drag_n_at_1_mps": float(surge),
            "sway_force_n_at_v_0p2": float(sway),
            "yaw_moment_nm_at_r_0p2": float(yaw)}


def run(output: str | Path) -> dict:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CAPYTAINE_CACHE_DIR", str(root / "capytaine_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for axis, label in enumerate(("length", "beam", "depth")):
        values = []
        for factor in (.75, 1., 1.25):
            size = [4., 2., 1.]
            size[axis] *= factor
            mesh = trimesh.creation.box(extents=size)
            entry = _generate(root, f"{label}_{factor}", mesh, 1025*np.prod(size)/4)
            values.append({"factor": factor, **entry})
        results[label] = values
    fullness = []
    for exponent in (.75, 1., 1.25):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices = np.sign(mesh.vertices)*np.abs(mesh.vertices)**exponent
        mesh.apply_scale((2., 1., .5))
        fullness.append({"exponent": exponent,
            **_generate(root, f"fullness_{exponent}", mesh, mesh.volume*1025/2)})
    results["fullness"] = fullness
    spacing = []
    for offset in (.7, 1., 1.3):
        port = trimesh.creation.box(extents=(4., .6, 1.))
        starboard = port.copy()
        port.apply_translation((0., -offset, 0.))
        starboard.apply_translation((0., offset, 0.))
        spacing.append({"half_spacing_m": offset,
            **_generate(root, f"spacing_{offset}", trimesh.util.concatenate((port, starboard)), 1230.)})
    results["catamaran_spacing"] = spacing
    (root / "geometry_sweep.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for index, (label, entries) in enumerate(results.items()):
        xkey = "factor" if label in ("length", "beam", "depth") else "exponent" if label == "fullness" else "half_spacing_m"
        x = [item[xkey] for item in entries]
        for row, key, title in ((0, "sway_force_n_at_v_0p2", "Y at v=0.2 m/s"),
                                (1, "yaw_moment_nm_at_r_0p2", "N at r=0.2 rad/s")):
            ax = axes[row, index]
            ax.plot(x, [item[key] for item in entries], marker="o")
            ax.set(xlabel=label, ylabel=title)
    fig.tight_layout(); fig.savefig(root / "geometry_sweep.png", dpi=140); plt.close(fig)
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/run_manta_milestone1.py OUTPUT_DIRECTORY")
    run(sys.argv[1])
