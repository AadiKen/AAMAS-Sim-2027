#!/usr/bin/env python3
"""Compare current Python dynamics to archived pinned MSS Otter traces."""
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from mss_6dof_validation import bcod_parameters, case_definition, resolved_parameters, simulate_bcod

SOURCE = ROOT / "artifacts/mss-6dof-validation/latest"
OUT = ROOT / "paper_results/mss"
OUT.mkdir(parents=True, exist_ok=True)
cases = {"T01": "surge", "T08": "initial_decay", "T10": "turn_left", "T12": "combined_actuator"}
params = bcod_parameters(resolved_parameters())
all_rows = []
start = time.perf_counter()
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for case_id, name in cases.items():
    path = SOURCE / "cases" / f"{case_id}_{name}" / "mss_trajectory.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    reference = np.array([[float(row[key]) for key in reader.fieldnames] for row in rows])
    dt = float(reference[1, 0] - reference[0, 0])
    case = case_definition(case_id, dt)
    if len(reference) != len(case["times"]) or not np.allclose(reference[:, 0], case["times"], atol=1e-12):
        raise RuntimeError(f"Time grid mismatch: {case_id}")
    actual_n = reference[:, 25:27]
    header, actual = simulate_bcod(case, params, actual_n)
    if not np.array_equal(actual[:, 0], reference[:, 0]):
        raise RuntimeError(f"Timestamp mismatch: {case_id}")
    pos_error = np.linalg.norm(actual[:, 1:4] - reference[:, 1:4], axis=1)
    heading_error = np.arctan2(np.sin(actual[:, 10] - reference[:, 10]), np.cos(actual[:, 10] - reference[:, 10]))
    diff = actual - reference
    values = {
        "position_rmse_m": float(np.sqrt(np.mean(pos_error**2))),
        "position_max_m": float(np.max(pos_error)),
        "position_terminal_m": float(pos_error[-1]),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading_error**2))),
        "heading_max_rad": float(np.max(np.abs(heading_error))),
        "heading_terminal_rad": float(abs(heading_error[-1])),
        "surge_rmse_mps": float(np.sqrt(np.mean(diff[:, 11]**2))),
        "sway_rmse_mps": float(np.sqrt(np.mean(diff[:, 12]**2))),
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(diff[:, 16]**2))),
    }
    all_rows.append({"case": case_id, "name": name, **values})
    with (OUT / f"trace_{case_id}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow([f"bcod_{x}" for x in header] + [f"mss_{x}" for x in header]); writer.writerows(np.column_stack((actual, reference)))
    if case_id == "T10":
        axes[0,0].plot(reference[:,1], reference[:,2], label="MSS"); axes[0,0].plot(actual[:,1], actual[:,2], label="Current Python")
        axes[0,1].plot(reference[:,0], reference[:,10]); axes[0,1].plot(actual[:,0], actual[:,10])
        axes[1,0].plot(reference[:,0], reference[:,11]); axes[1,0].plot(actual[:,0], actual[:,11])
        axes[1,1].plot(reference[:,0], reference[:,16]); axes[1,1].plot(actual[:,0], actual[:,16])
for ax, title in zip(axes.flat, ("XY trajectory", "Heading (rad)", "Surge speed (m/s)", "Yaw rate (rad/s)")):
    ax.set_title(title); ax.grid(True, alpha=.3)
axes[0,0].legend(); fig.tight_layout(); fig.savefig(OUT / "mss_comparison.png", dpi=180); plt.close(fig)
with (OUT / "metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
manifest = {"command": f"{sys.executable} tools/paper_mss_current.py", "commit": commit, "dirty": dirty, "reference_manifest": str((SOURCE / "reference/source_manifest.json").relative_to(ROOT)), "reference_type": "archived pinned MSS traces", "device": "CPU", "python": platform.python_version(), "platform": platform.platform(), "seed": None, "runtime_s": time.perf_counter() - start, "status": "PASS", "evidence_class": "external_reference", "resolved_cases": list(cases)}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(OUT / "report.md").write_text("# Current Python vs pinned MSS Otter traces\n\nThe MSS reference traces were produced by the pinned source listed in the archived source manifest. Current Python code was run anew with matching time grids, initial conditions, wrench inputs, and archived actual shaft speeds. This is a comparison to archived external traces, not a fresh Octave run.\n\n" + "\n".join(f"- {row['case']} {row['name']}: position RMSE {row['position_rmse_m']:.6g} m; heading RMSE {row['heading_rmse_rad']:.6g} rad" for row in all_rows) + "\n")
print(json.dumps({"cases": len(all_rows), "runtime_s": manifest["runtime_s"]}))
