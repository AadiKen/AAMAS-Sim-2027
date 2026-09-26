#!/usr/bin/env python3
"""Short current-code timestep convergence sweep, with a 5 ms reference."""
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

OUT = ROOT / "paper_results/timestep"; OUT.mkdir(parents=True, exist_ok=True)
params = bcod_parameters(resolved_parameters())
dt_values = (.1, .05, .02, .01, .005)
scenarios = {"straight": "T01", "turning": "T06", "combined_wrench": "T07"}
records = []; start = time.perf_counter()
for name, case_id in scenarios.items():
    traces = {}
    for dt in dt_values:
        case = case_definition(case_id, dt)
        n = int(round(10 / dt)) + 1
        for key in ("times", "wrench", "commands"):
            case[key] = case[key][:n]
        case["duration"] = 10.0
        t0 = time.perf_counter()
        header, data = simulate_bcod(case, params, np.zeros((n, 2)))
        elapsed = time.perf_counter() - t0
        traces[dt] = data
        np.savetxt(OUT / f"trace_{name}_dt{dt:.3f}.csv", data, delimiter=",", header=",".join(header), comments="")
        records.append({"scenario": name, "dt_s": dt, "steps": n-1, "runtime_s": elapsed, "steps_per_s": (n-1)/elapsed})
    ref = traces[.005]
    for row in records:
        if row["scenario"] != name: continue
        dt = row["dt_s"]; trace = traces[dt]
        indices = np.rint(trace[:,0] / .005).astype(int)
        aligned = ref[indices]
        pos = np.linalg.norm(trace[:,1:4] - aligned[:,1:4], axis=1)
        yaw = np.arctan2(np.sin(trace[:,10]-aligned[:,10]), np.cos(trace[:,10]-aligned[:,10]))
        vel = np.linalg.norm(trace[:,11:14]-aligned[:,11:14], axis=1)
        row.update(trajectory_rmse_m=float(np.sqrt(np.mean(pos**2))), terminal_position_error_m=float(pos[-1]), attitude_rmse_rad=float(np.sqrt(np.mean(yaw**2))), velocity_rmse_mps=float(np.sqrt(np.mean(vel**2))))
with (OUT / "metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, records[0].keys()); writer.writeheader(); writer.writerows(records)
fig, axes = plt.subplots(1, 2, figsize=(10,4))
for name in scenarios:
    rows = [row for row in records if row["scenario"] == name and row["dt_s"] > .005]
    axes[0].plot([r["dt_s"] for r in rows], [r["trajectory_rmse_m"] for r in rows], "o-", label=name)
    axes[1].plot([r["dt_s"] for r in rows], [r["runtime_s"] for r in rows], "o-", label=name)
axes[0].set(xlabel="Timestep (s)", ylabel="Position RMSE vs 5 ms (m)"); axes[1].set(xlabel="Timestep (s)", ylabel="Runtime (s)")
for ax in axes: ax.grid(True, alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "convergence.png", dpi=180); plt.close(fig)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
(OUT / "manifest.json").write_text(json.dumps({"command": f"{sys.executable} tools/paper_timestep.py", "commit": commit, "dirty": dirty, "python": platform.python_version(), "platform": platform.platform(), "device": "CPU", "seed": None, "resolved_scenarios": scenarios, "dt_values_s": dt_values, "reference_dt_s": .005, "duration_s": 10, "runtime_s": time.perf_counter()-start, "status": "PASS", "evidence_class": "internal_consistency"}, indent=2)+"\n")
print(json.dumps({"scenarios": len(scenarios), "runs": len(records)}))
