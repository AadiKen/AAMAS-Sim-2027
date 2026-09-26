#!/usr/bin/env python3
"""Short CPU scaling and supported feature-cost probes."""
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from stage5c_scaling_validation import bench_vector, bench_custom, environment_case, sensor_case

def write(path,rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,rows[0].keys());writer.writeheader();writer.writerows(rows)

start=time.perf_counter(); scale=ROOT/"paper_results/scaling"; feature=ROOT/"paper_results/feature_cost"
scale.mkdir(parents=True,exist_ok=True);feature.mkdir(parents=True,exist_ok=True)
scaling=[row for count in (1,8,32,128) for row in bench_vector(1,count,reps=2,steps=8)]
write(scale/"cpu_vector.csv",scaling)
features=[row for kind in ("still","current","wind","regular","irregular","combined","bathymetry","obstacles") for row in bench_custom(lambda kind=kind:environment_case(kind),kind,reps=2,steps=8)]
write(feature/"environment_cost.csv",features)
sensors=[row for kind in ("dynamics","gps_imu","lidar","sonar","mixed") for row in bench_custom(lambda kind=kind:sensor_case(kind),kind,reps=2,steps=8)]
write(feature/"sensor_cost.csv",sensors)
fig,ax=plt.subplots(figsize=(6,4)); counts=sorted(set(row["env_count"] for row in scaling)); means=[sum(row["aggregate_env_steps_per_s"] for row in scaling if row["env_count"]==n)/2 for n in counts]
ax.plot(counts,means,"o-");ax.set(xlabel="Environments",ylabel="Aggregate environment steps/s",title="CPU vector scaling");ax.grid(True,alpha=.3);fig.tight_layout();fig.savefig(scale/"throughput.png",dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4)); kinds=list(dict.fromkeys(row["workload"] for row in features)); vals=[sum(row["steps_per_s"] for row in features if row["workload"]==k)/2 for k in kinds]
ax.bar(kinds,vals);ax.tick_params(axis="x",rotation=35);ax.set(ylabel="Steps/s",title="Supported environmental configurations");fig.tight_layout();fig.savefig(feature/"environment_cost.png",dpi=180);plt.close(fig)
commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
manifest={"command":f"{sys.executable} tools/paper_runtime_quick.py","commit":commit,"dirty":dirty,"python":platform.python_version(),"torch":torch.__version__,"platform":platform.platform(),"device":"CPU","seed":53,"warmup_steps":2,"timed_steps":8,"repetitions":2,"runtime_s":time.perf_counter()-start,"status":"PASS","evidence_class":"characterization","caveat":"VectorEnvironment uses sequential Python stepping; no GPU implementation is claimed."}
for path in (scale,feature): (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"scaling_rows":len(scaling),"feature_rows":len(features)+len(sensors),"runtime_s":manifest["runtime_s"]}))
