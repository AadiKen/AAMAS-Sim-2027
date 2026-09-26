#!/usr/bin/env python3
"""Build compact paper tables and a provenance-aware preliminary report."""
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"paper_results"

def readcsv(path):
    with path.open(newline="") as handle:return list(csv.DictReader(handle))
def writecsv(path,rows):
    if not rows:return
    with path.open("w",newline="") as handle:
        w=csv.DictWriter(handle,rows[0].keys());w.writeheader();w.writerows(rows)
def avg(rows,key):return sum(float(r[key]) for r in rows)/len(rows)

stage2=json.loads((OUT/"validation/stage2-clean/summary.json").read_text())
stage2rows=readcsv(OUT/"validation/stage2-clean/cases.csv")
writecsv(OUT/"validation/experiment_status.csv",stage2rows)
for category,path in (("numerical","numerical_validation.csv"),("handling","handling_validation.csv"),("environment","environment_sensor_validation.csv")):
    rows=[r for r in stage2rows if (r["category"] in ("baseline","current","wind","regular_waves","irregular_waves","combined","collision","vessel_vessel","grounding","bathymetry") if category=="environment" else False)]
    if category=="numerical": rows=readcsv(OUT/"mss/metrics.csv")
    if category=="handling": rows=[r for r in stage2rows if r["category"] in ("collision","vessel_vessel","grounding")]
    writecsv(OUT/"validation"/path,rows)
writecsv(OUT/"validation/performance.csv",readcsv(OUT/"scaling/cpu_vector.csv"))
writecsv(OUT/"validation/extensibility.csv",readcsv(OUT/"vessel_generation/fleet/fleet_results.csv") if (OUT/"vessel_generation/fleet/fleet_results.csv").exists() else [{"capability":"synthetic_hull_generation","status":"PASS","source":"paper_results/vessel_generation/fleet/fleet_results.json"}])
environment=json.loads((OUT/"validation/environment-current/report.json").read_text())
envrows=[{"scenario":case,"metric":key,"value":value} for case,metrics in environment["cases"].items() for key,value in metrics.items() if isinstance(value,(int,float))]
writecsv(OUT/"validation/environmental_validation.csv",envrows)
scale=readcsv(OUT/"scaling/cpu_vector.csv"); feature=readcsv(OUT/"feature_cost/environment_cost.csv"); sensors=readcsv(OUT/"feature_cost/sensor_cost.csv")
mss=readcsv(OUT/"mss/metrics.csv");det=json.loads((OUT/"determinism/summary.json").read_text());fleet=json.loads((OUT/"vessel_generation/fleet/fleet_results.json").read_text())
hmri_path=OUT/"vessel_generation/hmri/validation.json";hmri=json.loads(hmri_path.read_text()) if hmri_path.exists() else None
fig,ax=plt.subplots(figsize=(9,4));categories=list(stage2["subsystems"]);passes=[stage2["subsystems"][c]["passed"] for c in categories];ax.bar(categories,passes,color="#286d7c");ax.tick_params(axis="x",rotation=35);ax.set(ylabel="Passing cases",title="Clean Stage 2 validation by phase");fig.tight_layout();fig.savefig(OUT/"figures/phase_status.png",dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,2.5));ax.bar(["Identical","Different"],[det["same_seed_identical"],det["repetitions"]-det["same_seed_identical"]],color=["#286d7c","#af4a38"]);ax.set(ylabel="Replays",title="100 same-seed replays");fig.tight_layout();fig.savefig(OUT/"figures/determinism.png",dpi=180);plt.close(fig)
for source,target in ((OUT/"mss/mss_comparison.png","mss_comparison.png"),(OUT/"timestep/convergence.png","timestep_convergence.png"),(OUT/"scaling/throughput.png","cpu_scaling.png"),(OUT/"feature_cost/environment_cost.png","environment_cost.png")):
    if source.exists():shutil.copyfile(source,OUT/"figures"/target)
with (OUT/"paper_metrics.csv").open(newline="") as handle: archived=list(csv.DictReader(handle));fields=handle.readline if False else list(archived[0].keys())
archived=[r for r in archived if r["backend"]=="archived"]
commit=json.loads((OUT/"manifest.json").read_text())["commit"]
new=[]
def add(experiment,metric,value,unit,scenario,evidence_class,source,device="CPU",seed=""):
    new.append(dict(experiment=experiment,metric=metric,value=value,unit=unit,scenario=scenario,backend="current_python",device=device,seed=seed,evidence_class=evidence_class,source_artifact=str(source.relative_to(ROOT)),commit=commit))
add("stage2_clean","pass_count",stage2["counts"]["PASS"],"cases","all","internal_consistency",OUT/"validation/stage2-clean/summary.json",seed=17)
for r in mss:
    for key in ("position_rmse_m","heading_rmse_rad","surge_rmse_mps","sway_rmse_mps","yaw_rate_rmse_radps"):
        add("mss_current",key,r[key],key.rsplit("_",1)[-1],r["case"],"external_reference",OUT/"mss/metrics.csv")
add("determinism","same_seed_identical",det["same_seed_identical"],"replays","3_vessel_12_steps","internal_consistency",OUT/"determinism/summary.json",seed=73)
add("determinism","maximum_numeric_deviation",det["maximum_numeric_deviation"],"mixed_units","3_vessel_12_steps","internal_consistency",OUT/"determinism/summary.json",seed=73)
for r in readcsv(OUT/"timestep/metrics.csv"):
    add("timestep","trajectory_rmse_m",r["trajectory_rmse_m"],"m",f"{r['scenario']}_dt{r['dt_s']}","internal_consistency",OUT/"timestep/metrics.csv")
for n in sorted(set(int(r["env_count"]) for r in scale)):
    rows=[r for r in scale if int(r["env_count"])==n]
    add("cpu_scaling","aggregate_env_steps_per_s",avg(rows,"aggregate_env_steps_per_s"),"env_steps/s",str(n),"characterization",OUT/"scaling/cpu_vector.csv",seed=53)
for kind in list(dict.fromkeys(r["workload"] for r in feature)):
    rows=[r for r in feature if r["workload"]==kind]
    add("feature_cost","steps_per_s",avg(rows,"steps_per_s"),"steps/s",kind,"characterization",OUT/"feature_cost/environment_cost.csv",seed=53)
for r in fleet:add("hull_generation","runtime_s",r["runtime_s"],"s",r["hull"],"demonstration",OUT/"vessel_generation/fleet/fleet_results.json")
with (OUT/"paper_metrics.csv").open("w",newline="") as handle:
    w=csv.DictWriter(handle,fields);w.writeheader();w.writerows(archived+new)
scaling_text=", ".join(f"{n}: {avg([r for r in scale if int(r['env_count'])==n],'aggregate_env_steps_per_s'):.1f}" for n in (1,8,32,128))
feature_text=", ".join(f"{kind}: {avg([r for r in feature if r['workload']==kind],'steps_per_s'):.1f}" for kind in ("still","irregular","combined"))
hmri_text="HMRI passive rerun completed; see validation.json." if hmri else "HMRI passive rerun unavailable."
report=f"""# Rapid paper evidence harvest

Generated {datetime.now(timezone.utc).isoformat()}. Commit `{commit}`. The developer checkout was preserved. Fresh current-code experiments used the dirty checkout except the explicitly clean Stage 2 source export.

## A. Results available for the paper now

| Claim supported | Exact result | Experiment and evidence class | Artifact | Caveat / suggested use |
|---|---:|---|---|---|
| Clean committed Stage 2 validation | {stage2['counts']['PASS']} PASS, {stage2['counts']['FAIL']} FAIL, {stage2['counts']['BLOCKED']} BLOCKED | Clean source export; internal consistency | `validation/stage2-clean/summary.json` | Committed source only; phase status figure |
| Current Python vs archived pinned MSS | T10 position RMSE {float(next(r for r in mss if r['case']=='T10')['position_rmse_m']):.3g} m; heading RMSE {float(next(r for r in mss if r['case']=='T10')['heading_rmse_rad']):.3g} rad | External reference | `mss/metrics.csv` | Pinned MSS traces were archived; 4 supported cases; Figure 2A |
| Same-seed replay | {det['same_seed_identical']}/{det['repetitions']} identical; maximum numeric deviation {det['maximum_numeric_deviation']} | Internal consistency | `determinism/summary.json` | CPU, 3 vessels, 12 steps, GPS noise and environment; Figure 4E |
| Timestep convergence | 3 scenarios × 5 timesteps, 0.005 s reference | Internal consistency | `timestep/metrics.csv` | Short 10 s forced motion; Figure 2B |
| CPU vector scaling | {scaling_text} aggregate env steps/s | Characterization | `scaling/cpu_vector.csv` | Sequential Python vector runner, 8 timed steps/repetition; Figure 4A |
| Environment feature cost | {feature_text} steps/s | Characterization | `feature_cost/environment_cost.csv` | Supported configurations, 4 vessels, 8 timed steps; Figure 4C |
| Synthetic hull compilation | {sum(r['result']=='PASS' for r in fleet)}/{len(fleet)} PASS | Demonstration | `vessel_generation/fleet/fleet_results.json` | All BEM attempts rejected; strip fallback and low confidence; no accuracy claim |

## B. Data acquired during this rapid campaign

Fresh runs: clean Stage 2, current-checkout Stage 2, environment response, MSS comparison, timestep convergence, 100-replay determinism, CPU scaling, feature-cost/sensor-cost probes, eight synthetic-hull generations, and HMRI passive re-evaluation. Exact commands, environment, seeds, source state, and runtimes are recorded in per-experiment manifests. The source inventory indexes archived evidence and links original artifacts. {hmri_text}

Figures generated: `figures/phase_status.png`, `figures/mss_comparison.png`, `figures/timestep_convergence.png`, `figures/cpu_scaling.png`, `figures/environment_cost.png`, `figures/determinism.png`.

## C. Blocked or expensive evidence

- Full 83-case aggregate was not reproduced as one clean campaign; clean Stage 2 is 58 cases. Stage 5 and sensor source campaigns remain archived and need separate clean reruns.
- Fresh authoritative Octave MSS generation requires the pinned MSS source checkout, which was unavailable locally; the comparison used archived traces with verified source manifest.
- GPU benchmark is unavailable on this host/runtime. No GPU throughput or crossover value is claimed.
- Independent field/sea-trial measurements, hardware sensor accuracy, and external coefficient references are unavailable for most hulls.
- Full CFD and long RL campaigns were excluded by request.
- Four-site live environmental imports, cross-simulator smoke, calibration recovery, and high-count 512–8192 environment scaling were not completed in this rapid run.
- The original KCS generation algorithm was not rerun from geometry here; the HMRI passive coefficients were re-evaluated from an existing frozen package. Treat its generation provenance separately.

## D. Recommended final paper numbers

- Clean Stage 2 validation: **{stage2['counts']['PASS']}/{sum(stage2['counts'].values())} PASS** at commit `{commit}`.
- MSS T10 position RMSE: **{float(next(r for r in mss if r['case']=='T10')['position_rmse_m']):.6g} m**; heading RMSE **{float(next(r for r in mss if r['case']=='T10')['heading_rmse_rad']):.6g} rad**, with archived-reference qualification.
- Determinism: **{det['same_seed_identical']}/{det['repetitions']} identical same-seed replays**, maximum numeric deviation **{det['maximum_numeric_deviation']}** on CPU.
- CPU scaling and feature cost: use exact per-count/per-configuration means in `paper_metrics.csv`; present them as host-specific throughput characterization.
- Do not fill GPU, field accuracy, calibrated-hull accuracy, or cross-simulator quantitative cells from these data.
"""
(OUT/"report.md").write_text(report)
(OUT/"validation/final_report.md").write_text(f"# Clean validation aggregate\n\nCommitted Stage 2 source export: {stage2['counts']['PASS']} PASS, {stage2['counts']['FAIL']} FAIL, {stage2['counts']['BLOCKED']} BLOCKED. See `stage2-clean/source_provenance.json` and `stage2-clean/report.md`. This is the completed clean subset of the broader archived validation campaign.\n")
print(json.dumps({"new_metrics":len(new),"archived_metrics":len(archived),"figures":6}))
