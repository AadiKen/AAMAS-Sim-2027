#!/usr/bin/env python3
"""Index existing numeric evidence without modifying source artifacts."""
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_results"
FIELDS = ["experiment", "metric", "value", "unit", "scenario", "backend", "device", "seed", "evidence_class", "source_artifact", "commit"]
SKIP_PARTS = {".git", ".venv", ".venv-pyquaticus", "node_modules", "paper_results", "cfd", "openfoam"}


def run(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True).stdout.strip()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def main():
    for name in ("validation", "mss", "timestep", "determinism", "scaling", "feature_cost", "heterogeneity", "sensors", "environment_import", "vessel_generation", "calibration", "cross_sim", "figures"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    roots = [ROOT / name for name in ("stage2_results", "stage3_results", "stage4_results", "stage5_results", "validation", "benchmark", "benchmarks", "results", "reports", "artifacts")]
    files = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in (".json", ".csv", ".parquet") and not SKIP_PARTS.intersection(path.relative_to(ROOT).parts) and path.stat().st_size <= 5_000_000)
    files = sorted(set(files))
    commit = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    now = datetime.now(timezone.utc).isoformat()
    inventory, metrics = [], []
    for path in files:
        rel = str(path.relative_to(ROOT))
        count = 0
        error = ""
        cls = "external_reference" if "mss" in rel or "kcs" in rel or "dtmb" in rel else "internal_consistency" if "validation" in rel or "stage2" in rel else "characterization"
        def add(metric, value, scenario=""):
            nonlocal count
            metrics.append({"experiment": path.parent.name, "metric": metric, "value": value, "unit": "unspecified", "scenario": scenario or path.stem, "backend": "archived", "device": "unspecified", "seed": "unspecified", "evidence_class": cls, "source_artifact": rel, "commit": "see_source_manifest"})
            count += 1
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(errors="replace"))
                for key, value in flatten(data):
                    add(key, value)
            elif path.suffix.lower() == ".csv" and any(token in path.stem.lower() for token in ("timeseries", "trajectory", "trace", "input", "comparison", "pressure", "forces")):
                error = "Raw series preserved at source; omitted from summary metrics"
            elif path.suffix.lower() == ".csv":
                with path.open(newline="", errors="replace") as handle:
                    for idx, row in enumerate(csv.DictReader(handle)):
                        if idx >= 1000:
                            error = "First 1000 rows indexed; full table preserved at source"
                            break
                        for key, value in row.items():
                            try:
                                number = float(value)
                            except (TypeError, ValueError):
                                continue
                            if number == number and abs(number) != float("inf"):
                                add(key, value, f"row_{idx}")
            else:
                error = "Parquet indexed, numeric extraction unavailable"
        except (OSError, ValueError, csv.Error) as exc:
            error = f"{type(exc).__name__}: {exc}"
        inventory.append({"source_artifact": rel, "format": path.suffix.lower(), "bytes": path.stat().st_size, "numeric_metrics": count, "evidence_class": cls, "publication_ready": "review_required", "note": error})
    with (OUT / "paper_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS); writer.writeheader(); writer.writerows(metrics)
    with (OUT / "validation/existing_evidence_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, inventory[0].keys()); writer.writeheader(); writer.writerows(inventory)
    (OUT / "manifest.json").write_text(json.dumps({"timestamp_utc": now, "command": f"{sys.executable} tools/harvest_paper_evidence.py", "commit": commit, "dirty": dirty, "platform": platform.platform(), "python": platform.python_version(), "seed": None, "files_indexed": len(files), "numeric_values_extracted": len(metrics), "status": "PASS", "evidence_class": "characterization"}, indent=2) + "\n")
    (OUT / "validation/existing_evidence_summary.md").write_text(f"# Existing evidence inventory\n\nIndexed {len(files)} JSON/CSV/Parquet files and extracted {len(metrics)} numeric values at {now}. Original artifact paths are preserved in the inventory and master CSV.\n\nPublication readiness requires review of each source's manifest, reference provenance, units, and dirty status. Values with unspecified units or commit are discovery records, not final paper claims. Parquet files are indexed but not parsed. Large files and CFD trees are excluded.\n")
    print(json.dumps({"files": len(files), "metrics": len(metrics)}))


if __name__ == "__main__":
    main()
