"""Bounded diagnostic monitor for the sole +6 degree outlet-damping run."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

import monitor_kcs_hmri_plus6 as base

ROOT = Path("stage3_results/kcs-validation/track_b_hmri/static_drift_plus6_outlet_damping")
base.ROOT = ROOT
base.FORCES_DIR = ROOT / "postProcessing/forces"
LOG = ROOT / "solver_damping.log"
CONTROL = ROOT / "system/controlDict"
HISTORY = ROOT / "damping_history.jsonl"
SUMMARY = ROOT / "damping_summary.json"
EXIT = ROOT / "solver_damping.exit"
MAX_WALL_S = 4200


def stop(reason: str) -> None:
    content = CONTROL.read_text()
    if "stopAt endTime;" in content:
        CONTROL.write_text(content.replace("stopAt endTime;", "stopAt writeNow;"))
    print(f"Requested stop: {reason}", flush=True)


def main() -> None:
    start = time.monotonic()
    last_bucket = 30  # final bounded continuation from the 15.217887 s checkpoint
    prior_accepted = False
    records = []
    reason = None
    baseline = [json.loads(line) for line in
                (ROOT.parent / "static_drift_plus6_wall_ramped/extension_history.jsonl")
                .read_text().splitlines() if line.strip()]
    while True:
        wall = time.monotonic() - start
        log = LOG.read_text() if LOG.exists() else ""
        t, physical, resisting = base.force_history()
        if len(t):
            now = float(t[-1])
            bucket = int(now / 0.5)
            if bucket > last_bucket:
                entry, accepted = base.snapshot(t, physical, resisting, log)
                mask = t >= now - 1.0
                for name, axis in (("X", 0), ("Y", 1), ("N", 5)):
                    entry["channels"][name]["peak_to_peak"] = float(np.ptp(physical[mask, axis]))
                    entry["channels"][name]["standard_deviation"] = float(np.std(physical[mask, axis]))
                entry["damping_wall_s"] = wall
                with HISTORY.open("a") as out:
                    out.write(json.dumps(entry) + "\n")
                records.append(entry)
                print(json.dumps({
                    "time_s": now,
                    "Y_mean_slope_amplitude": [entry["channels"]["Y"][k]
                                               for k in ("mean", "slope_per_s", "peak_to_peak")],
                    "N_mean_slope_amplitude": [entry["channels"]["N"][k]
                                               for k in ("mean", "slope_per_s", "peak_to_peak")],
                    "stationary": accepted,
                }), flush=True)
                last_bucket = bucket
                if accepted and prior_accepted:
                    reason = "qualified_X_Y_N_stationary_tail"
                    stop(reason)
                prior_accepted = accepted

                if now >= 15.0 and not reason:
                    old = [r for r in baseline if 7.0 <= r["time_s"] <= 10.05]
                    new = [r for r in records if 12.5 <= r["time_s"]]
                    if len(new) >= 5:
                        ratios = {}
                        for name in ("Y", "N"):
                            old_amp = np.median([r["channels"][name]["peak_to_peak"] for r in old])
                            new_amp = np.median([r["channels"][name]["peak_to_peak"] for r in new])
                            ratios[name] = float(new_amp / old_amp)
                        print(f"Two-period amplitude ratios: {ratios}", flush=True)
                        if all(ratio >= 0.75 for ratio in ratios.values()):
                            reason = "no_material_amplitude_decay_after_two_periods"
                            stop(reason)

        dt = base.last(r"deltaT = ([\d.eE+-]+)", log)
        if not reason and (re.search(r"\bnan\b|\binf\b|FOAM FATAL", log, re.I)
                           or (dt is not None and dt < 1e-7)):
            reason = "numerical_failure"
            stop(reason)
        if not reason and wall >= MAX_WALL_S:
            reason = "wall_time_limit"
            stop(reason)
        if EXIT.exists() or "\nEnd\n" in log or reason:
            SUMMARY.write_text(json.dumps({
                "reason": reason or "solver_ended_or_failed",
                "last_snapshot": records[-1] if records else None,
                "damping_wall_s": wall,
            }, indent=2) + "\n")
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
