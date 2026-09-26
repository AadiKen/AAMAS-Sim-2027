"""Bounded continuation monitor for the saved HMRI +6 degree checkpoint."""
from __future__ import annotations

import json
import re
import time

import numpy as np

from monitor_kcs_hmri_plus6 import ROOT, force_history, last, snapshot

LOG = ROOT / "solver_extension.log"
CONTROL = ROOT / "system/controlDict"
HISTORY = ROOT / "extension_history.jsonl"
SUMMARY = ROOT / "extension_summary.json"
EXIT = ROOT / "solver_extension.exit"
MAX_WALL_S = 3600


def request_stop(reason: str) -> None:
    content = CONTROL.read_text()
    if "stopAt endTime;" in content:
        CONTROL.write_text(content.replace("stopAt endTime;", "stopAt writeNow;"))
    print(f"Requested stop: {reason}", flush=True)


def main() -> None:
    start = time.monotonic()
    last_bucket = 12  # first new half-second bucket follows 6.265341 s
    prior_accepted = False
    records = []
    reason = None
    while True:
        wall_s = time.monotonic() - start
        log = LOG.read_text() if LOG.exists() else ""
        times, physical, resisting = force_history()
        if len(times):
            now = float(times[-1])
            bucket = int(now / 0.5)
            if bucket > last_bucket:
                entry, accepted = snapshot(times, physical, resisting, log)
                mask = times >= now - 1.0
                for name, axis in (("X", 0), ("Y", 1), ("N", 5)):
                    entry["channels"][name]["peak_to_peak"] = float(np.ptp(physical[mask, axis]))
                    entry["channels"][name]["standard_deviation"] = float(
                        np.std(physical[mask, axis]))
                entry["extension_wall_s"] = wall_s
                HISTORY.open("a").write(json.dumps(entry) + "\n")
                records.append(entry)
                print(json.dumps({
                    "time_s": now,
                    "Y_mean_slope": [entry["channels"]["Y"][key] for key in ("mean", "slope_per_s")],
                    "N_mean_slope": [entry["channels"]["N"][key] for key in ("mean", "slope_per_s")],
                    "stationary": accepted,
                }), flush=True)
                last_bucket = bucket
                if accepted and prior_accepted:
                    reason = "qualified_X_Y_N_stationary_tail"
                    request_stop(reason)
                prior_accepted = accepted

                if now >= 10 and not accepted and not reason:
                    early = [r for r in records if 6.5 <= r["time_s"] <= 8]
                    recent = [r for r in records if r["time_s"] >= 8.5]
                    if len(early) >= 2 and len(recent) >= 3:
                        no_decay = []
                        for name in ("Y", "N"):
                            old = np.median([abs(r["channels"][name]["slope_per_s"]) for r in early])
                            new = np.median([abs(r["channels"][name]["slope_per_s"]) for r in recent])
                            old_amp = np.median([r["channels"][name]["peak_to_peak"] for r in early])
                            new_amp = np.median([r["channels"][name]["peak_to_peak"] for r in recent])
                            no_decay.append(new >= 0.7*old and new_amp >= 0.7*old_amp)
                        if any(no_decay):
                            reason = "no_material_slope_or_amplitude_decay_by_10s"
                            request_stop(reason)

        dt = last(r"deltaT = ([\d.eE+-]+)", log)
        omega_max = last(r"bounding omega, min: [^\n]* max: ([\d.eE+-]+)", log)
        k_max = last(r"bounding k, min: [^\n]* max: ([\d.eE+-]+)", log)
        if not reason and (re.search(r"\bnan\b|\binf\b", log, re.I)
                           or (dt is not None and dt < 1e-7)
                           or (omega_max is not None and omega_max > 1e7)
                           or (k_max is not None and k_max > 1e4)):
            reason = "numerical_failure"
            request_stop(reason)
        if not reason and wall_s >= MAX_WALL_S:
            reason = "extension_wall_time_limit"
            request_stop(reason)
        if EXIT.exists() or "\nEnd\n" in log or "FOAM FATAL" in log or reason:
            SUMMARY.write_text(json.dumps({
                "reason": reason or "solver_ended_or_failed",
                "last_snapshot": records[-1] if records else None,
                "extension_wall_s": wall_s,
            }, indent=2) + "\n")
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
