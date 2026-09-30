#!/usr/bin/env python3
"""Create additive exclusive subsystem shares from an existing cProfile dump."""
import argparse
import csv
import pstats
from collections import defaultdict
from pathlib import Path


def categorize(filename, name):
    f = filename.replace("\\", "/").lower()
    n = name.lower()
    if "plant6.py" in f or "/dynamics/" in f:
        return "plant6_dynamics"
    if "/collision/" in f:
        return "collision_contact"
    if "/actuators/" in f or "_action_commands" in n:
        return "actuator_action_dispatch"
    if "/sensors/" in f or "_sensor_contexts" in n or "sensor" in n and "scheduler" not in n:
        return "sensors"
    if "scheduler" in f or "scheduler" in n:
        return "scheduler_event_bookkeeping"
    if "_load_terms" in n or "_external" in n or "environment_loads" in f or "/world/" in f:
        return "environment_loads"
    if "/tasks/" in f or "compose_reward" in n or "evaluate" == n:
        return "task_reward"
    if "observation" in f or "_deliver" in n or "_state_snapshot" in n or "pettingzoo_env.py" in f:
        return "observation_construction_adapter"
    if "_grounding_telemetry" in n:
        return "diagnostic_bookkeeping"
    if "~" in filename or "torch" in f:
        if any(token in n for token in ("clone", "new_", "tensor", "stack", "cat", "copy", "zeros", "full")):
            return "tensor_allocation_copy"
        if any(token in n for token in ("item", "isfinite", "allclose", "method 'all'")):
            return "tensor_python_conversion_validation"
        return "tensor_operations_other"
    return "misc_python_framework"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stats", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    stats = pstats.Stats(str(args.stats))
    totals = defaultdict(float)
    for (filename, line, func), (_, _, self_seconds, _, _) in stats.stats.items():
        totals[categorize(filename, func)] += self_seconds
    denom = sum(totals.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subsystem", "exclusive_self_time_s", "share_of_profiled_python_wall_pct"])
        for label, seconds in sorted(totals.items(), key=lambda item: item[1], reverse=True):
            w.writerow([label, f"{seconds:.6f}", f"{100 * seconds / denom:.4f}"])
    print(f"exclusive self time {denom:.6f} seconds; wrote {args.output}")


if __name__ == "__main__":
    main()
