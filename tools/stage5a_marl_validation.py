"""Run the executable Stage 5A registry and write an auditable result bundle."""

import csv
import hashlib
import json
import platform
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/validation/test_stage5a_marl.py"
GROUPS = {
    "pettingzoo": "API", "one_agent": "TERMINATION", "all_agents": "TERMINATION",
    "mixed_termination": "TRUNCATION", "heterogeneous": "ACTION_MAPPING",
    "cross_agent": "STATE_LEAKAGE", "physics_event": "PHYSICS_EVENT",
    "goal_completion": "PHYSICS_EVENT", "seeded_random": "RNG",
    "sensor_dropout": "SENSOR_INTEGRATION", "dropout_with": "SENSOR_INTEGRATION",
    "duplicate_agent": "ERROR_HANDLING", "reset": "RESET",
    "agent_order": "AGENT_IDENTITY", "reward": "REWARD",
    "team_and": "REWARD", "vector_environment": "API",
    "terminal": "TRUNCATION", "missing_extra": "ERROR_HANDLING",
    "sensor_owner": "SENSOR_INTEGRATION", "invalid_thrust": "ERROR_HANDLING",
    "long_rollout": "DETERMINISM", "registry": "API",
}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "stage5_results" / f"stage5a-{run_id}"
    for name in ("api", "reset", "actions", "observations", "rewards",
                 "termination", "determinism", "isolation", "sensors", "events",
                 "long_rollout", "failures"):
        (out / name).mkdir(parents=True)
    registry = tuple(runpy.run_path(str(TEST))["STAGE5A_CASES"]) + ("test_registry_complete",)
    manifest = {
        "run_id": run_id,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True).strip(),
        "python": platform.python_version(),
        "dependencies": {n: package_version(n) for n in
                         ("torch", "numpy", "pydantic", "pytest", "gymnasium", "pettingzoo")},
        "api_type": "PettingZoo ParallelEnv with explicit Gymnasium spaces and native typed codecs",
        "seeds": [3, 5, 7, 9, 10, 13, 17, 21, 31, 41, 73, 77, 111, 112],
        "config_hashes": {"stage5a_fixture_sha256": hashlib.sha256(TEST.read_bytes()).hexdigest()},
        "test_registry": registry,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    xml = out / "api" / "pytest.xml"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TEST),
                             f"--junitxml={xml}"], cwd=ROOT, capture_output=True, text=True)
    (out / "api" / "pytest.txt").write_text(result.stdout + result.stderr)
    rows = []
    for case in ElementTree.parse(xml).iter("testcase"):
        name = case.get("name", "")
        matched = next((key for key in GROUPS if name.startswith("test_" + key)), None)
        status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else "PASS"
        if case.find("skipped") is not None:
            status = "BLOCKED"
        rows.append({"case": name, "category": GROUPS.get(matched, "UNKNOWN"),
                     "status": status, "duration_s": case.get("time", "")})
    with (out / "cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("case", "category", "status", "duration_s"))
        writer.writeheader()
        writer.writerows(rows)
    executed = {row["case"].split("[")[0] for row in rows}
    gaps = [f"Registered case did not execute: {case}" for case in registry
            if case not in executed]
    if not package_version("pettingzoo") or not package_version("gymnasium"):
        gaps.append("Official PettingZoo validation dependency unavailable")
    counts = {state: sum(row["status"] == state for row in rows)
              for state in ("PASS", "FAIL", "BLOCKED")}
    status = "FAIL" if counts["FAIL"] or gaps else "PASS"
    summary = {"status": status, "cases": counts, "pytest_exit_code": result.returncode,
               "unvalidated_requirements": gaps, "production_bugs_fixed": [
                   "Terminal transition returns final observations and rejects post-episode step.",
                   "Parallel adapter exposes per-agent termination and sensor freshness metadata.",
                   "Parallel adapter validates explicit spaces and supports typed action/observation codecs."],
               "validation_harness_bugs_fixed": [
                   "Invalid thrust assertion now expects CommandBoundsError."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "report.md").write_text(
        f"# Stage 5A validation\n\nStatus: **{status}**. "
        f"Executable cases: {counts['PASS']} pass, {counts['FAIL']} fail, "
        f"{counts['BLOCKED']} blocked.\n\n"
        "The executable registry covers official PettingZoo parallel and seed tests; "
        "heterogeneous spaces, sensors and actuators; independent and mixed endings; "
        "cross-agent isolation; collision, grounding and goal logging; reset, RNG, "
        "reward accounting, and a four-agent 10,000-step replay.\n\n"
        "Collision contacts are nonterminal and affect reward only through configured task "
        "progress. Sensor dropout uses an explicit validity field where a fixed observation "
        "space is required.\n\n"
        "## Unvalidated requirements\n\n" +
        ("\n".join(f"- {gap}" for gap in gaps) if gaps else "None") + "\n")
    print(out)
    print(result.stdout[-1000:])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
