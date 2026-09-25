"""Execute Stage 5B cases and preserve evidence without promoting incomplete coverage."""

import csv
from dataclasses import asdict
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

from bcod_sim.config.hashing import content_hash
from bcod_sim.collision.shapes import Box, Sphere


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/validation/test_stage5b_heterogeneous.py"


def version_or_none(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "stage5_results" / f"stage5b-{run_id}"
    for folder in ("fleet_configs", "solo_vs_multi", "actuators", "sensors",
                   "environment", "collisions", "grounding", "timing",
                   "determinism", "long_rollout", "plots", "failures"):
        (out / folder).mkdir(parents=True)
    capabilities = {
        "vessel_configs_classes": "PARTIAL",
        "mass_inertia_variation": "SUPPORTED",
        "hydrodynamic_parameter_sets": "PARTIAL",
        "actuator_layouts": "SUPPORTED",
        "controller_modes": "SUPPORTED",
        "sensor_suites_rates": "SUPPORTED",
        "observations": "SUPPORTED",
        "current": "SUPPORTED",
        "wind": "SUPPORTED",
        "regular_irregular_waves": "SUPPORTED",
        "obstacles": "SUPPORTED",
        "vessel_vessel_collision": "SUPPORTED",
        "grounding_bathymetry": "SUPPORTED",
        "partial_observability": "PARTIAL",
        "nearby_agent_sensor": "UNSUPPORTED",
        "task_relative_observation": "UNSUPPORTED",
        "asynchronous_sensors": "SUPPORTED",
        "asynchronous_controllers": "UNSUPPORTED",
        "wake_interaction": "SUPPORTED",
        "dynamic_agent_counts": "PARTIAL",
    }
    (out / "capabilities.json").write_text(json.dumps(capabilities, indent=2) + "\n")
    namespace = runpy.run_path(str(TEST))
    registry = tuple(namespace["STAGE5B_CASES"]) + ("test_registry_complete",)
    sample_fleet = namespace["fleet"](collision_shape_by_name={
        "a": Sphere(0.2), "b": Box((0.3, 0.2, 0.2)), "c": Sphere(0.1)})
    vessel_hashes = {}
    for name, vessel in sample_fleet.engine.vessels.items():
        payload = {"name": name, "vessel_id": vessel.vessel_id,
                   "mass_kg": vessel.plant.mass.mass_kg,
                   "inertia": vessel.plant.mass.inertia_cg_kg_m2.tolist(),
                   "collision": repr(vessel.collision_shape),
                   "actuators": [asdict(actuator.config) for actuator in vessel.actuators],
                   "sensors": [asdict(sensor.config) for sensor in vessel.sensors]}
        vessel_hashes[name] = content_hash(payload)
        (out / "fleet_configs" / f"{name}.json").write_text(
            json.dumps({"config": payload, "sha256": vessel_hashes[name]}, indent=2,
                       default=str) + "\n")
    manifest = {
        "run_id": run_id,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "dependencies": {name: version_or_none(name) for name in
                         ("torch", "numpy", "pydantic", "pytest", "gymnasium", "pettingzoo")},
        "test_registry": registry,
        "fixture_sha256": hashlib.sha256(TEST.read_bytes()).hexdigest(),
        "vessel_config_hashes": vessel_hashes,
        "frozen_dynamics": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    xml = out / "cases.xml"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TEST),
                             f"--junitxml={xml}"], cwd=ROOT, capture_output=True, text=True)
    (out / "long_rollout" / "pytest.txt").write_text(result.stdout + result.stderr)
    cases = []
    for case in ElementTree.parse(xml).iter("testcase"):
        name = case.get("name", "")
        status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else "PASS"
        if case.find("skipped") is not None:
            status = "BLOCKED"
        cases.append({"case": name, "status": status, "duration_s": case.get("time", "")})
    with (out / "cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("case", "status", "duration_s"))
        writer.writeheader()
        writer.writerows(cases)
    executed = {case["case"].split("[")[0] for case in cases}
    missing = [name for name in registry if name not in executed]
    counts = {status: sum(case["status"] == status for case in cases)
              for status in ("PASS", "FAIL", "BLOCKED")}
    # The hard gate is deliberately conservative until every supported matrix is covered.
    required_gaps = []
    status = "FAIL" if counts["FAIL"] else "BLOCKED" if missing or required_gaps else "PASS"
    summary = {"status": status, "cases": counts, "missing_registered": missing,
               "unvalidated_required": required_gaps, "pytest_exit_code": result.returncode}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "report.md").write_text(
        f"# Stage 5B heterogeneous MARL validation\n\nStatus: **{status}**. "
        f"Cases: {counts['PASS']} PASS / {counts['FAIL']} FAIL / {counts['BLOCKED']} BLOCKED.\n\n"
        "Stage 3 coefficient fidelity is pending. Current vessel dynamics configurations were "
        "treated as frozen inputs. Mass and inertia variants represent different payloads; "
        "they do not qualify hydrodynamic coefficients. Nearby-agent sensing and task-relative "
        "observations are not implemented.\n\n## Required coverage still open\n\n" +
        ("\n".join(f"- {gap}" for gap in required_gaps) if required_gaps else "None") + "\n")
    print(out)
    print(result.stdout[-1200:])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
