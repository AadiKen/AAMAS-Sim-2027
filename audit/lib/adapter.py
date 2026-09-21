"""Adapter from preserved audit claims to executable new-core oracle packs.

The legacy TypeScript adapter reached into implementation-specific classes. This
adapter invokes only public new-core tests and returns structured evidence.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess


@dataclass(frozen=True)
class OraclePack:
    id: str
    description: str
    selectors: tuple[str, ...]


ORACLE_PACKS = {
    "config-provenance": OraclePack("config-provenance", "closed schemas, identities, hashes, and manifests",
        ("tests/unit/test_config.py", "tests/integration/test_phase10_artifacts_policy.py")),
    "physics-analytic": OraclePack("physics-analytic", "6-DOF analytic, actuator, world, contact, and wake anchors",
        ("tests/physics",)),
    "frames-units": OraclePack("frames-units", "frame, datum, unit, and transform properties",
        ("tests/property/test_frames.py",)),
    "sensor-contracts": OraclePack("sensor-contracts", "sensor mounts, rates, noise, range, FOV, and contracts",
        ("tests/sensors/test_contracts.py",)),
    "external-fixtures": OraclePack("external-fixtures", "offline checksummed source fixtures and coverage failures",
        ("tests/data_sources/test_phase9_adapters.py",)),
    "lifecycle-replay": OraclePack("lifecycle-replay", "episode reset, checkpoint replay, ordering, and termination",
        ("tests/integration/test_episode_phase7.py",)),
    "batch-determinism": OraclePack("batch-determinism", "batch isolation and order-independent reductions",
        ("tests/integration/test_batching_phase8.py",)),
    "web-headless": OraclePack("web-headless", "web export and authoritative headless equivalence",
        ("tests/web/test_phase11_web.py",)),
    "vessel-identification": OraclePack("vessel-identification", "generated vessels, identification, and calibration",
        ("tests/vessel_generation/test_phase12.py",)),
    "legacy-isolation": OraclePack("legacy-isolation", "production source contains no legacy imports",
        ("tests/test_legacy_isolation.py",)),
}


def execute_pack(pack: OraclePack, *, project_root: Path, python: str) -> dict:
    command = [python, "-m", "pytest", "-q", *pack.selectors]
    completed = subprocess.run(command, cwd=project_root, capture_output=True, text=True,
                               env={**os.environ,"PYTHONPATH": str(project_root/"src")})
    return {"id": pack.id, "description": pack.description,
            "verdict": "PASS" if completed.returncode == 0 else "FAIL",
            "exit_code": completed.returncode, "selectors": list(pack.selectors),
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
