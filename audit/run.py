"""Execute every portable audit oracle and emit a reproducible JSON report."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from audit.lib.adapter import ORACLE_PACKS, execute_pack


ROOT = Path(__file__).parents[1]
LEDGER = Path(__file__).with_name("legacy_probe_migration.json")


def run_audit(*, python: str=sys.executable, output: Path|None=None) -> dict:
    ledger=json.loads(LEDGER.read_text())
    packs={identity:execute_pack(pack,project_root=ROOT,python=python) for identity,pack in ORACLE_PACKS.items()}
    results=[]
    for probe in ledger["probes"]:
        if probe["status"] in {"LIVE_GATE_DEFERRED", "OUT_OF_SCOPE", "NO_ACCEPTANCE_CLAIM"}:
            verdict="ACCEPTED_SKIP"
        else:
            verdict="PASS" if all(packs[x]["verdict"]=="PASS" for x in probe["oracles"]) else "FAIL"
        results.append({**probe,"verdict":verdict})
    unexpected=[row["probe_id"] for row in results if row["verdict"]=="FAIL"]
    report={"schema_version":1,"generated_at":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"ledger_sha256":hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
        "summary":{"source_probes":len(results),"pass":sum(x["verdict"]=="PASS" for x in results),
            "accepted_skip":sum(x["verdict"]=="ACCEPTED_SKIP" for x in results),
            "unexpected_fail":len(unexpected)},"unexpected_failures":unexpected,
        "oracle_packs":packs,"probe_results":results}
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,sort_keys=True,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=ROOT/"audit"/"reports"/"latest.json")
    args=parser.parse_args(); report=run_audit(output=args.output)
    summary=report["summary"]
    print(f"Phase 13 audit: {summary['pass']} PASS, {summary['accepted_skip']} accepted SKIP, "
          f"{summary['unexpected_fail']} unexpected FAIL")
    raise SystemExit(1 if summary["unexpected_fail"] else 0)


if __name__ == "__main__": main()
