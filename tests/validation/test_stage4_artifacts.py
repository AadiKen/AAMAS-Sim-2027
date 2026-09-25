import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT/"stage4_results"/"stage4-2026-09-22"


def test_stage4_evidence_is_identified_and_fail_closed():
    manifest = json.loads((RESULT/"manifest.json").read_text())
    report = (RESULT/"report.md").read_text()
    assert manifest["baseline_dirty"] is True
    assert len(manifest["git_commit"]) == 40 and len(manifest["working_tree_diff_sha256"]) == 64
    assert manifest["campaign"]["failed"] == 0
    assert manifest["campaign"]["blocked"] == 0
    assert manifest["campaign"]["passed"] >= 67
    assert "**STAGE 4 PASS**" in report
    assert all(f"{name} | PASS" in report for name in ("SENS-023", "SENS-036", "SENS-080", "SENS-081"))
    assert all((RESULT/manifest[name]).is_file() for name in ("campaign_junit", "full_suite_junit"))
