import json
from pathlib import Path

from audit.lib.adapter import ORACLE_PACKS
from audit.run import LEDGER


def test_migration_ledger_covers_every_preserved_probe_exactly_once():
    document=json.loads(LEDGER.read_text()); probes=document["probes"]
    assert len(probes)==178
    assert len({row["probe_id"] for row in probes})==178
    assert all(row["reason"] and row["legacy_claim"] for row in probes)
    assert all(set(row["oracles"]).issubset(ORACLE_PACKS) for row in probes)


def test_only_explicit_external_or_spec_exclusions_can_skip():
    probes=json.loads(LEDGER.read_text())["probes"]
    skipped=[x for x in probes if x["status"] in {"LIVE_GATE_DEFERRED","OUT_OF_SCOPE","NO_ACCEPTANCE_CLAIM"}]
    assert skipped
    assert all(not x["oracles"] for x in skipped)
    assert {x["status"] for x in skipped} == {"LIVE_GATE_DEFERRED","OUT_OF_SCOPE","NO_ACCEPTANCE_CLAIM"}


def test_every_executable_migration_has_a_new_core_oracle():
    probes=json.loads(LEDGER.read_text())["probes"]
    executable=[x for x in probes if x["status"] not in {"LIVE_GATE_DEFERRED","OUT_OF_SCOPE","NO_ACCEPTANCE_CLAIM"}]
    assert all(x["oracles"] for x in executable)
    assert {x["status"] for x in executable} <= {"PORTABLE","REPLACED_BY_NEW_ORACLE"}
