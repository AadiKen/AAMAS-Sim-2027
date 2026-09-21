#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${BCOD_PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$ROOT/.venv/bin/python"; fi
if [[ -z "$PYTHON_BIN" && -x /private/tmp/bcod-phase1-venv/bin/python ]]; then PYTHON_BIN=/private/tmp/bcod-phase1-venv/bin/python; fi
if [[ -z "$PYTHON_BIN" ]]; then PYTHON_BIN=python3; fi

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m bcod_sim.vessel_generation.cfd_smoke "$@"
exit $?
