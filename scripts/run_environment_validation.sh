#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${BCOD_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
PYTHONPATH="$ROOT/src:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$ROOT/tools/environment_validation.py" "$@"
