#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${BCOD_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/bcod-mss-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$ROOT/tools/mss_6dof_validation.py" "$@"
