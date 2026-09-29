#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
out="${OUTPUT_DIR:-paper_results/linux_gpu/scaling}"
[[ ! -e "$out" || "${1:-}" == --overwrite ]] || { echo "Output exists: $out" >&2;exit 2; }
mkdir -p "$out";exec >> "$out/job.log" 2>&1
exec .venv-linux/bin/python tools/linux_gpu/clean_scaling.py --output-dir "$out" "$@"
