#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
mkdir -p paper_results/linux_gpu/logs
shopt -s nullglob
for log in *-slurm-*.out *-slurm-*.err slurm-*.out slurm-*.err paper_results-preflight-*.out paper_results-preflight-*.err; do
  cp -n "$log" paper_results/linux_gpu/logs/
done
exec .venv-linux/bin/python tools/linux_gpu/package_results.py "$@"
