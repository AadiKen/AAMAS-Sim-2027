#!/usr/bin/env bash
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/bcod-no-mps-${UID:-0}-${SLURM_JOB_ID:-local}}"
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
exec .venv-linux/bin/python tools/linux_gpu/preflight.py "$@"
