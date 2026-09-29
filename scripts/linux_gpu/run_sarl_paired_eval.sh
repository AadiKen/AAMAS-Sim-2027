#!/usr/bin/env bash
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/bcod-no-mps-${UID:-0}-${SLURM_JOB_ID:-local}}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
py="$root/.venv-linux/bin/python"
out="${OUTPUT_DIR:-paper_results/linux_gpu/sarl/paired-eval-seed-11}"
checkpoint="${CHECKPOINT:-paper_results/linux_gpu/sarl/train-10k-seed-11/sarl.pt.best-tracking.pt}"
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2;exit 2; }
[[ ! -e "$out" || "${1:-}" == --overwrite ]] || { echo "Output exists: $out" >&2;exit 2; }
mkdir -p "$out";exec >> "$out/job.log" 2>&1
printf '%s\n' "$0 $* CHECKPOINT=$checkpoint EPISODES=${EPISODES:-24} STEPS=${EVAL_STEPS:-500}" > "$out/commands.txt"
"$py" tools/linux_gpu/hardware_manifest.py --output "$out/hardware.json" >/dev/null
"$py" tools/linux_gpu/preflight.py --output "$out/preflight.json" --require-cuda
"$py" -m bcod_sim.benchmark_v3.deadline_harness sarl-eval --device cuda --checkpoint "$checkpoint" --output "$out/evaluation" --episodes "${EPISODES:-24}" --steps "${EVAL_STEPS:-500}" --seed "${EVAL_SEED:-44001}"
