#!/usr/bin/env bash
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/bcod-no-mps-${UID:-0}-${SLURM_JOB_ID:-local}}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
py="$root/.venv-linux/bin/python"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to the development checkpoint}"
out="${OUTPUT_DIR:-paper_results/linux_gpu/marl/development-eval}"
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2;exit 2; }
[[ ! -e "$out" || "${1:-}" == --overwrite ]] || { echo "Output exists: $out" >&2;exit 2; }
mkdir -p "$out";exec >> "$out/job.log" 2>&1
printf '%s\n' "$0 $* CHECKPOINT=$checkpoint DEVICE=${DEVICE:-cuda}" > "$out/commands.txt"
"$py" tools/linux_gpu/hardware_manifest.py --output "$out/hardware.json" >/dev/null
"$py" - "$checkpoint" "$out" "${DEVICE:-cuda}" <<'PY'
import json,sys
from pathlib import Path
import torch
from bcod_sim.benchmark_v3.deadline_harness.algorithms import MAPPO
from bcod_sim.benchmark_v3.deadline_harness.__main__ import marl_validation
checkpoint,out,device=sys.argv[1:]
if device.startswith('cuda') and not torch.cuda.is_available():raise SystemExit('CUDA requested but unavailable')
payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
model=MAPPO().to(device);model.load_state_dict(payload['model']);model.eval()
result=marl_validation(model,payload.get('steps',0),episodes=32,seed=81001)
result.update(checkpoint=checkpoint,device=device,split='development',native_gate_pass=result['success_rate']>=.80 and result['collision_rate']<=.20)
Path(out,'development.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
