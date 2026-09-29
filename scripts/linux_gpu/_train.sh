#!/usr/bin/env bash
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/bcod-no-mps-${UID:-0}-${SLURM_JOB_ID:-local}}"
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
policy="$1"; mode="$2"; shift 2
py="$root/.venv-linux/bin/python"
[[ -x "$py" ]] || { echo 'Run bootstrap first' >&2; exit 2; }
if [[ -z "${DEVICE:-}" ]]; then DEVICE="$($py -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')"; fi
seed="${SEED:-11}";steps="${STEPS:-$([[ $mode == smoke ]] && echo 5000 || echo 400000)}"
outdir="${OUTPUT_DIR:-$root/paper_results/linux_gpu/$policy/$mode-seed-$seed}"
if [[ -e "$outdir" && "${1:-}" != --overwrite ]]; then echo "Output exists: $outdir (pass --overwrite to allow)" >&2; exit 2; fi
mkdir -p "$outdir";exec >> "$outdir/job.log" 2>&1
printf '%s\n' "$0 $* DEVICE=$DEVICE SEED=$seed STEPS=$steps INIT_CHECKPOINT=${INIT_CHECKPOINT:-}" > "$outdir/commands.txt"
"$py" tools/linux_gpu/hardware_manifest.py --output "$outdir/hardware.json" >/dev/null
"$py" tools/linux_gpu/preflight.py --output "$outdir/preflight.json" $([[ "$DEVICE" == cuda* ]] && echo --require-cuda) || exit 2
"$py" - "$outdir" "$policy" "$mode" "$DEVICE" "$seed" "$steps" <<'PY'
import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);p.joinpath('manifest.json').write_text(json.dumps(dict(policy=sys.argv[2],mode=sys.argv[3],policy_training_device=sys.argv[4],simulator_physics_device='CPU',seed=int(sys.argv[5]),steps=int(sys.argv[6]),initialization_checkpoint=os.environ.get("INIT_CHECKPOINT"),initialization_semantics="weights_only_optimizer_reset" if os.environ.get("INIT_CHECKPOINT") else "behavior_cloning",scenario_mixture=os.environ.get("MARL_SCENARIO_MIXTURE","mixture") if sys.argv[2]=="marl" else None),indent=2)+'\n')
PY
args=("$policy-$mode" --steps "$steps" --seed "$seed" --device "$DEVICE" --output "$outdir/$policy.pt")
if [[ "$policy" == marl ]]; then
 args+=(--preflight "${PREFLIGHT_PATH:-runs/deadline/marl-preflight.json}" --bc "${BC_PATH:-runs/deadline/marl-bc.pt}")
else
 args+=(--preflight "${PREFLIGHT_PATH:-runs/deadline/sarl-preflight.json}" --bc "${BC_PATH:-runs/deadline/sarl-bc.pt}" --replay-data "${REPLAY_PATH:-runs/deadline/sarl-pid.npz}")
fi
if [[ "$mode" == train ]]; then
 args+=(--smoke-report "${SMOKE_REPORT:-runs/deadline/$policy-smoke.pt.smoke.json}")
 if [[ -n "${INIT_CHECKPOINT:-}" ]]; then args+=(--init-checkpoint "$INIT_CHECKPOINT"); fi
 if [[ "$policy" == marl ]]; then args+=(--scenario-mixture "${MARL_SCENARIO_MIXTURE:-mixture}"); fi
fi
"$py" -m bcod_sim.benchmark_v3.deadline_harness "${args[@]}"
