#!/usr/bin/env bash
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/bcod-no-mps-${UID:-0}-${SLURM_JOB_ID:-local}}"
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
out="${OUTPUT_DIR:-paper_results/linux_gpu/pyquaticus-smoke}"
[[ ! -e "$out" || "${1:-}" == --overwrite ]] || { echo "Output exists: $out" >&2;exit 2; }
mkdir -p "$out";exec >> "$out/job.log" 2>&1
"$root/.venv-linux/bin/python" tools/linux_gpu/hardware_manifest.py --output "$out/hardware.json" >/dev/null
printf '%s\n' "$0 $*" > "$out/commands.txt"
py="${PYQUATICUS_PYTHON:-$root/.venv-pyquaticus/bin/python}"
[[ -x "$py" ]] || bash scripts/install_pyquaticus_benchmark.sh
"$py" - <<'PY'
import json,sys,os,importlib.metadata
from pathlib import Path
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
env=PyQuaticusEnv();obs,info=env.reset(seed=11)
for _ in range(3):
    actions={agent:env.action_space(agent).sample() for agent in env.agents}
    obs,reward,terminated,truncated,info=env.step(actions)
env.close();result={'interpreter':sys.executable,'version':importlib.metadata.version('pyquaticus'),'steps':3};Path(os.environ.get('OUTPUT_DIR','paper_results/linux_gpu/pyquaticus-smoke')).joinpath('status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
PY
