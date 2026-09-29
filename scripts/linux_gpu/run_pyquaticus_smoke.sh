#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
out="${OUTPUT_DIR:-paper_results/linux_gpu/pyquaticus-smoke}"
[[ ! -e "$out" || "${1:-}" == --overwrite ]] || { echo "Output exists: $out" >&2;exit 2; }
mkdir -p "$out";exec >> "$out/job.log" 2>&1
py="${PYQUATICUS_PYTHON:-$root/.venv-pyquaticus/bin/python}"
[[ -x "$py" ]] || bash scripts/install_pyquaticus_benchmark.sh
"$py" - <<'PY'
import json,sys,importlib.metadata
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
env=PyQuaticusEnv();obs,info=env.reset(seed=11)
for _ in range(3):
    actions={agent:env.action_space(agent).sample() for agent in env.agents}
    obs,reward,terminated,truncated,info=env.step(actions)
env.close();print(json.dumps({'interpreter':sys.executable,'version':importlib.metadata.version('pyquaticus'),'steps':3}))
PY
