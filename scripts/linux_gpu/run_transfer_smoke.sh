#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
out="${OUTPUT_DIR:-paper_results/linux_gpu/transfer-smoke}"
[[ ! -e "$out" || "${1:-}" == --overwrite ]] || { echo "Output exists: $out" >&2;exit 2; }
mkdir -p "$out";exec >> "$out/job.log" 2>&1
"${root}/.venv-linux/bin/python" - <<'PY' | tee "$out/status.json"
import json
from bcod_sim.benchmark_v3.deadline_harness.marl import ResidualCoordinatorEnv,make_four_vessel_scenario,GeometricCoordinator
import numpy as np
e=ResidualCoordinatorEnv(scenario=make_four_vessel_scenario(np.random.default_rng(11),"easy"));obs,_=e.reset(seed=11)
for _ in range(3):
 obs,_,_,_,_=e.step(GeometricCoordinator().actions(e))
assert all(np.isfinite(x).all() for x in obs.values());e.close()
print(json.dumps({'native_nominal_steps':3,'foreign_deadline_adapters':'NOT_IMPLEMENTED','transfer_performance_claim':False}))
PY
