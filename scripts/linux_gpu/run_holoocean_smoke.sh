#!/usr/bin/env bash
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/bcod-no-mps-${UID:-0}-${SLURM_JOB_ID:-local}}"
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
out="${OUTPUT_DIR:-paper_results/linux_gpu/holoocean-smoke}"
[[ ! -e "$out" || "${1:-}" == --overwrite ]] || { echo "Output exists: $out" >&2;exit 2; }
mkdir -p "$out";exec >> "$out/job.log" 2>&1
"$root/.venv-linux/bin/python" tools/linux_gpu/hardware_manifest.py --output "$out/hardware.json" >/dev/null
printf '%s\n' "$0 $*" > "$out/commands.txt"
[[ "$(uname -s)" == Linux ]] || { echo 'REQUIRES_LINUX_VERIFICATION';exit 2; }
[[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" || -x "$(command -v xvfb-run || true)" ]] || { echo 'X display or xvfb-run required';exit 2; }
py="${HOLOOCEAN_PYTHON:-$root/.benchmark-deps/holoocean-linux/venv/bin/python}"
[[ -x "$py" ]] || bash scripts/setup_holoocean_linux.sh
runner=("$py")
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then runner=(xvfb-run -a "$py"); fi
"${runner[@]}" - <<'PY'
from bcod_sim.benchmark.core import Benchmark,generate_scenario
from bcod_sim.benchmark.holoocean_adapter import HoloOceanAdapter
e=Benchmark(HoloOceanAdapter())
try:
 e.reset(generate_scenario(8))
 for _ in range(3):e.step({f'vessel_{i}':(0.,0.) for i in range(4)})
 print('HoloOcean reset and 3 steps passed')
finally:e.close()
PY
