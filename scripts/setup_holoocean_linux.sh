#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'HoloOcean Ocean package requires a Linux x86_64 host with OpenGL.' >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_ref="${HOLOOCEAN_REF:-v2.3.0}"
install_root="${HOLOOCEAN_INSTALL_ROOT:-$repo_root/.benchmark-deps/holoocean-linux}"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python 3.12+ is required by bcod-sim; set PYTHON_BIN to Python 3.12+")
PY

sudo apt-get update
sudo apt-get install -y build-essential git python3-venv libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
"$python_bin" -m venv "$install_root/venv"
"$install_root/venv/bin/python" -m pip install --upgrade pip
if [[ ! -d "$install_root/source/.git" ]]; then
  git clone https://github.com/byu-holoocean/HoloOcean.git "$install_root/source"
fi
git -C "$install_root/source" fetch --tags origin
if ! git -C "$install_root/source" rev-parse --verify "$source_ref^{commit}" >/dev/null 2>&1; then
  echo "HoloOcean ref $source_ref is unavailable; set HOLOOCEAN_REF to a compatible release." >&2
  exit 3
fi
git -C "$install_root/source" checkout --detach "$source_ref"
git -C "$install_root/source" rev-parse HEAD > "$install_root/source-commit.txt"
"$install_root/venv/bin/python" -m pip install "$install_root/source/client" "$repo_root"
"$install_root/venv/bin/python" -c 'import holoocean; assert callable(holoocean.make); holoocean.install("Ocean")'
"$install_root/venv/bin/python" - <<'PY'
import holoocean
from bcod_sim.benchmark.core import Benchmark, generate_scenario
from bcod_sim.benchmark.holoocean_adapter import HoloOceanAdapter
env = Benchmark(HoloOceanAdapter())
try:
    env.reset(generate_scenario(8))
    env.step({f"vessel_{i}": (0.0, 0.0) for i in range(4)})
finally:
    env.close()
print("HoloOcean four-vessel launch and one tick passed")
PY
