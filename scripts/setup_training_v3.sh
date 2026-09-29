#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${TRAINING_V3_PYTHON:-python3.13}"
venv_dir="$repo_dir/.venv-training-v3"
lock_file="$repo_dir/configs/benchmark_v3/requirements.lock"

if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-training-v3}"
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    uv venv "$venv_dir" --python "$python_bin"
  fi
  uv pip install --python "$venv_dir/bin/python" -r "$lock_file"
  uv pip install --python "$venv_dir/bin/python" -e . --no-deps --no-build-isolation
else
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$python_bin" -m venv "$venv_dir"
  fi
  "$venv_dir/bin/python" -m pip install -r "$lock_file"
  "$venv_dir/bin/python" -m pip install -e . --no-deps --no-build-isolation
fi

"$venv_dir/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys
print("Interpreter:", sys.executable)
print("Python:", sys.version.split()[0])
for name in ("bcod-sim", "stable-baselines3", "gymnasium", "torch", "numpy", "PyYAML", "pytest"):
    print(f"{name}: {metadata.version(name)}")
PY
