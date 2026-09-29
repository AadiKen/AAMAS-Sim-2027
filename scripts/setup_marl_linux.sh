#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${MARL_PYTHON:-python3.12}"
venv_dir="${MARL_VENV:-.venv-marl}"
"$python_bin" -m venv "$venv_dir"

if [[ -n "${MARL_WHEELHOUSE:-}" ]]; then
  "$venv_dir/bin/python" -m pip install --no-index --find-links "$MARL_WHEELHOUSE" \
    -r configs/marl/requirements-lock.txt
  "$venv_dir/bin/python" -m pip install --no-index --find-links "$MARL_WHEELHOUSE" \
    --no-deps --no-build-isolation -e .
else
  "$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel
  "$venv_dir/bin/python" -m pip install -r configs/marl/requirements-lock.txt
  "$venv_dir/bin/python" -m pip install --no-deps -e .
fi

"$venv_dir/bin/python" - <<'PY'
import torch, tensordict, torchrl, benchmarl, pettingzoo, gymnasium
from benchmarl.algorithms import MappoConfig
print({m.__name__: m.__version__ for m in
       (torch, tensordict, torchrl, benchmarl, pettingzoo, gymnasium)})
print("MAPPO:", MappoConfig)
PY
