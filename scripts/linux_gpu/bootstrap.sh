#!/usr/bin/env bash
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root" || exit 1
mkdir -p paper_results/linux_gpu/bootstrap
exec >> paper_results/linux_gpu/bootstrap/bootstrap.log 2>&1
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then echo 'Linux x86_64 required'; exit 2; fi
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" --version || exit 2
command -v nvidia-smi >/dev/null && nvidia-smi || echo 'nvidia-smi unavailable'
"$python_bin" -m venv .venv-linux || exit 2
py=.venv-linux/bin/python
"$py" -m pip install --upgrade 'pip<27' 'setuptools>=68' wheel || exit 2
"$py" -m pip install -e '.[marl,validation,test]' || exit 2
"$py" -c 'import bcod_sim,torch,benchmarl,torchrl,tensordict,pettingzoo,gymnasium; print("core imports OK; CUDA:",torch.cuda.is_available())' || exit 2
if [[ "${INSTALL_PYQUATICUS:-0}" == 1 ]]; then bash scripts/install_pyquaticus_benchmark.sh || echo 'OPTIONAL Pyquaticus install failed'; fi
if [[ "${INSTALL_HOLOOCEAN:-0}" == 1 ]]; then bash scripts/setup_holoocean_linux.sh || echo 'OPTIONAL HoloOcean install failed'; fi
