#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${repo_root}/.benchmark-deps/pyquaticus-src"
venv_root="${repo_root}/.venv-pyquaticus"
commit="72b50e067ab311929390ecd4e59131452be15c6d"

mkdir -p "${repo_root}/.benchmark-deps"
if [[ ! -d "${source_root}/.git" ]]; then
    git init -q "${source_root}"
    git -C "${source_root}" fetch --depth 1 https://github.com/mit-ll-trusted-autonomy/pyquaticus.git "${commit}"
    git -C "${source_root}" checkout -q FETCH_HEAD
fi
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${commit}" ]]; then
    echo "Pyquaticus source commit mismatch" >&2
    exit 1
fi

uv python install 3.10
if [[ ! -x "${venv_root}/bin/python" ]]; then
    uv venv "${venv_root}" --python 3.10
fi
uv pip install --python "${venv_root}/bin/python" -r "${repo_root}/configs/pyquaticus-requirements.lock"
# pymoos is a bridge to physical MOOS hardware, unused by this benchmark, and
# has no matching macOS ARM wheel. The local simulator is installed without it.
uv pip install --python "${venv_root}/bin/python" --no-deps "${source_root}"
"${venv_root}/bin/python" -c 'from pyquaticus.envs.pyquaticus import PyQuaticusEnv; print("Pyquaticus import OK")'
