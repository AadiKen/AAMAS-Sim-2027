#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${repo_root}/.venv/bin/python" "${repo_root}/tools/stage5b_heterogeneous_validation.py"
