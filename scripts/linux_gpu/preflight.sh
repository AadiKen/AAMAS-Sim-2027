#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
exec .venv-linux/bin/python tools/linux_gpu/preflight.py "$@"
