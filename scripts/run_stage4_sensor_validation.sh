#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
mkdir -p stage4_results/stage4-2026-09-22/logs
.venv/bin/python -m pytest -q -rxX -o junit_family=xunit1 \
  tests/validation/test_stage4_remediation.py \
  tests/validation/test_stage4_sensor_validation.py \
  tests/sensors/test_sdk_completion.py \
  tests/sensors/test_contracts.py \
  tests/data_sources/test_phase9_adapters.py \
  --junitxml=stage4_results/stage4-2026-09-22/logs/campaign-junit.xml
.venv/bin/python -m pytest -q \
  --junitxml=stage4_results/stage4-2026-09-22/logs/full-suite-junit.xml
