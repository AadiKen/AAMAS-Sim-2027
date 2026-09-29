#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)";cd "$root"
source_csv="${SOURCE_CSV:-paper_results/linux_gpu/scaling-h100-high-20260929/scaling.csv}"
out="${OUTPUT_DIR:-paper_results/linux_gpu/scaling-8192-continuation}"
mkdir -p "$out";exec >> "$out/continuation.log" 2>&1
completed="$(python3 - "$source_csv" <<'PY'
import csv,sys
with open(sys.argv[1],newline='') as stream:
 print(sum(row['count']=='8192' for row in csv.DictReader(stream)))
PY
)"
if (( completed >= 5 )); then echo "8192 already has $completed repetitions";exit 0;fi
needed=$((5-completed))
echo "Continuing 8192 from repetition $completed for $needed repetitions"
OUTPUT_DIR="$out/data" bash scripts/linux_gpu/run_clean_scaling.sh --counts 8192 --repetitions "$needed" --rep-offset "$completed"
