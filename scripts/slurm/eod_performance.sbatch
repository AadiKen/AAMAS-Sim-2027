#!/bin/bash
#SBATCH --job-name=eod-perf
#SBATCH --partition=gpu-csl-cnice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --exclude=codenimbus-001-1
#SBATCH --time=02:30:00
#SBATCH --output=eod-perf-%j.out
#SBATCH --error=eod-perf-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
OUT="$PWD/paper_results/eod_performance"
mkdir -p "$OUT/profile" "$OUT/multiprocess" "$OUT/heterogeneity" "$OUT/figures"
export PATH="$PWD/.venv-linux/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
{
  date -Is
  hostname
  lscpu
  taskset -pc $$ || true
  scontrol show job "$SLURM_JOB_ID" || true
  ps -eo pid,psr,%cpu,%mem,comm --sort=-%cpu | head -30
  python -VV
  python -c 'import torch; print("torch", torch.__version__, "threads", torch.get_num_threads())'
  uptime
} > "$OUT/hardware_runtime.txt" 2>&1

python tools/performance/profile_step.py --steps "${PROFILE_STEPS:-1000}" --warmup 30 --output "$OUT/profile"
python tools/performance/multiprocess.py --steps "${SCALE_STEPS:-500}" --warmup 30 --reps "${REPS:-5}" --counts "${WORKER_COUNTS:-1,4,16,64}" --output "$OUT/multiprocess"
