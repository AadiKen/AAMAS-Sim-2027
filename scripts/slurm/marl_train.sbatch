#!/usr/bin/env bash
#SBATCH --job-name=bcod-marl_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec bash scripts/linux_gpu/run_marl_train.sh
