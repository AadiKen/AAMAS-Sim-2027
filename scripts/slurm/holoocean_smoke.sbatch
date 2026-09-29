#!/usr/bin/env bash
#SBATCH --job-name=bcod-holoocean_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec bash scripts/linux_gpu/run_holoocean_smoke.sh
