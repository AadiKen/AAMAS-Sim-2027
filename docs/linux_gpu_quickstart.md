# Linux GPU first run

Requires Linux x86_64, Python 3.12+, a user-writable clone, NVIDIA driver for CUDA jobs, and working package index access. No sudo is used. MANTA physics run on CPU; policy training can run on CUDA. HoloOcean uses a separate Linux graphics engine.

## Inside an interactive GPU allocation

```bash
bash scripts/linux_gpu/bootstrap.sh
bash scripts/linux_gpu/preflight.sh --require-cuda
bash scripts/linux_gpu/run_clean_scaling.sh
bash scripts/linux_gpu/run_marl_smoke.sh
SEED=11 STEPS=400000 DEVICE=cuda bash scripts/linux_gpu/run_marl_train.sh
bash scripts/linux_gpu/run_sarl_smoke.sh
SEED=11 STEPS=400000 DEVICE=cuda bash scripts/linux_gpu/run_sarl_train.sh
bash scripts/linux_gpu/run_pyquaticus_smoke.sh
bash scripts/linux_gpu/run_holoocean_smoke.sh
bash scripts/linux_gpu/run_transfer_smoke.sh
bash scripts/linux_gpu/collect_results.sh
```

The deadline harness deliberately gates training on existing BC and task preflight artifacts. Provide `BC_PATH`, `PREFLIGHT_PATH`, `REPLAY_PATH` (SARL), and `SMOKE_REPORT` as needed. Produce these via `python -m bcod_sim.benchmark_v3.deadline_harness --help`. Output directories are protected; set `OUTPUT_DIR` per run or pass `--overwrite`. The transfer smoke only validates native nominal behavior and reports that foreign deadline adapters remain unimplemented; it makes no performance claim.

## From a SLURM login node

Bootstrap in an interactive allocation first, then submit:

```bash
sbatch --partition=YOUR_GPU_PARTITION --time=02:30:00 scripts/slurm/scaling.sbatch
sbatch --partition=YOUR_GPU_PARTITION --time=12:00:00 scripts/slurm/marl_train.sbatch
sbatch --partition=YOUR_GPU_PARTITION --time=12:00:00 scripts/slurm/sarl_train.sbatch
sbatch --partition=YOUR_GPU_PARTITION scripts/slurm/holoocean_smoke.sbatch
sbatch --partition=YOUR_GPU_PARTITION scripts/slurm/transfer_smoke.sbatch
```

SLURM stdout/stderr use `slurm-%j.*` in the submit directory. Run logs and metadata are also written to each output directory. HoloOcean needs a working OpenGL display or `xvfb-run` installed by the site. Set `INSTALL_HOLOOCEAN=1` or `INSTALL_PYQUATICUS=1` on bootstrap to attempt optional installations; failures are isolated. Pyquaticus uses its pinned separate Python 3.10 environment; `PYQUATICUS_PYTHON` may override its interpreter.
