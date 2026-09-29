# Local M0 BenchMARL dependency stack

Platform: macOS arm64, Python 3.12.4. Exact platform text and versions are saved in `artifacts/marl-local-setup/platform.txt` and `versions.json`. Dependencies are isolated in `.venv-marl`; the existing project virtual environments were not modified.

| Package | Resolved version |
|---|---:|
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| tensordict | 0.11.0 |
| torchrl | 0.11.1 |
| benchmarl | 1.5.2 |
| pettingzoo | 1.24.3 |
| gymnasium | 1.3.0 |
| hydra-core | 1.3.7 |

The initial package resolution selected PettingZoo 1.25.0. TorchRL 0.11.1 warns that its PettingZoo wrapper is tested against 1.24.3, so the local environment was pinned to 1.24.3 before adapter qualification. The complete resolved package list is `configs/marl/requirements-lock.txt`.

## Local installation

```bash
/usr/local/bin/python3.12 -m venv .venv-marl
.venv-marl/bin/python -m pip install --upgrade pip setuptools wheel
.venv-marl/bin/python -m pip install -r configs/marl/requirements-lock.txt
.venv-marl/bin/python -m pip install -e . --no-deps
```

BenchMARL 1.5.2 exposes `benchmarl.algorithms.MappoConfig`, `benchmarl.experiment.Experiment`, `ExperimentConfig`, and `benchmarl.models.MlpConfig`. The custom task extends its installed `PettingZooClass(TaskClass)` API. TorchRL's supported `PettingZooWrapper(env=..., return_state=True, group_map=...)` converts the V3 ParallelEnv; the local actor spec is `[2,78]` and the separate `state` critic spec is `[89]`.

TorchRL 0.11.1 tensorizes PettingZoo info dictionaries. V3 infos contain provenance strings, so the thin `TensorInfoNavigationEnv` integration hides those fields from TorchRL and retains them as `last_v3_infos`; the V3 task, reward, actions, and observations remain authoritative. BenchMARL owns collection, GAE, clipped PPO loss, optimizer steps, and minibatching.

The local execution sandbox prevents `torch_shm_manager` from creating shared memory. BenchMARL training therefore requires a permitted unsandboxed local command. This is an execution permission issue, not a macOS ARM package incompatibility. M0 kinematic smoke successfully ran locally with the installed stack.

## Linux and offline cluster preparation

Run `scripts/setup_marl_linux.sh` on an internet-connected Linux host with Python 3.12. For an offline Linux node, first build a wheelhouse on a matching Linux architecture and Python version:

```bash
python3.12 -m venv /tmp/marl-wheel-build
/tmp/marl-wheel-build/bin/python -m pip install --upgrade pip setuptools wheel
/tmp/marl-wheel-build/bin/python -m pip wheel -r configs/marl/requirements-lock.txt --wheel-dir wheelhouse
/tmp/marl-wheel-build/bin/python -m pip wheel --no-deps . --wheel-dir wheelhouse
tar -czf marl-wheelhouse-linux-cp312.tar.gz wheelhouse
```

Transfer the repository and wheelhouse, then run:

```bash
export MARL_WHEELHOUSE="$PWD/wheelhouse"
./scripts/setup_marl_linux.sh
```

Wheel builds must be performed on Linux for Linux; macOS Torch wheels are not portable. The dependency *versions* remain pinned, while platform-specific wheel files differ. The Linux setup has been authored but has not been executed on a Linux host.
