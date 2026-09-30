# BCOD-Sim rewrite

This repository contains the BCOD-Sim simulator, including strict config
resolution, versioned definitions, a deterministic episode engine, 6-DOF
vessel dynamics, actuators, sensors, parametric worlds, logging, and policy
bundle inference. The supported web MVP runs a narrow one-vessel waypoint
workflow through the same simulator core.

The implementation contract and invariant ledger are in `docs/`. The local
MANTA web control plane MVP is documented in
[`docs/web_quickstart.md`](docs/web_quickstart.md); it supports canonical
experiment editing, qualification, local simulation jobs, policy evaluation,
and replay. The separate
legacy reference tree is `../ICRA27-Sim/`; it is **not** a package dependency,
vendored source, or runtime fallback. The behavioral audit remains there until
its adapter is migrated in Phase 13. Consult the migration matrix before
porting any behavior.

Run the simulator checks with:

```sh
python3 tools/check_legacy_imports.py
python3 -m pip install -e '.[test]'
python3 -m pytest -q
```

See the web quickstart for the browser and headless local runner workflow.
