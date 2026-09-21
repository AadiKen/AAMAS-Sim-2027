# BCOD-Sim rewrite

This repository is the new BCOD-Sim implementation. Phases 0–6 provide the
package scaffold, strict config resolution, identity registry, frame/unit
conversions, a pre-run provenance manifest, canonical vessel state, and the
6-DOF dynamics plant with constrained planar mode. Typed actuators, command
bounds, deterministic per-owner wrench reduction, a parametric world sampler,
physical, abstract, and debug sensor contracts are also available. Phase 6
adds deterministic contact response and a double-buffered shared wake field.
It does not yet provide a complete simulator or a runnable experiment.

The implementation contract and invariant ledger are in `docs/`. The separate
legacy reference tree is `../ICRA27-Sim/`; it is **not** a package dependency,
vendored source, or runtime fallback. The behavioral audit remains there until
its adapter is migrated in Phase 13. Consult the migration matrix before
porting any behavior.

Run the Phase 0 gate with:

```sh
python3 tools/check_legacy_imports.py
python3 -m pip install -e '.[test]'
python3 -m pytest -q
```

Later phases require explicit approval after each phase gate.
