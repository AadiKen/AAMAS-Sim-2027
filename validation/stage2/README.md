# Stage 2 environment and world-interaction validation

Run the campaign from the repository root:

```sh
scripts/run_stage2_validation.sh --run-id latest
```

The runner writes `stage2_results/<run-id>/` with a frozen manifest, a case
registry, per-case raw data and metrics, plots when Matplotlib is available,
and an honest `report.md` gate.  A missing production capability is recorded
as `BLOCKED`, never converted into a passing test.

The campaign deliberately imports production field, load, bathymetry,
collision, and `Plant6` APIs, but its analytical expectations are calculated
locally.  It never changes or retunes vessel dynamics.
