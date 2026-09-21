# New-core audit migration gate

This directory carries every ID from the frozen 178-probe legacy audit into the rewritten simulator. `legacy_probe_migration.json` records the latest preserved claim and its new disposition. `python -m audit.run` executes the public new-core oracle packs and writes `audit/reports/latest.json`.

The migration does not treat a changed implementation as a reason to hide a probe:

- Legacy implementation inspection probes are replaced by closed-schema, analytic physics, provenance, or legacy-isolation oracles.
- Portable L6 sensor and L8 lifecycle/metamorphic behavior runs against the rewritten core.
- Deterministic offline fixtures are authoritative for supported L7 products. The previously accepted live-provider smoke gate remains separately deferred.
- ERA5 stays outside coordinated V1, as specified in the migration matrix.
- The reserved L8-P12 slot has no acceptance claim and cannot be reported as a pass.

The command fails whenever an executable oracle pack fails. Accepted skips are enumerated and cannot conceal an executable failure.
