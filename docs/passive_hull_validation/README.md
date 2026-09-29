# Passive Hull Validation Campaign

Run the available campaign from the repository root:

```bash
python -m bcod_sim.validation.passive_hull_campaign \
  --output docs/passive_hull_validation/results
```

The runner verifies frozen generator hashes, stages previously acquired source
files without editing their bytes, verifies source digests, rebuilds the
KVLCC2M OBJ from the official NMRI grid, attempts M1 package generation, writes
all available metrics, plots, and the report. Missing packages and references
are recorded as unscored or unavailable results while the campaign continues.

Campaign revision 2 includes a topology-preserving geometry cleanup fix. Its
M1 tests passed 53/53 with Capytaine's cache directed to the workspace. The
KVLCC2M geometry is qualified against official offsets and NMRI properties.

Two numeric resistance references are recovered (KVLCC2M and KCS) and KVLCC2
derivatives are stored as contextual values. They are not yet physical scores:
the KVLCC2M source geometry ends at its design waterline and cannot satisfy the
frozen restoring LUT extent, while the KCS geometry is not yet staged. The
DTMB 5512 official page confirms force/moment data availability, but the
campaign source audit still lacks a defensible numeric file and moment mapping.

The original NMRI/ATMA source files already present in the workspace were
copied byte-for-byte into `raw_sources/` and verified against their previous
SHA-256 records. Current runtime network DNS is unavailable, so newly recovered
paper excerpts are represented by versioned reference data with full source
identifiers rather than fabricated raw downloads.
