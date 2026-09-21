# Phase 12 vessel identification and calibration

Phase 12 establishes that geometry, CFD adaptation, coefficient identification, and calibration work end to end against known synthetic truth. It does not establish real vessel accuracy.

All vessel creation paths emit the same versioned `CanonicalVessel`. CFD backends stop at a standardized force and moment dataset. OpenFOAM files remain inside the adapter. Coefficient fitting checks matrix rank and conditioning before returning a result. Calibration preserves the original parameter source and evaluates a separate maneuver family.

The required artifact manifest records the geometry hash, backend data and settings, maneuver matrix, raw results, fit diagnostics and uncertainty, calibration dataset and held-out result, code revision, and random seed. Real Surveyor or other field data can later replace synthetic logs without changing this architecture.
