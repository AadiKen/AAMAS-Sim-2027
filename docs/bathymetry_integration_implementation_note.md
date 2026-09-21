# Bathymetry integration: pre-change implementation note

Recorded before production behavior changes.

- **Sources and representation.** Parametric worlds support constant-flat and
  analytic planar bottoms through `world.Bathymetry`. Real-world bundles query
  frozen GEBCO samples. GEBCO source elevations are positive-up WGS84 and are
  explicitly negated into local NED bottom Z. The parametric implementation has
  no raster type; GEBCO currently uses bounded nearest-neighbour sampling.
- **Coordinates and datum.** Simulation world coordinates are NED
  (north/east/down), vessel coordinates are FRD. Bottom Z and depth below the
  NED origin increase downward. Every bathymetry source carries an explicit
  vertical-datum string. Parametric world boundaries fail closed outside their
  declared coverage; GEBCO also fails outside declared geographic coverage or
  its maximum sample radius.
- **Collision.** The deterministic collision system uses sweep-and-prune,
  pair-specific narrow phase, and a sequential rigid-body impulse solver.
  Static bodies may be oriented and may use sphere, box, capsule, convex hull,
  triangle mesh, or compound proxies, but triangle meshes are static-only and
  narrow-phase mesh contact is presently implemented only against spheres.
  Configured world objects expose sphere and box proxies. Vessel proxies are
  supplied by the vessel runtime definition. Contact impulses act at the
  detected point, so the solver includes the `r × J` rotational contribution.
- **Contact material.** Restitution, Coulomb friction, and positional correction
  are global episode parameters; no penalty stiffness/damping law exists.
- **Waves.** Regular Airy and deterministic seeded spectral waves currently use
  deep-water `k=omega²/g` and exponential depth decay. The real-world bundle
  similarly creates a deterministic local deep-water sinusoid from NDBC
  snapshots. No shoaling, breaking, or current interaction exists.
- **Currents.** Parametric current fields are uniform, linear, or sinusoidal 2D/
  3D NED vectors and currently return values at every query point. RTOFS queries
  source depth levels. Neither path currently reports water-column validity.
- **Passive bathymetry gap.** Bottom queries are returned in `WorldSample`, but
  bathymetry does not create collision bodies, constrain current queries, or
  participate in wave dispersion. Therefore grounding is impossible and Stage
  2 grounding cases are correctly blocked.

The implementation will formalize the existing bathymetry object as the single
query interface and make collision, current validity, and finite-depth waves
consume it. It will not alter or retune `Plant6`.
