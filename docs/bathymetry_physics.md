# Bathymetry physics

BCOD uses one canonical bathymetry API for bottom/depth queries, water-column
validity, wave calculations, sensors, and static seabed contact. Coordinates are
world NED: positive bottom Z and depth are downward. Source adapters must retain
their declared vertical datum; GEBCO positive-up elevations are converted once
at its adapter boundary.

When bathymetry collision is enabled, the collision system caches deterministic
spatial height-surface tiles. These use the existing rigid-body impulse solver,
including restitution, Coulomb friction, positional correction, the actual
contact point, and the resulting `r × J` moment. Parameters are numerical
contact settings, not measured sediment properties. No tile mesh is regenerated
per timestep.

Wave configurations remain `deep_water` by default. `finite_depth` solves
`omega² = g k tanh(kh)` independently for every regular/spectral component and
uses finite-depth vertical orbital structure. Optional shoaling conserves the
simplified energy-flux quantity `H² Cg`; optional breaking caps height at
`gamma*h`. Optional locally-uniform current interaction uses intrinsic
frequency. Refraction is explicitly disabled.

Current fields keep their upstream meaning. Parametric 2D fields are vertically
uniform throughout valid water; RTOFS retains its source-derived 3D levels.
Samples expose validity and depth semantics. A below-bottom sample is invalid,
not silently attenuated or replaced with zero.

Limitations: no sediment deformation, digging/plowing, surf-zone CFD,
diffraction, 2D refraction, arbitrary shear wave coupling, or fabricated
vertical current profile. Locally varying waves use a local-depth approximation
rather than ray tracing.
