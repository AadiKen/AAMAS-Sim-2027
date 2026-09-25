# Offline 6-DOF CFD identification

BCOD's production identification pipeline describes CFD cases independently of
the solver runner. A case selects one of steady velocity, steady rotation,
forced translation, or forced rotation and one body-FRD degree of freedom. All
six force/moment channels are retained. Steady sweeps must contain both signs.

`single_phase` is intended for submerged craft. `free_surface` selects a
two-phase VOF case for surface vessels. Geometry, water properties, motion,
mesh, solver, domain, and OpenFOAM version all participate in the case hash.
Surface-vessel identification defaults to k-omega SST RANS, a 34 × 20 × 10 m
far-field domain, and three smooth-wall prism layers. The selected turbulence
model, inlet k/omega/nut, prism-layer thickness, and source waterline are
explicit case inputs and recorded in metadata. The source STL is translated
rigidly so its mass-derived equilibrium waterline is z=0 in the CFD frame;
the original geometry hash and translation are retained for provenance.
The solver uses water/air inlet patches, pressure-controlled outlet and
atmosphere, far-field slip bottom/sides, a no-slip hull, and adaptive
Courant/alpha-Courant time stepping. Hull y+ is logged. A production coefficient
case must still pass mesh, Courant, alpha-Courant, timestep, residual,
water-volume, and six-channel stationarity checks; using SST does not waive
any gate. The laminar mode remains explicit for legacy evidence and is not
the default for surface-vessel identification.
Only results that pass residual and force-stationarity checks may enter the
valid-result cache or coefficient fit. Mesh, timestep, and domain sensitivity
use the same reusable three-level convergence check.

Six-channel stationarity uses the same 3% mean-change and slope limits for
material loads. A load close to zero cannot be divided by its own mean. For
each force or moment channel, let `S` be the declared physical force or moment
scale, `m` its window mean, and `w = min(1, |m|/(0.01 S))`. The denominator for
each test is `(1-w)(A/R)S + w|m|`, where `A` is the existing absolute near-zero
limit and `R` the existing relative limit. Mean change and window-integrated
slope use `A=0.005, R=0.03`; standard deviation uses `A=0.01, R=0.10`.
At zero mean this exactly retains the physical-scale near-zero limit; above
1% of the scale it is the original material relative test. The periodic-load
detector uses the same physical-scale transition so a negligible cross-axis
ripple cannot spuriously override the window gate. No acceptance limit was
relaxed. A qualified case carries a hashed fitting-window manifest, and the
OpenFOAM parser recomputes the six-component mean from only those verified
samples. Startup history remains available for audit but is excluded from fit.

The fitter discovers the coefficient form exposed by Plant6: a symmetric 6x6
added-mass matrix, a full linear damping matrix, diagonal quadratic damping,
and explicitly configured nonlinear cross terms. It does not create terms the
runtime cannot consume. Forced-motion samples are fit in the time domain using
the prescribed velocity and acceleration; their phase separation identifies
damping and added mass.

Hydrostatic displacement, equilibrium waterline/draft, center of buoyancy,
waterplane area and moments, and heave/roll/pitch restoring stiffness are
derived from closed geometry rather than CFD. STL geometry is interpreted in a
right-handed source frame with positive z upward and converted to body FRD in
the generated vessel package.

Prepare the default signed/free-surface campaign with:

```bash
bcod vessel identify --geometry vessel.stl --mass 180 --cg 0,0,0 \
  --output generated_vessels/my_vessel
```

After an execution backend has produced a quality-gated observation JSON,
repeat the command with `--observations observations.json` to write
`vessel.yaml`, `coefficients.json`, `provenance.json`, `geometry.json`, and
`fit_report.md`. `LocalDispatcher` supplies process-based local execution;
cluster/cloud runners implement the small `DispatchBackend.map` interface.
