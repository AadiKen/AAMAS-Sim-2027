# Phase 6 contact and wake contract

The collision stage runs after each continuous RK4 substep. Broad-phase pairs
and narrow-phase contacts are sorted by stable `(environment, body, child)`
identity. The solver applies impulses using the same full 6-DOF mass matrix as
the plant, or its active `[surge, sway, yaw]` projection in planar mode. It
then corrects penetration without adding kinetic energy. Contact diagnostics
include body-frame impulse and an equivalent wrench for the substep duration.

Supported narrow-phase pairs are sphere–sphere, sphere–oriented-box,
oriented-box–oriented-box, sphere–capsule, capsule–capsule, sphere–convex-hull,
and sphere–static-triangle-mesh. Compound shapes expand into child primitives.
An overlapping unsupported pair raises `CollisionSolverError`. Dynamic triangle
meshes are rejected at construction. Continuous collision detection is not
implemented, so very large substeps can tunnel through thin geometry; vessel
operating envelopes must set an appropriate substep limit.

The world owns a Gaussian wake field with distinct current and next buffers.
Every vessel reads only the current buffer during a master step. Emissions are
keyed by `(environment, source vessel)`, sorted, and reduced deterministically
when sampled after the boundary swap. The kernel is an analytic baseline and
has no field-calibrated fidelity claim.
