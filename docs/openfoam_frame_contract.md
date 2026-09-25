# OpenFOAM–BCOD frame contract (v1)

`bcod-openfoam-frd-v1` is the production identification contract. A case without
this identifier cannot be parsed into a production observation.

## Axes and origin

| Representation | Handedness | X | Y | Z | Origin and units |
| --- | --- | --- | --- | --- | --- |
| Input STL / hydrostatics | right | bow/forward | port | up | Declared vessel origin; metres |
| OpenFOAM mesh/fields | right | bow/forward | port | up | Source origin translated so waterline is z=0; SI |
| BCOD body / Plant6 / fitted coefficients / vessel.yaml | right | forward | starboard | down | Declared vessel origin; SI |

All six-vectors use `[surge, sway, heave, roll, pitch, yaw]` and
`[Fx,Fy,Fz,Mx,My,Mz]`. Angular rates and moments use the right-hand rule
about their respective positive axes. The source STL must be in metres with
the stated axes. A hull shape alone cannot prove which end is the bow or
where the CG lies; the importer requires a declared frame and geometry
orientation remains a provenance obligation. The supplied `cg_frd_m` is
explicitly recorded; the CFD moment point defaults to that CG in the CLI.

## Geometry and waterline

Let `R = diag(1,-1,-1)`, a proper rotation (`det R = +1`). For a vector,
`v_foam = R v_FRD` and `v_FRD = R v_foam`. The incoming source geometry
already uses OpenFOAM axes, so mesh vertices are `x_foam = x_source -
(0,0,z_waterline_source)`. The hydrostatic source waterline is positive-up.
The free surface is z=0 in OpenFOAM. For a BCOD point `p`, including CG or
moment reference, `p_foam = R p - (0,0,z_waterline_source)`. Metadata records
the source waterline, translation, CG and both moment-center coordinates.

## Flow and prescribed motion

For a **fixed hull** representing body translation `v_body` in still water,
`U_inlet,foam = -R v_body`: positive surge gives negative X inflow, positive
sway gives positive Y inflow, and positive heave gives positive Z inflow.
This is a relative-flow identity, not a change to the CFD numerical model.
For a **moving hull** in still water, translational/rotational motion axes
map through `R` without an additional minus sign. Thus positive body roll
maps to positive X angular motion, pitch to negative Y, yaw to negative Z;
angular acceleration maps identically. The prescribed-motion dictionaries
were execution-qualified on a tiny box; see `openfoam_motion_qualification.md`.

## Force, moment and fitting signs

The OpenFOAM Foundation 11 `forces` function object integrates pressure and
viscous traction on the `hull` patch about its configured `CofR`. Its source
uses pressure times the fluid patch area vector and viscous stress times that
vector, then forms moments from `(face centre - CofR) × force`. This is the
**fluid-on-hull** physical wrench. Our cases do not enable porosity; the
two-channel parser rejects unexpected output layouts. Source:
https://cpp.openfoam.org/v11/forces_8C_source.html

The fitter stores **positive resisting** coefficients, while Plant6 applies
their negative wrench to motion. Let `A` be the OpenFOAM `CofR`, `B` the
desired BCOD reference expressed in OpenFOAM coordinates, and `r_BA=A-B`.
For the physical wrench, `M_B = M_A + r_BA × F`. The observation sent to
the fitter is `w_resist = -blockdiag(R,R) [F; M_B]`. This shift uses the
*same physical point* after waterline translation. A qualified window and
its sample hash are required; startup samples and the final single sample
cannot silently enter a production fit. `debug_last_sample=True` is for
smoke/diagnostics only.

Positive body surge with relative negative-X flow should yield negative-X
fluid load, hence positive surge resistance. Positive body sway yields
positive sway resistance after the Y-axis conversion. Positive body yaw
should yield negative body yaw physical moment and positive yaw resistance.
Analogous signs apply to heave, roll and pitch. These are expected physical
checks; they cannot substitute for executing and qualifying those cases.

### Worked examples

* Surge: `v_FRD=(1,0,0)` m/s gives `U_inlet=(-1,0,0)` m/s. A synthetic
  `F_foam=(-10,0,0)` N gives `F_resist,FRD=(10,0,0)` N; Plant6 applies
  `(-10,0,0)` N for that operating point.
* Sway/yaw: `v_FRD=(0,1,0)` gives `U_inlet=(0,1,0)`; a fluid load
  `F_foam=(0,10,0)` maps to physical `F_FRD=(0,-10,0)` and resistance
  `(0,10,0)`. Positive yaw rate maps to OpenFOAM angular rate `(0,0,-r)`.
  If the same force is reported about `A=(1,0,0)` rather than `B=(0,0,0)`,
  `M_B,foam=(0,0,10)` N·m and the resisting BCOD yaw moment is +10 N·m.

## Qualification boundary

This contract establishes algebraic signs and metadata. It does not prove
that every imported CAD file has the declared orientation, or that
all production cases pass quality and experimental validation. Existing
pre-v1 CFD artifacts lack the contract identifier and require regeneration
or an explicit reviewed migration before production ingestion.
