# 2D+t transverse-viscous maneuvering model reconstruction

Primary source: Ø. Rabliås and T. Kristiansen, [“A 2D+t approach for the transverse viscous loads in a modular maneuvering model,” *Ocean Engineering* 228 (2021) 108853](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/3003486/1-s2.0-S0029801821002882-main.pdf?isAllowed=y&sequence=1), DOI 10.1016/j.oceaneng.2021.108853. The paper is open access. It tested DTC 25° and 35° rudder turning circles against free-running measurements, including regular waves. The authors report that the integrated, section-shape-dependent history method improved turning-circle agreement over conventional cross-flow in their study, but that the simpler scaled-cylinder history can be a useful starting point below about 25° drift.

## Equations adopted for the Level A experiment

Take body x positive forward, y starboard, z down (BCOD FRD). The paper uses its own axes; signs here are transformed to physical resisting load. For a section at x, local transverse body speed is `q(x)=v+(x-x_ref)r` (paper Eq. 16–17). Local forward speed for a laterally offset hull is `u_i=u-y_i r`; the latter is BCOD rigid-body kinematics, an extension beyond the paper's centered monohull example.

The authors introduce an earth-fixed transverse plane first met by the bow at `x_b`. For constant `u,v,r`, the lateral travel while that plane moves from bow to section is (paper Eq. 18–19):

`s_y(x) = [v(x_b-x) + (r/2)((x_b-x_ref)^2-(x-x_ref)^2)] / u`.

The nondimensional displacement/age is `t*(x)=|s_y(x)|/T(x)` (paper Eq. 20), where `T(x)` is **local draft**. It is not `U t/L`, the global speed, or an arbitrary relaxation time. For nonconstant motion, the underlying definition is the integral of local transverse velocity along the plane's passage, with the entrance time determined by its forward travel. A steady captive shortcut may use Eq. 19; a runtime implementation requires the velocity history. At zero or reverse `u`, the bow-first plane mapping is inapplicable.

Conventional separated cross-flow is (paper Eq. 16–17):

`dY = -0.5 rho Cd(x) T(x) q(x)|q(x)| dx`, `dN=(x-x_ref)dY`.

The 2D+t substitution is `Cd(x) -> Cd(t*(x))`. The authors describe three choices: (i) nearest Aarsnes 1984 section startup curves scaled to steady drag (`2D+t_0`), (ii) a circular-cylinder startup curve scaled to mean steady hull drag (`2D+t_cyl`), and (iii) integrate shape-dependent `dCd/dt*` along the hull, then normalize to steady drag (paper Eq. 21–23). Method (iii) performed best for DTC but requires Aarsnes' per-section curves or equivalent 2D data. Method (ii) is the literature-driven Level A candidate here; it is **not** claimed to reproduce method (iii).

The paper's cylinder fit (Eq. 21 and Table 1) is:

`Cd(t*) = (Cd_inf/1.2) P(t*)`, where

`P(t*) = 2.481e-7 t*^5 - 3.647e-5 t*^4 + 1.906e-3 t*^3 - 4.417e-2 t*^2 + 0.4315 t* + 0.07339`.

The fit represents Sarpkaya's impulsive-cylinder data as reported by the authors. Its steady reference 1.2 is from that experiment. The source graph/polynomial has an initial low value, an overshoot, and a late return near 1.2. The polynomial is not a physically valid extrapolation beyond its plotted span: the Level A code uses it through `t*=25` and sets the **normalized** value to 1 afterward. This cutoff is a documented numerical domain restriction, not a tuned vessel parameter. `Cd_inf` is the *existing geometry-derived sectional* V5 drag coefficient. This differs from the paper's simplified `2D+t_cyl`, which uses a single hull-mean `Cd_inf`; it is a shape-aware Level A hypothesis and must be judged separately.

For a dynamic maneuver, the flow-plane age must be computed from past `u,v,r`; resetting `t*` from current state would discard the proposed physics. A history implementation must account for acceleration/deceleration and changing yaw. The paper's Eq. 22–23 shape-specific integration is the Level B target once 2D viscous curves become available; independently computed 2D section drag data do not automatically include 3D inter-hull flow or lifting circulation.

## Assumptions and domain

- Separated transverse drag only. BEM added mass, added-mass Coriolis, longitudinal resistance, hydrostatics, propulsors, and any distinct linear lift stay separate. Reusing a whole V5 lateral load plus 2D+t lateral load would double count drag.
- The bow-first flow-plane construction assumes positive forward speed and a hull shape that passes each plane in order. It is not justified at zero surge, reverse, or where rotational local axial speed reverses.
- The generalized cylinder curve is not a geometry-specific viscous solution. Section depth and steady geometry-derived drag enter Level A, while detailed section curvature, bilge radius, Reynolds transition, free surface, and 3D vortex interaction are unrepresented.
- The DTC turning-circle comparison includes rudder, propeller, and wave models. Its agreement cannot be transferred as a numerical accuracy guarantee for bare-hull KCS captive Y/N or WAM-V catamarans.
- The older [Hooft (1994) segmented-ship experiments](https://www.sciencedirect.com/science/article/pii/0029801894900043) emphasize the longitudinal distribution of cross-flow force; that motivates exporting `dY/dx`, `dN/dx`, and cumulative moments rather than judging integrated Y alone.

## Parameter provenance

The six polynomial constants and `1.2` come directly from Rabliås–Kristiansen Eq. 21/Table 1. The `t*=25` domain limit is a conservative code boundary selected where the polynomial has returned to approximately 1.2; it is **not** a published universal cutoff and must be sensitivity-tested. Existing V5 `Cd_inf(x)` is geometry-derived using the frozen Hoerner/fullness rule, with its original applicability limitations. No KCS/HMRI value enters any of these choices.
