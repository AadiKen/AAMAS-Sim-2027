# DTMB 5512 PMM load contract

The comparator targets the experimental fluid reaction only. Rigid-body inertia is excluded when the publication reports hydrodynamic forces. For prescribed motion, Plant6's mass matrix remains part of the equation solver, but it must not be added to the reference force being compared.

For a physical added-mass model, use the passive fluid reaction

```text
tau_fluid = -M_A * nu_dot - C_A(nu) * nu - D(nu)
```

where `M_A` is the generated added-mass matrix, `C_A` is Plant6's analytic added-mass Coriolis operator, and `D` is the serialized damping/resistance/crossflow wrench. Evaluate acceleration and velocity components separately to distinguish acceleration-phase errors from damping-phase errors. Do not include rigid-body Coriolis or rigid-body inertia in a hydrodynamic-only EFD quantity.

Current evidence limitation: the workspace source audit records DTMB 5512 as rejected for quantitative comparison due to bilge-keel configuration and numerical moment-origin mapping. Raw time histories or numeric harmonic coefficients have not been admitted. Status remains `NOT_AVAILABLE_PUBLICLY` pending a defensible source map.
