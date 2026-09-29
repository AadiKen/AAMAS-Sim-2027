# Convention validation

- Frame: KVLCC2M package is BCOD FRD. Source NMRI axes and right-handed signs follow the staged NMRI axis figure and admitted reference mapping.
- Moment origin: source/reference convention is midship on the undisturbed waterplane; the reconstructed mesh retains the symmetric normalized source origin and was not translated for force fitting.
- Positive drift: for beta=+6 deg, set `u=0.988554764 m/s`, `v=-0.103901292 m/s`, `r=0`.
- Manual sanity check at +6 deg: EFD has `CY=+0.02560`, `CN=+0.01392`; MANTA gives `CY=+0.041473`, `CN=+0.011192`. Both signs match, so there is no global Y/N sign inversion.
- Oblique force denominator is `0.5*rho*U^2*Lpp*T = 812.878336 N`; moment denominator is `0.5*rho*U^2*Lpp^2*T = 4040.005332 N m`.
- Resistance denominator is `0.5*rho*U^2*S0`, with official wetted-area reference `S0=6.58919 m^2`.
- Runtime ownership roundtrip: 343 state samples; maximum independent-vs-Plant6 force residual `6.82e-13 N` (or N m for moment entries); pass.
