# MANTA Phase 2 implementation status

The new physical actuator stack is in `bcod_sim.actuators.physical`. It produces a six-component FRD load for Plant6's existing `propulsion` external term. Its `ActuatorPipeline.step()` returns that load and per-device diagnostics. `ActuatorSet.from_yaml()` loads the independent v1 package format; the example is `configs/manta_actuator_example.yaml`.

Implemented: fixed and tunnel thrusters, azimuth pods, a finite-aspect-ratio rudder, simple propeller slipstream, static maps, open-water KT maps, power-law and power-based thrust estimates, thrust/RPM/steering lags, physical force and lever-arm moment, deterministic fixed and nonlinear allocation, residual diagnostics, basic authority analysis, and speed/heading control with allocation feedback.

The new stack is an opt-in API. The existing episode engine still uses the legacy `ActuatorBank`; the new pipeline has not been routed into episode configuration, RL action spaces, or the engine's substep loop. Therefore the three modes currently work through `ActuatorPipeline`, but not through the web runtime or training environments. The generic stack must be wired into those surfaces before claiming the Phase 2 milestone.

Still outstanding: comprehensive force-ownership validation across hull packages, robust nonlinear fallback with fixed-angle thrust resolution, all requested synthetic fixtures and benchmarks, calibration and uncertainty sampling, Wageningen B-series, CAD-derived rudder geometry, Surveyor actuator packaging and held-out validation. No real-vessel accuracy claim is made.
