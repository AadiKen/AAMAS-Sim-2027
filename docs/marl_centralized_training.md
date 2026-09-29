# V3 centralized-training boundary

Section 3 implementation status: the V3 environment and training-only global-state interface are implemented and API-tested. **BenchMARL MAPPO 1.5.2 is now integrated and smoke-qualified locally, but M0 is only partially training-qualified.** See `docs/marl_m0_validation.md` for the completed three-seed result. No policy-gradient code was substituted.

## Environment and information boundary

`NavigationParallelEnv` is a native PettingZoo `ParallelEnv` for simultaneous vessel actions. It exposes `possible_agents`, `agents`, per-agent observation/action spaces, `reset`, `step`, `state`, and `close`. Actor observations use the existing 78-field, measured ego-relative schema `ego-relative-v2`; the same actor network can be applied to every homogeneous vessel. Neither `reset` nor `step` returns the centralized state in an actor observation or agent info. `state()` is an explicit training-only call for a centralized critic. This interface does not give actors hidden truth, other agents' policy state, future information, or critic input.

The separate global schema is `fleet-global-state-v1`, with a SHA-256 derived from its ordered field list. It has 89 normalized fields: remaining time; four padded agent slots containing measured position, heading, speed, yaw rate, finite-difference world velocity, goal, reached flag, and presence; and ten padded static-obstacle slots with position, radius, and presence. It is built from the backend-independent `BackendFrame` readings and the versioned task/scenario. The same encoded state is identical for an identical frame with kinematic, BCOD-reduced, or BCOD-full task configurations. This is a schema contract test, not native BCOD qualification.

The ParallelEnv supports both V3 action modes, while `desired-speed-heading-v1` remains the intended primary MARL mode. For that mode, each local action passes through the existing shared heading controller before the backend receives desired speed and yaw rate. The environment itself contains no optimizer or MAPPO loss implementation.

## MAPPO integration

The installed BenchMARL 1.5.2 `PettingZooClass` custom task and TorchRL 0.11.1 `PettingZooWrapper` now consume this ParallelEnv. Inspection of the constructed experiment confirms actor input `('agents', 'observation')`, critic input `state`, shared actor parameters, and a centralized critic. The adapter, dependency lock, smoke result, and three-seed qualification are documented in `docs/marl_training_system.md`, `docs/marl_dependency_stack.md`, and `docs/marl_m0_validation.md`.
