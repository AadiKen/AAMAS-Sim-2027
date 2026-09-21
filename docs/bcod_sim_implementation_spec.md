# BCOD-Sim Rewrite — Full Implementation Specification

**Status:** implementation-ready baseline  
**Purpose:** technical specification for the ground-up rewrite of the marine multi-agent simulation platform  
**Authority:** derived from the frozen system architecture, the September 2026 behavioral audit, and subsequent design decisions  
**Legacy baseline:** `AadiKen/ICRA-simulator`, legacy implementation under `sim-v3/`, audited at commit `fac3fd0fa46bb6ebb69de6f8f3978745b83d5a40` with dirty-tree patch preserved separately

---

## 0. How to use this document

This document is the implementation contract for the new simulator. It is intentionally more concrete than the system-architecture document: it assigns modules, data structures, execution order, default technologies, validation rules, migration boundaries, and acceptance gates.

### 0.1 Order of authority

When sources disagree, use this order:

1. **This implementation specification.**
2. **The frozen system-architecture document and invariant ledger.**
3. **Behavioral-audit findings and portable regression probes.**
4. **Explicit migration notes for a legacy component.**
5. **Legacy source code.**
6. **Legacy comments, labels, presets, defaults, figures, and historical metadata.**

The legacy implementation is source material, not the specification. A legacy behavior is not preserved merely because code exists for it.

### 0.2 Non-negotiable rewrite rule

The new production package **must not import the legacy simulator at runtime**. Reused algorithms must be deliberately migrated into new modules, covered by new tests, and made to satisfy the interfaces and invariants below. The old tree may be referenced from audit tooling and migration tests only.

### 0.3 Primary design objective

A researcher should be able to:

1. select, create, or import vessels;
2. select or generate a world;
3. define a task through config;
4. run or train multi-agent policies through a standard API;
5. reproduce the run exactly from the emitted artifact bundle;
6. use the web application to author the same configs and demonstrate trained policies;
7. drop to lower-level controls, sensors, physics, or custom plugins only when necessary.

The system is **config-first, code-optional, backend-singular, fail-closed, and reproducible by construction**.

---

# 1. Recommended implementation stack

These are implementation defaults, not architectural invariants. They are chosen to make the rewrite directly usable by MARL researchers while retaining GPU/vector execution.

## 1.1 Core and research stack

- **Language:** Python 3.12+
- **Tensor/numerical backend:** PyTorch
- **Config/schema validation:** Pydantic v2
- **Config authoring:** YAML as the primary human format; JSON accepted as an equivalent serialized form
- **Testing:** pytest + Hypothesis for property-based tests
- **Tabular artifacts:** Apache Parquet / PyArrow
- **Large tensor/sensor artifacts:** Zarr
- **Event streams:** JSONL
- **RL interoperability:** PettingZoo Parallel API adapter
- **Web backend:** FastAPI
- **Web frontend:** TypeScript + React; Three.js or equivalent for 3D visualization
- **CFD baseline adapter:** OpenFOAM, isolated behind a vessel-generation interface

## 1.2 Numerical modes

Two numerical profiles should be exposed:

- `validation`: float64, deterministic operations required, expensive diagnostics enabled when requested.
- `training`: float32 by default, deterministic or throughput mode selectable. Any run claiming exact reproducibility must use the deterministic path and record the backend/device/dtype.

Exact equality is promised only for **same backend + same device class + same dtype + same resolved config + same seed + deterministic mode**. Cross-platform comparisons use explicit tolerances.

## 1.3 Hot-loop rule

The physics loop is in-process. No HTTP/RPC/network request may occur per physics step. Web communication, external-data fetching, policy upload, and asset resolution happen outside the hot loop.

---

# 2. Repository and package layout

Recommended top-level structure:

```text
bcod-sim/
  pyproject.toml
  README.md
  configs/
    schemas/
    examples/
    vessels/
    worlds/
    tasks/
  src/bcod_sim/
    core/
      engine.py
      clock.py
      lifecycle.py
      errors.py
      events.py
      ids.py
    config/
      models.py
      resolver.py
      registry.py
      hashing.py
      validation.py
    frames/
      types.py
      transforms.py
      geodesy.py
      bathymetry.py
      display.py
    state/
      vessel_state.py
      world_state.py
      tables.py
    dynamics/
      plant6.py
      matrices.py
      coriolis.py
      damping.py
      restoring.py
      integrators.py
      planar_constraint.py
      operating_envelope.py
      diagnostics.py
    actuators/
      base.py
      thruster.py
      azipod.py
      rudder.py
      propeller.py
      allocation.py
      autopilot.py
      registry.py
    world/
      world.py
      fields.py
      bathymetry.py
      weather.py
      waves.py
      currents.py
      obstacles.py
      traffic.py
      wake.py
      import_pipeline.py
    collision/
      shapes.py
      broadphase.py
      narrowphase.py
      contacts.py
      solver.py
      damage.py
    sensors/
      base.py
      scheduler.py
      gps.py
      imu.py
      lidar.py
      sonar.py
      abstract.py
      ground_truth.py
      registry.py
    tasks/
      base.py
      rewards.py
      termination.py
      waypoint.py
      formation.py
      coverage.py
      registry.py
    scenario/
      scenario.py
      generator.py
      distributions.py
      resolved.py
    batching/
      population.py
      grouping.py
      reduction.py
      buffers.py
    rl/
      pettingzoo_env.py
      vector_env.py
      observation.py
      actions.py
      centralized_state.py
    policy/
      bundle.py
      manifest.py
      runtime.py
    logging/
      manifest.py
      recorder.py
      metrics.py
      sinks.py
      provenance.py
    data_sources/
      base.py
      gebco.py
      enc.py
      rtofs.py
      ndbc.py
      nws.py
      coops.py
      ais.py
    vessel_generation/
      base.py
      manual.py
      procedural.py
      importers.py
      cfd/
        openfoam.py
        cases.py
        fitting.py
      calibration.py
    web/
      api.py
      models.py
    migration/
      audit_adapter.py
  web/
    ... React/TypeScript client ...
  tests/
    unit/
    property/
    physics/
    sensors/
    data_sources/
    integration/
    determinism/
    regression/
  audit/
    ... preserved behavioral-audit suite ...
  legacy/
    ... optional read-only checkout/submodule; never imported by production ...
```

The exact file names may change, but the responsibility boundaries should not.

---

# 3. Canonical identifiers and object identity

## 3.1 Identity-bearing objects

The following are identity-bearing and must resolve centrally before execution:

- vessel definitions;
- dynamics/coefficient sets;
- actuator types and actuator instances;
- sensor types and sensor instances;
- tasks;
- policies;
- assets;
- environment data products;
- scenario templates.

Unknown references are fatal. There is no fallback vehicle, fallback task, fallback sensor, fallback coefficient set, or silent plugin drop.

## 3.2 Stable IDs

Each resolved object carries:

- `id`: human-readable stable identifier;
- `version`: explicit semantic/config version;
- `content_hash`: canonical hash of the fully resolved payload;
- `source`: provenance record.

Friendly names are never sufficient provenance. A run manifest records the content hash of the object that actually executed.

## 3.3 Canonical vessel catalog for initial implementation

The initial catalog contains:

1. **SeaRobotics Surveyor** — primary real-vessel family. If the two physical Surveyor boats differ materially, represent them as explicit calibrated instances derived from one base geometry, e.g. `surveyor_usv1` and `surveyor_usv2`; do not silently pool divergent calibration runs.
2. **MSS Otter** — reference-only vessel used for physics verification. It is not a paper contribution and must be sourced from a pinned primary MSS reference with per-number provenance where available.
3. **Dual-azipod reference vessel** — built from scratch from authoritative geometry/actuator definitions and later CFD-derived coefficients. Do not port the legacy `Vehicle C` object or its coefficients as canonical truth.
4. **Generated/imported vessels** — procedural, manual, CAD/mesh, and future CFD-generated definitions use the same vessel contract.

Legacy identities `Vehicle A`, `Vehicle B`, `Vehicle C`, and ambiguous `Otter`/`ASV Surveyor` presets do not survive as production identities.

---

# 4. Coordinate frames, units, and conversion boundary

## 4.1 Canonical internal convention

- Navigation/world frame: **NED** — +X North, +Y East, +Z Down.
- Vessel body frame: **FRD** — +X Forward, +Y Right/starboard, +Z Down.
- Units: **SI only** internally.
- Angles: radians internally.
- Forces: Newtons.
- Moments: N·m.
- Linear velocity: m/s.
- Angular velocity: rad/s.

## 4.2 Orientation representation

Internally store vessel orientation as a normalized quaternion representing the rotation **body FRD → world NED**.

Canonical quaternion ordering: `[w, x, y, z]`.

Euler roll/pitch/yaw are derived views for UI, logging, and convenience; they are not the authoritative orientation state.

## 4.3 One conversion subsystem

All conversion logic is centralized in `frames/`. No adapter, sensor, web component, or harness may implement ad hoc sign flips.

Required conversions include:

- geodetic WGS84 ↔ local NED;
- NED ↔ ENU;
- FRD ↔ FLU;
- body ↔ world vectors and wrenches;
- source-specific bathymetry elevation/depth ↔ canonical NED Z;
- canonical state ↔ web-display coordinates.

Every imported data product must declare source frame, units, datum, and vertical convention.

## 4.4 Bathymetry rule

A provider may return elevation-positive-up or depth-positive-down. The adapter must explicitly declare which and convert once at the boundary. Ambiguous vertical datum or sign is fatal.

## 4.5 Required property tests

At minimum:

- NED→ENU→NED round trip;
- FRD→FLU→FRD round trip;
- vector and wrench rotation round trips;
- quaternion normalization and inverse transform;
- positive starboard actuator offset produces the documented yaw-moment sign;
- bathymetry sign conversion for known positive/negative examples;
- web-display transform round trip for test points.

---

# 5. Configuration system

## 5.1 Authoring model

Users author versioned YAML/JSON. Configuration is composed from reusable definitions plus explicit overrides. Deep inheritance is not supported.

Unknown keys are errors.

Defaults are allowed only during config resolution and must be **materialized into the resolved config**. No downstream subsystem may inject a hidden default after resolution.

## 5.2 Resolution pipeline

```text
raw YAML/JSON
  → schema parse
  → identity/reference resolution
  → explicit defaults materialized
  → units normalized to SI
  → cross-object semantic checks
  → model-specific physical validation
  → scenario/world availability checks
  → canonical serialization
  → resolved-config hash
  → immutable ResolvedExperiment
```

Execution consumes only `ResolvedExperiment`, never a raw config.

## 5.3 Fail-closed rules

The resolver must reject:

- unknown keys;
- unknown presets/IDs/plugins/tasks;
- duplicate IDs within scope;
- nonfinite numeric values;
- missing units when the schema requires a dimensional value;
- illegal actuator bounds;
- invalid sensor mount transforms;
- unavailable required external data;
- invalid mass/inertia or model-specific matrices;
- scenario requests outside a declared hard operating constraint unless an explicit override mode exists and is logged.

## 5.4 Command bounds

Actuator commands outside declared bounds **must not be silently clamped**.

Supported policies:

- `error` — default, raises and terminates/invalidates the run;
- `clamp_with_event` — only when explicitly configured, emits a structured event with original and applied values.

## 5.5 Example experiment config

```yaml
schema_version: 1
experiment:
  id: surveyor_multiagent_example
  seed: 42
  numerical_profile: validation

simulation:
  dynamics_mode: full6       # full6 | planar3
  master_dt_s: 0.02
  dynamics_substeps: 2
  policy_every_n_master_steps: 5
  deterministic: true

world:
  source:
    kind: parametric          # parametric | real_world
  environment:
    current: {kind: uniform, ned_mps: [0.1, 0.0, 0.0]}
    wind: {kind: uniform, ned_mps: [2.0, 0.0, 0.0]}
    waves: {kind: regular, height_m: 0.2, period_s: 3.0, direction_rad: 0.0}
    visibility_m: 5000
  obstacles: []

vessels:
  - instance_id: leader
    definition: surveyor_usv1@1
    controller:
      mode: direct_actuator
    spawn:
      ned_m: [0.0, 0.0, 0.0]
      rpy_rad: [0.0, 0.0, 0.0]
  - instance_id: follower
    definition: surveyor_usv2@1
    controller:
      mode: high_level
    spawn:
      ned_m: [10.0, 0.0, 0.0]
      rpy_rad: [0.0, 0.0, 0.0]

task:
  type: waypoint_team
  reward:
    individual_weight: 0.5
    team_weight: 0.5
  disabled_agent_behavior: deactivate_keep_physical

logging:
  metrics: [reward, success, collision_count, distance_traveled]
  states: true
  sensor_payloads: false
```

---

# 6. Canonical runtime state

## 6.1 Vessel state

Use a flattened global vessel table across all active environment instances. Do not require a padded `[environments, max_vessels, ...]` tensor in the core.

For `N` total vessel instances:

- `env_id: int64[N]`
- `vessel_id: int64[N]`
- `position_ned: float[N,3]`
- `q_body_to_ned: float[N,4]`
- `nu_body: float[N,6]` ordered `[u,v,w,p,q,r]`
- `actuator_state_ref` / component ownership mappings
- optional health/energy arrays
- active-RL mask
- physical-entity-active mask

The canonical externally visible vessel state always remains 6-DOF even when the scenario uses constrained planar mode.

## 6.2 Wrench convention

Every physical contribution entering dynamics is expressed as a body-frame FRD wrench:

`[X, Y, Z, K, M, N]`

where X/Y/Z are forces and K/M/N are roll/pitch/yaw moments.

## 6.3 Immutable inputs

Policy actions and resolved episode inputs are values, not mutable shared objects. No environment instance may retain or mutate another environment's action object.

---

# 7. Simulation time and lifecycle

## 7.1 Master clock

Each environment has one fixed deterministic master timestep `dt_master`.

All vessels in that environment use:

- the same dynamics mode;
- the same number of fixed dynamics substeps per master step;
- the same policy tick frequency.

The environment may differ from another environment in a separate batch only if the batching scheduler groups compatible timing schemas. Initial implementation should strongly prefer a small set of timing profiles.

## 7.2 Fixed substeps

`dt_sub = dt_master / dynamics_substeps`

No adaptive timestep is used in V1.

A vessel operating envelope may declare a minimum required substep resolution. The environment must satisfy the strictest requirement among its vessels before starting; otherwise resolution fails.

## 7.3 Sensor rates

Each built-in sensor resolves its nominal update rate into an integer master-step interval for V1. If the requested rate cannot be represented within configured tolerance, resolution fails rather than silently approximating.

Latency is represented as an integer number of master steps in V1.

## 7.4 Common policy tick

All active RL agents in an environment act on one common policy tick. Between policy ticks, the last action is held.

Per-agent asynchronous policy timing is deferred.

## 7.5 Authoritative step order

Per master step:

1. If policy tick: receive one immutable action for every active RL agent.
2. Resolve high-level controls through autopilot/allocation into actuator commands.
3. Validate command bounds; error or explicit clamp event.
4. Update actuator internal dynamics for the first substep as required.
5. For each fixed dynamics substep:
   1. read the current immutable environment/shared-field buffer;
   2. evaluate each continuous force/moment contribution;
   3. perform dynamics integration;
   4. perform collision detection/contact resolution at the defined contact stage;
   5. enforce full6 or planar3 state constraints;
   6. accumulate new vessel-emitted shared-field contributions into the next buffer using deterministic reduction.
6. Swap shared-field buffers at the master-step boundary.
7. Update sensors that are due; push samples through latency queues.
8. Assemble policy observations from delivered sensor/abstract-state samples.
9. Compute reward terms.
10. Evaluate task success/failure and agent/episode termination.
11. Compute enabled metrics.
12. Emit structured events and asynchronous logs.
13. Return observations/rewards/termination/info at policy/API boundaries as appropriate.

No vessel may observe another vessel's partially-updated wake simply because of iteration order.

---

# 8. Primary 6-DOF dynamics plant

## 8.1 Governing form

Use a Fossen-style rigid-body marine model as the canonical dynamics form:

```text
M(ν) ν_dot + C(ν) ν + D(ν) ν + g(η) = τ_control + τ_environment + τ_contact
η_dot = J(η) ν
```

Implementation may organize `M`, `C`, `D`, and restoring terms into more specific rigid-body and hydrodynamic pieces, but the force accounting must remain inspectable.

Required conceptual components:

- rigid-body mass/inertia;
- added-mass matrix;
- rigid-body Coriolis/centripetal terms;
- added-mass Coriolis terms;
- damping, potentially nonlinear/state-dependent;
- hydrostatic/buoyancy restoring;
- actuator/propulsion wrench;
- current-relative-flow effects;
- wind load;
- wave load;
- wake/shared-field load;
- contact wrench.

## 8.2 Mass matrix validation

At construction, validate the required total mass/inertia structure for the chosen model. Validation is model-specific, not a simplistic sign check on every coefficient.

At minimum:

- all values finite;
- rigid-body mass > 0;
- rigid inertia physically valid;
- required total mass matrix nonsingular and appropriately positive definite for the formulation;
- condition number below an explicit safety threshold;
- damping behavior satisfies the model's dissipativity assumptions over declared validation probes;
- no duplicate or conflicting coefficient IDs.

## 8.3 Added mass

Added mass belongs in the solved mass matrix. It must not be implemented as delayed feedback using previous-step acceleration.

The old audit's negative test against the VRX-style failure mode becomes a permanent regression gate.

## 8.4 Integrator

Default continuous-force integrator: fixed-step RK4.

Quaternion orientation is integrated consistently with angular velocity and renormalized after each substep. Quaternion normalization errors beyond a small tolerance trigger a diagnostic event; nonfinite state is fatal.

Collision impulses are applied at a deterministic contact stage relative to the continuous integrator and documented in the run manifest.

## 8.5 Force-term diagnostics

The new core must retain an equivalent of the audit wrench probe. Given a frozen state, it can return:

- rigid-body Coriolis contribution;
- added-mass / added-mass-Coriolis contribution;
- damping;
- restoring;
- propulsion;
- current;
- wind;
- wave;
- wake;
- contact;
- total.

The diagnostic asserts that contributions sum to the applied total within numerical tolerance.

This is a production-supported debug interface, not audit-only instrumentation.

---

# 9. Constrained planar mode

## 9.1 Purpose

Planar mode exists as a lower-cost operating mode of the same 6-DOF plant. There is no second production 3-DOF plant and no independent 3-DOF coefficient set.

## 9.2 Active state

Active DOFs:

- surge `u`;
- sway `v`;
- yaw rate `r`;
- North/East position;
- yaw orientation.

Inactive DOFs:

- heave position fixed to vessel-specific equilibrium;
- roll fixed to vessel-specific equilibrium;
- pitch fixed to vessel-specific equilibrium;
- `w = p = q = 0`.

## 9.3 Reduced solve

Do not integrate full 6-DOF and merely overwrite the inactive states afterward.

Use an explicit projection/selection of the full 6-DOF model onto active DOFs. Conceptually, with selection matrix `S` for `[surge, sway, yaw]`:

```text
M_active = Sᵀ M S
rhs_active = Sᵀ rhs_full_at_constrained_state
νdot_active = solve(M_active, rhs_active)
```

The constrained state is still derived from the same 6-DOF parameters and force models.

## 9.4 Contact in planar mode

Collision detection may use full geometry. The contact wrench is projected into the active surge/sway/yaw subspace. Out-of-plane contact response is an explicit fidelity limitation of planar mode.

## 9.5 Switching rule

Dynamics mode cannot change during an episode. Training curricula may switch modes only between episodes/stages.

A mode switch starts a new episode/reset. No mid-rollout lifting rule is required.

---

# 10. Vessel definition

A resolved vessel definition contains:

- identity/version/hash/provenance;
- visual geometry references;
- collision geometry references;
- mass, CG, inertia;
- buoyancy/restoring parameters;
- full 6-DOF hydrodynamic parameters;
- equilibrium heave/roll/pitch used in planar mode;
- actuator instances and mounts;
- sensor instances and mounts;
- control mappings/autopilot config;
- operating limits;
- validated operating envelope;
- optional energy/health/damage configuration;
- optional wake emitter configuration.

Visual and collision geometry are separate assets.

A CAD file is not a vessel definition until it has been converted into all required runtime fields.

---

# 11. Control and actuator architecture

## 11.1 Two control modes

Every vessel may expose either or both:

1. **High-level control** — e.g. desired heading, speed/throttle, waypoint-relative commands. A vessel-specific autopilot/allocation layer maps these into actuator commands.
2. **Direct actuator control** — policy/user controls the actual actuator command schema.

The config explicitly selects the mode used in an experiment.

## 11.2 Built-in actuator classes

Initial built-ins should include:

- fixed-axis thruster;
- steerable azimuth pod / azipod;
- propeller + rudder;
- generic scalar/vector actuator base for SDK extensions.

Each actuator declares:

- owner vessel;
- body-frame mount position and orientation;
- command schema;
- physical command bounds;
- rate limits/deadband if modeled;
- actuator dynamics/time constant if modeled;
- force/moment mapping;
- power/energy mapping if modeled;
- stable unique instance ID.

## 11.3 Force accumulation

Actuator wrenches are batched by compatible actuator schema. Each row carries `env_id` and `owner_vessel_id`. Contributions are reduced deterministically onto vessel wrench buffers.

## 11.4 Symmetry tests

For symmetric dual-actuator vessels, permanent property tests must include:

- equal mirrored thrust → zero net yaw within tolerance;
- mirrored geometry + mirrored commands → mirrored trajectory;
- sign test for differential thrust.

---

# 12. World architecture

The world owns everything external to vessels:

- water-current field;
- waves;
- air/wind field;
- weather/perception conditions including rain/fog/visibility;
- bathymetry;
- static obstacles and charted objects;
- dynamic non-learning traffic;
- collision/contact environment;
- shared wake/disturbance field.

Vessels sample the world; they do not own environmental state.

## 12.1 Field interface

A continuous field implements a batch query interface conceptually like:

```python
sample(position_ned, sim_time, env_id) -> values
```

Runtime fields should be local tensors/grids/functions prepared at world-build time. External network APIs are not queried in the simulation hot loop.

## 12.2 Parametric world initialization

Must support config-defined:

- uniform or analytic currents;
- wind;
- basic wave models;
- visibility/rain/fog parameters;
- bathymetry primitives;
- static obstacles;
- dynamic scripted entities;
- spawn regions and boundaries.

## 12.3 Real-world initialization

Real-world import is a preprocessing/build pipeline. It produces a frozen local world asset bundle in canonical coordinates and units.

Pipeline:

1. user supplies geographic origin/extent;
2. resolve required data products;
3. fetch and checksum payloads;
4. validate coverage, time validity, units, datum, and source metadata;
5. convert geographic coordinates to the local NED frame;
6. convert vertical values to canonical NED semantics;
7. resample continuous products into canonical local fields;
8. convert charted objects into canonical obstacle/world entities;
9. optionally construct traffic entities from AIS;
10. write a resolved world bundle and provenance manifest;
11. only then launch simulation.

If a requested required product is unavailable or out of coverage, world construction fails. There is no zero fill and no silent substitution.

---

# 13. Coordinated V1 real-world data stack

V1 uses one canonical source per primary quantity to avoid implicit source-selection logic.

## 13.1 GEBCO — bathymetry

Role: seabed/topographic geometry.

Requirements:

- preserve source product/version;
- declare source vertical convention/datum;
- convert explicitly into NED Z;
- record payload/checksum or dataset tile hashes;
- reject ambiguous vertical semantics.

## 13.2 NOAA ENC — coastline/charted objects

Role: shoreline/chart geometry, buoys, aids to navigation, known hazards/objects.

Imported objects must become actual world entities with identity, geometry/collision semantics, and source provenance. They are not merely visual markers.

## 13.3 RTOFS 3D — water currents

Role: current field.

Use the actual requested depth product; surface current uses the 0 m level, not the depth-averaged/barotropic `2ds` quantity.

Coverage must be explicit. Out-of-coverage values are errors.

## 13.4 NDBC — waves

Role: measured marine wave conditions from station observations.

Station radius/coverage is explicit. Do not extrapolate silently beyond coverage.

## 13.5 NWS — wind/weather

Role: atmospheric weather/wind.

All incoming units are explicitly converted to SI. The historical km/h→m/s failure becomes a regression test.

## 13.6 CO-OPS — tides/water level

Role: coastal water level/tidal state and related station data where used.

Datum must be explicit and recorded.

## 13.7 AIS — traffic

Role: non-learning/replayed vessel traffic.

Because legacy AIS extraction provenance is incomplete, the new adapter is built from authoritative source documentation and fresh fixtures rather than ported assumptions.

## 13.8 V1 time semantics

Historical replay/climatology is deferred. Initial real-world import uses a resolved **latest/current world-build snapshot**. Every source returns and records its actual valid time. A source whose current/latest data cannot satisfy the world-build request causes an error.

Future historical/forecast replay may extend the adapter contract without changing the world representation.

---

# 14. Obstacles, collision, and contact

Legacy collision dynamics are not migrated; this subsystem is built fresh.

## 14.1 Collision geometry

Each physical entity exposes one or more collision shapes separate from rendering geometry.

Initial supported shapes:

- sphere;
- box;
- capsule/cylinder as useful;
- convex hull;
- compound of convex shapes;
- static triangle mesh for world geometry, with preprocessing/acceleration structure.

Complex vessel meshes should use simplified convex/compound collision proxies for training throughput.

## 14.2 Detection pipeline

- deterministic broad phase, preferably spatial hash/grid or deterministic sweep-and-prune per environment;
- deterministic narrow phase for supported shape pairs;
- stable contact IDs and stable sorted contact ordering.

## 14.3 Contact resolution

Default V1 response: deterministic impulse/constraint-based rigid-body contact with configurable restitution and friction.

Requirements:

- uses the vessel's current inertial model consistently;
- produces explicit contact wrench/impulse diagnostics;
- does not depend on entity iteration order;
- does not create/lose energy beyond configured restitution/friction behavior;
- detects nonfinite/unstable contact solutions and fails visibly.

The exact solver may later be replaced behind the interface, but V1 must ship with one tested implementation.

## 14.4 Damage

Persistent damage is an optional extension point, not required for core completion. If enabled later, contact events may degrade actuators/sensors/health; task logic decides whether the episode terminates.

---

# 15. Shared wake / environment-mediated coupling

The world maintains a double-buffered shared disturbance field.

At step `t`:

- every vessel reads only the immutable current field;
- vessel wake emitters compute contributions based on current state;
- contributions are accumulated into the next buffer using deterministic reduction;
- the next buffer becomes current only at the step boundary.

No agent can affect another earlier in the same step merely because it was processed first.

## 15.1 Wake model interface

A wake emitter declares:

- source vessel;
- parameter set/provenance;
- spatial support;
- batch function from vessel state to disturbance-field contribution.

Ship the interface and a simple parameterized baseline wake kernel suitable for fitting. The scientific claim depends on later validation against the two-Surveyor interaction data; do not label it high fidelity before validation.

---

# 16. Sensors

## 16.1 Sensor contract

Every sensor declares:

- sensor instance ID and type;
- owner vessel;
- mount position and orientation relative to FRD body frame;
- update rate;
- integer-step latency;
- noise model and seed stream;
- range/FOV/resolution where applicable;
- output units;
- output frame;
- output schema;
- provenance/version.

The master clock drives sensor timing.

## 16.2 Mount kinematics

Physical sensor pose is computed from full vessel pose and mount transform.

Sensor-point velocity includes rigid-body lever-arm motion:

`v_sensor = v_origin + ω × r`

Frame derivatives/accelerations must include the relevant rotating-frame transport terms. The old missing `ω×v` and `ω×r` defects become mandatory regression tests.

## 16.3 Built-in V1 sensors

Required built-ins:

- GPS;
- IMU;
- LiDAR;
- sonar.

The architecture supports camera/radar/custom sensors, but they are not required for the minimum core unless already needed by a planned experiment. If implemented, configured FOV/range/resolution must be enforced.

## 16.4 Physical vs abstract vs ground truth

Three explicit modes:

1. **Physical sensor** — simulates the declared measurement process.
2. **Abstract sensor** — derives structured observations from ground truth but is constrained by declared range/FOV and simple seed-driven numeric noise. V1 does not simulate false positives/missed detections.
3. **Ground-truth/debug channel** — directly exposes state for debugging/baselines and is never misrepresented as a physical sensor.

## 16.5 No ground-truth leakage

An abstract/physical sensor must not depend on world state it is declared unable to observe. Property tests perturb hidden state and verify the sensor output is unchanged.

---

# 17. Observation architecture

The simulator does not force every agent to have the same observation shape.

The observation assembler takes delivered sensor outputs and configured abstract/derived channels and creates each agent's declared observation schema.

Each observation contract has a stable hash containing:

- ordered fields/modalities;
- shapes;
- dtypes;
- units;
- frames;
- normalization/preprocessing definition;
- sensor/abstract-channel identity.

A policy bundle will not run unless its expected observation-contract hash matches or an explicit compatible adapter is declared.

---

# 18. Tasks, rewards, and agent lifecycle

## 18.1 Config-first task model

V1 ships simple task primitives and composable reward/termination terms. Normal task variation requires config changes, not code.

Starter task types may include:

- waypoint/navigation;
- multi-agent waypoint/team navigation;
- formation keeping;
- coverage/search.

Concrete paper benchmarks remain a study-design decision.

## 18.2 Reward model

A task may produce:

- individual reward components per agent;
- team/shared reward components;
- weighted combinations.

Every reward component is separately loggable.

## 18.3 Termination

Separate:

- task success;
- task failure;
- agent disabled;
- physics/numerical failure;
- invalid external data/runtime contract failure;
- time limit.

Every termination has a structured reason code.

## 18.4 Disabled agents

Task-configurable options include:

- terminate the entire episode;
- deactivate the RL agent but keep the vessel as a physical entity;
- explicit removal when a task genuinely requires it.

Default recommended behavior for nonterminal disablement: `deactivate_keep_physical`.

## 18.5 Non-learning traffic/controllers

A reactive vessel is a full physical entity. Its controller can be:

- learned policy;
- scripted controller;
- autopilot;
- replay controller;
- human/manual controller.

“Agent” must not imply “RL-controlled.”

---

# 19. Scenario generation

A scenario is a resolved composition of:

- world;
- vessel instances/spawns;
- task;
- controllers;
- episode limits;
- randomization parameters.

A scenario generator accepts a template plus seeded distributions/ranges and produces a fully resolved scenario for each episode.

Every sampled scenario is serializable and stored or reproducibly regenerable from its seed and template hash.

Supported distribution primitives should include at least fixed, uniform, categorical, normal-with-bounds, and sampled-from-list.

Adaptive teacher/curriculum generation is future work built above this interface.

---

# 20. Heterogeneous batching and parallel execution

## 20.1 Primary batching axis

Throughput comes from **environment instances × vessels × compatible components**.

The implementation must not assume a single environment contains enough agents to saturate a GPU.

## 20.2 Flattened population tables

Maintain flat tables across all environments, with stable mappings:

```text
environment table
vessel table:        env_id, local_id, state...
actuator tables:     env_id, owner_vessel_id, schema-specific state...
sensor tables:       env_id, owner_vessel_id, schema-specific state...
contact tables:      env_id, entity_a, entity_b, contact_id...
```

## 20.3 Schema-compatible groups

Batch together components only when their operation/schema is compatible. Example groups:

- fixed thrusters;
- azipods;
- IMUs with compatible output schema;
- GPS sensors;
- same-resolution LiDAR configurations, with bucketing if needed.

Do not force all sensors or actuators into a single global padded tensor solely to simplify batching.

## 20.4 Deterministic many-to-one reduction

For component forces/moments, wake emissions, metrics, and other many-to-one operations:

- deterministic mode uses stable sort/group by `(env_id, owner_vessel_id, component_id)` or equivalent deterministic segmented reduction;
- unordered GPU atomic accumulation is forbidden on paths claiming exact deterministic behavior.

The reduction implementation must have tests showing invariance to input component ordering.

## 20.5 Isolation tests

Required:

- environment A action cannot mutate B;
- reordering environments in the batch does not change per-environment results after remapping;
- changing batch size does not change a deterministic rollout;
- adding an unrelated environment does not change another environment's trajectory;
- component order permutations produce the same deterministic reductions.

---

# 21. PettingZoo and RL adapter

The core simulator is not built around PettingZoo; PettingZoo is a thin adapter.

## 21.1 Parallel API semantics

At each common policy tick:

- return observations for active agents;
- accept one action per active agent;
- step the core through the required number of master steps;
- return rewards, terminations, truncations, and infos.

Per-agent observation/action spaces may differ. The adapter must accurately expose each agent's declared contract rather than pretending homogeneity.

## 21.2 Centralized critic state

Optionally expose a `state()`/global training view containing explicitly declared information for centralized-training/decentralized-execution algorithms. It is separate from per-agent observations and never leaks into deployed decentralized observations accidentally.

## 21.3 Vectorized environments

Provide a native vector wrapper over many independent environments so MARL training does not instantiate Python environment loops per episode where avoidable.

---

# 22. Policy bundle

A portable policy bundle contains:

```text
policy_bundle/
  manifest.json
  model.onnx              # preferred constrained demo format
  observation_contract.json
  action_contract.json
  preprocessing.json
  normalization.json
  training_provenance.json
  optional recurrent_state_schema.json
```

Native PyTorch weights may be allowed for trusted local research workflows, but public/web demonstration should prefer a constrained non-arbitrary-code format such as ONNX.

Manifest includes:

- policy ID/version;
- model hash;
- observation/action contract hashes;
- simulator commit/config hash used in training;
- trainer commit/version;
- seed(s);
- normalization statistics provenance.

The runtime refuses incompatible bundles.

---

# 23. Logging, metrics, and artifacts

## 23.1 Mandatory run artifact layout

```text
run_<id>/
  manifest.json
  config.resolved.yaml
  provenance/
    software.json
    assets.json
    external_data.json
  metrics.parquet
  events.jsonl
  states.parquet          # when enabled
  sensors/                # optional Zarr or modality-specific payloads
  scenario.resolved.yaml
```

## 23.2 Run manifest

Must contain at least:

- run ID;
- UTC start/end;
- resolved config hash;
- git commit and dirty-patch hash if dirty;
- dependency/lockfile hash;
- backend/device/dtype;
- deterministic flag;
- master dt/substeps/policy tick;
- dynamics mode;
- vessel definition hashes;
- asset hashes;
- world-data source URLs/IDs/valid times/checksums;
- random seeds/seed streams;
- policy/trainer provenance if applicable;
- observation/action contract hashes;
- termination reason.

Provenance is computed from runtime-resolved objects, never copied from user-entered validation labels.

## 23.3 Metrics registry

Built-in opt-in metrics include:

- reward and reward components;
- success/completion;
- collision count/severity;
- time to completion;
- distance traveled;
- energy/power if modeled;
- control effort;
- communication cost if enabled;
- per-agent and team aggregates.

Metrics are selectable in config and extendable through a metrics SDK.

## 23.4 Logging performance

Logging is asynchronous and bounded. If the log queue cannot keep up, behavior is explicit according to config (`block`, `drop_noncritical_with_event`, or terminate); silent data loss is not allowed.

---

# 24. Web application

The web app is a first-class **client** of the backend, never a second simulator.

## 24.1 Required workflows

### Vessel authoring

- select canonical/generated vessel;
- edit allowed vessel parameters;
- place/configure actuators;
- place/configure sensors with visible mount pose/FOV;
- validate;
- export canonical vessel config/bundle.

### World authoring

- parametric world creation;
- real-world import;
- place/edit obstacles and waypoints;
- inspect environmental fields;
- export world/scenario config.

### Scenario/task authoring

- place vessels/spawns;
- select basic task;
- configure reward/termination options;
- configure scenario randomization.

### Policy demo

- upload validated policy bundle;
- choose scenario;
- change permitted scenario parameters/obstacles;
- run policy through the same backend;
- visualize trajectories/sensor state/metrics.

## 24.2 Backend API

FastAPI endpoints should operate on canonical configs/artifacts. Suggested resource families:

- `/v1/vessels`
- `/v1/worlds`
- `/v1/scenarios`
- `/v1/validate`
- `/v1/runs`
- `/v1/policies`
- `/v1/artifacts`

Simulation streaming may use WebSocket/SSE outside the physics loop.

## 24.3 Browser-only execution

Not supported in V1. Explicit future extension only.

---

# 25. Vessel generation and calibration subsystem

This subsystem is architecturally separate from runtime simulation.

Every path outputs the same canonical vessel definition.

## 25.1 Supported creation paths

- manual config;
- imported existing vessel definition;
- procedural/random generated vessel;
- CAD/mesh → CFD → coefficient fitting;
- optional short real-world calibration;
- future ML acceleration.

## 25.2 CFD baseline

Recommended baseline adapter: OpenFOAM.

Conceptual pipeline:

1. import/sanitize hull geometry;
2. establish coordinate frame and scale;
3. compute/enter mass properties and actuator geometry;
4. generate a standardized CFD case matrix for required hydrodynamic responses;
5. execute cases;
6. store raw force/moment outputs and CFD metadata;
7. fit the simulator's full 6-DOF coefficient representation;
8. generate a vessel definition with explicit CFD provenance and uncertainty/fit statistics;
9. optionally refine through short real-world system-identification runs.

Do not make the simulator depend on ML coefficient prediction. ML may later predict a prior or accelerate CFD selection, but the deterministic CFD path remains sufficient.

## 25.3 Calibration separation

Validation/comparison harnesses may not secretly tune coefficients. Calibration is an explicit pipeline that outputs a new versioned vessel definition. A validation run uses a frozen definition.

---

# 26. Validation and behavioral acceptance

The old audit suite is preserved as a portable regression asset. Reimplement `audit/lib/adapter.ts` against the new core as soon as the skeleton is executable.

## 26.1 Validation layers

1. schema/config validation;
2. model-specific physical validation;
3. component unit tests;
4. analytic physics anchors;
5. property/metamorphic tests;
6. integration tests;
7. deterministic replay tests;
8. external-data adapter tests;
9. audit-suite regression;
10. real-field validation where available.

## 26.2 Required inherited regression classes

Permanent tests include:

- unknown identity reference must throw;
- unknown config key must throw;
- NaN/nonfinite physical input must throw;
- invalid mass matrix must throw;
- duplicate actuator IDs must throw;
- out-of-range command must error or explicit clamp-event;
- missing sensor plugin must throw;
- missing task must throw;
- out-of-coverage environmental query must error;
- NWS wind converted to SI;
- frame round trips;
- rotating-frame acceleration correction;
- sensor mount offset and `ω×r` lever-arm velocity;
- declared sensor update rates honored;
- range/FOV/resolution honored;
- symmetric thrust symmetry;
- added mass inside mass matrix;
- force-term sum matches total;
- batch isolation;
- deterministic reduction order independence;
- authoritative stepping/no double-step in external harnesses;
- comparison preconditions: timestamps, lengths, valid coordinates, matched plant fingerprints.

## 26.3 Physics anchors for new 6-DOF core

The old dedicated planar kernel is a **reference/oracle only**. It is not imported into production and is not a paper feature.

Use it and its audit evidence to construct verification cases, but validate the new 6-DOF/constrained-mode core independently through:

- simple analytic force+damping cases;
- equilibrium/zero-motion cases;
- no-damping conservation cases;
- timestep convergence;
- added-mass placement;
- symmetry/mirror/rotation/translation properties;
- published MSS Otter values where provenance is sufficiently pinned;
- Surveyor field data for relevant planar behaviors.

Do not require the constrained 6-DOF mode to bit-match the old separately-identified planar plant.

## 26.4 Validation labels

A simulator-generated label like `validated` has no evidentiary status on its own.

The system may compute validation status only from explicit recorded gates, parameter hashes, operating-envelope checks, and source evidence. User-entered labels never override computed status.

---

# 27. External simulator comparison harnesses

Legacy cross-simulator harnesses are reference-only and must be rebuilt later.

The new generalized comparison contract requires each external adapter to expose:

- exact plant fingerprint;
- mass/added-mass/damping/restoring parameters or explicit equivalence declaration;
- coordinate/frame mapping;
- actuator mapping;
- step/clock semantics;
- measured-output definitions and frames;
- iteration/timestamp IDs.

Before computing an error metric, the harness asserts:

- equal/matched sample counts or an explicitly defined resampling operation;
- timestamp alignment;
- valid finite coordinates;
- matched initial conditions;
- matched plant fingerprint/declared intentional differences;
- matched actuator command history.

If preconditions fail, it returns no comparison metric.

This harness work is not required to make the new core usable, but no new parity claim may use the legacy harnesses.

---

# 28. Legacy migration matrix

| Legacy area | Disposition | New-system rule |
|---|---|---|
| Floating vehicle presets (`Otter`, A/B/C, Surveyor aliases) | **DISCARD / REBUILD** | canonical versioned vessel definitions only |
| Vehicle B | **DISCARD** | historical failure evidence only |
| Legacy Vehicle C definition/coefficients | **REFERENCE ONLY** | dual-azipod vessel rebuilt from authoritative geometry/CFD |
| Dedicated planar kernel | **REFERENCE/ORACLE** | not production; used to validate new core |
| Production vehicle dispatch | **DISCARD** | central fail-closed resolver |
| `legacy-production-engine` | **DISCARD** | one new engine path |
| Old internal harness | **DISCARD** | core public API is the harness |
| Old Gazebo/VRX/etc. harnesses | **REFERENCE ONLY** | generalized comparison contract rebuilt later |
| Old sensor implementations | **REBUILD behavior; reuse algorithms selectively** | standard sensor contract + audit gates |
| Sensor SDK concept | **ADAPT** | new typed/versioned plugin contract |
| Parametric world generation | **ADAPT/VERIFY** | canonical World config |
| Real-world import plumbing | **ADAPT/VERIFY** | centralized frames/units/coverage/provenance |
| GEBCO/ENC obstacle import | **ADAPT/VERIFY** | obstacles become physical entities |
| Collision response | **BUILD FRESH** | deterministic tested contact subsystem |
| RTOFS adapter | **REBUILD/ADAPT with gates** | 3D product, explicit coverage, checksums |
| NDBC/CO-OPS good station-radius behavior | **REUSE AS TESTED BEHAVIOR** | reimplement under new adapter contract |
| NWS adapter | **REBUILD** | SI conversion and provenance mandatory |
| ERA5 | **DEFER** | not in coordinated V1 stack |
| AIS | **REBUILD from authoritative source** | legacy output provenance unavailable |
| Old validation labels | **DISCARD** | computed provenance/gate status only |
| Old PPO curves with unknown producer | **DISCARD/REGENERATE** | new policy bundles include trainer/core hashes |
| Audit probes/goldens | **KEEP** | acceptance suite via adapter |
| Evidence archive | **KEEP** | historical record, not runtime dependency |
| Web authoring/demo concepts | **ADAPT** | backend-only physics, canonical configs |

---

# 29. Implementation sequence and phase gates

The coding agent should implement in phases. Do not build the whole system in one unreviewed pass.

## Phase 0 — repository scaffold and legacy isolation

Deliver:

- new package/repo structure;
- legacy tree mounted/read-only or separate;
- CI skeleton;
- architecture/spec/invariants included in repo;
- no production legacy imports.

Gate: static check proves production package has no import dependency on legacy runtime.

## Phase 1 — config, identity, frames, provenance

Implement:

- Pydantic schemas;
- registry/resolver;
- content hashing;
- fail-closed validation;
- NED/FRD/SI conversion module;
- canonical run manifest skeleton.

Gate: all malformed-input and frame-conversion regression tests pass before physics begins.

## Phase 2 — canonical state and 6-DOF dynamics core

Implement:

- vessel state tables;
- Fossen-style 6-DOF plant;
- mass/added-mass/Coriolis/damping/restoring;
- RK4 fixed-step integration;
- force-term diagnostics;
- constrained planar projection mode;
- operating-envelope checks.

Gate: analytic/conservation/timestep/additional-mass/symmetry tests pass.

## Phase 3 — actuators and control

Implement:

- fixed thruster;
- azipod;
- rudder/propeller;
- direct control;
- high-level/autopilot adapter;
- strict bounds/events;
- deterministic wrench accumulation.

Gate: actuator sign/symmetry/owner-isolation tests pass.

## Phase 4 — core world and parametric environment

Implement:

- current/wind/wave/visibility interfaces;
- parametric fields;
- bathymetry representation;
- static/dynamic entity abstraction;
- world sampling APIs.

Gate: world queries deterministic and unit/frame-correct.

## Phase 5 — sensors and observations

Implement:

- scheduler/latency/noise;
- GPS, IMU, LiDAR, sonar;
- abstract sensors;
- ground-truth debug channels;
- observation contracts/hashes.

Gate: all Lane-L6-derived mount/rate/FOV/frame tests pass for implemented sensors.

## Phase 6 — collision/contact and shared wake field

Implement:

- collision shapes;
- deterministic broad/narrow phase;
- contact solver;
- full6 contact response;
- planar projection response;
- double-buffer wake field and emitter interface.

Gate: collision conservation/dissipation tests, ordering invariance, no agent-order wake dependence.

## Phase 7 — task/scenario/episode system

Implement:

- basic task primitives;
- reward composition;
- agent lifecycle;
- seeded scenario generator;
- termination codes.

Gate: reset/checkpoint/episode lifecycle suite passes.

## Phase 8 — heterogeneous batching and RL adapters

Implement:

- flattened env×vessel population tables;
- component schema grouping;
- deterministic segmented reductions;
- vector environment;
- PettingZoo adapter;
- optional centralized critic state.

Gate: batch-size/order/isolation/determinism tests pass.

## Phase 9 — real-world data import stack

Implement in this order:

1. GEBCO;
2. ENC;
3. RTOFS 3D;
4. NDBC;
5. NWS;
6. CO-OPS;
7. AIS.

Gate for each adapter before enabling it in production: payload checksum/provenance, coverage tests, unit tests, time/datum tests, known-fixture comparison.

## Phase 10 — logging, metrics, and policy bundle

Implement:

- artifact layout;
- async recorder;
- metrics registry/SDK;
- policy bundle validation/runtime;
- ONNX demo path.

Gate: run reproducible from artifact bundle and policy-contract mismatches fail.

## Phase 11 — web client

Implement authoring and demo workflows against backend APIs only.

Gate: exported web configs run identically through headless API; no duplicated frontend physics logic.

## Phase 12 — vessel generation/calibration

Implement:

- procedural/manual vessel creation;
- OpenFOAM adapter;
- coefficient fitting pipeline;
- explicit calibration pipeline.

Gate: generated vessel definition is complete, versioned, and passes construction/physics validation before runtime use.

## Phase 13 — full audit migration gate

Implement new `audit/lib/adapter.ts` equivalent against the new core and run all portable probes plus the new invariants.

Gate: no unexpected FAIL. Any intentional behavior change is documented with a new oracle/test rather than simply suppressing the legacy probe.

---

# 30. CI and release gates

Every pull request should run at least:

- format/lint/type checks;
- config fail-closed tests;
- frame/unit property tests;
- unit/component tests;
- fast analytic physics tests;
- deterministic CPU regression subset;
- sensor contract subset.

Nightly/extended CI:

- full audit suite;
- GPU deterministic tests;
- large-batch isolation tests;
- real-world data fixtures/live-checks where network policy allows;
- web/headless equivalence tests;
- CFD smoke cases after that subsystem exists.

A release is blocked by:

- any unresolved silent-wrong-answer failure;
- any production fallback for an identity-bearing reference;
- nonfinite state on a validated scenario;
- provenance manifest mismatch;
- deterministic regression on a supported deterministic profile;
- external adapter returning values outside declared coverage without error.

---

# 31. Deliberately deferred items

These are not omissions from the implementation plan; they are explicit future work unless promoted later:

- historical weather/ocean replay and climatology;
- adaptive teacher/curriculum generator;
- ML geometry→coefficient prediction;
- browser-only simulation;
- persistent damage model;
- high-fidelity communication/channel model;
- false-positive/missed-detection perception error models;
- arbitrary asynchronous per-agent policy rates;
- full final SDK surface beyond clearly required sensors/actuators/metrics/tasks/policies/vessel models;
- final performance targets until profiling;
- final AAMAS benchmark suite and algorithm baselines;
- new external-simulator parity claims until the generalized harness exists.

---

# 32. Definition of done for the full rewrite

The system is implementation-complete when all of the following are true:

1. No production code imports the legacy runtime.
2. One canonical simulation engine powers headless, RL, and web workflows.
3. The primary 6-DOF plant and constrained planar mode pass the agreed analytic/property/regression gates.
4. A scenario cannot mix dynamics modes among vessels.
5. Surveyor, reference Otter, and a new dual-azipod vessel all load through the same vessel contract.
6. Config/reference failures are fail-closed; no silent defaults remain.
7. NED/FRD/SI are enforced internally through one conversion subsystem.
8. Direct and high-level control both work, with heterogeneous actuator layouts.
9. GPS/IMU/LiDAR/sonar honor mounts, rates, latency, noise, range/FOV/resolution contracts where applicable.
10. Parametric and real-world worlds both resolve to the same World representation.
11. GEBCO/ENC/RTOFS/NDBC/NWS/CO-OPS/AIS adapters implement coverage, units, provenance, and checksum rules.
12. Collision dynamics exist and are deterministic/tested.
13. Shared wake fields are double-buffered and ordering-independent.
14. Tasks/rewards/agent lifecycle are config-driven.
15. Scenario generation is seeded and reproducible.
16. Heterogeneous env×vessel/component batching works without requiring identical full-vehicle tensors.
17. Deterministic mode uses deterministic many-to-one reductions.
18. PettingZoo Parallel API and vectorized training adapter work through the same core.
19. Every run emits a complete provenance artifact.
20. Policy bundles validate observation/action contracts and can drive the web demo.
21. The web app authors/exports canonical configs and never implements physics independently.
22. CFD/calibration are separate vessel-generation workflows, not hidden validation tuning.
23. The portable audit suite runs against the new core and all accepted gates pass.
24. Known legacy silent-wrong-answer classes have explicit structural prevention or regression tests.

---

# 33. Coding-agent operating rules

The implementation agent must follow these rules:

1. **Do not preserve legacy behavior for compatibility unless this spec explicitly requires it.**
2. **Do not introduce compatibility shims that recreate old architectural paths.**
3. **Do not add a fallback when resolution fails. Stop and report the conflict.**
4. **Do not silently clamp, zero-fill, drop a plugin, or substitute a default.**
5. **Do not create a second simulation stepping path for tests, web, or validation.**
6. **Do not let the web frontend calculate authoritative physics.**
7. **Do not use legacy run labels as validation evidence.**
8. **Do not migrate a legacy subsystem until its migration disposition is known.**
9. **When code and specification disagree, follow the specification and flag the legacy conflict.**
10. **Every migrated behavior gets a test before the old implementation is retired.**
11. **Every physical term must remain inspectable through force/wrench diagnostics.**
12. **Every external data adapter must declare units, frame, coverage, valid time, and checksum/provenance.**
13. **Every run must be reproducible from its resolved artifacts within the stated same-backend guarantee.**
14. **Implement phases in order and satisfy each phase gate before building higher-level features on top.**

If the agent encounters an ambiguity that would change a locked invariant, it must stop and ask rather than invent a new architectural decision.

---

# Appendix A — Minimal canonical error taxonomy

Use structured exceptions/events, not generic strings. Suggested families:

- `ConfigSchemaError`
- `UnknownReferenceError`
- `DuplicateIdentityError`
- `PhysicalValidationError`
- `OperatingEnvelopeError`
- `CommandBoundsError`
- `FrameConversionError`
- `ExternalDataUnavailableError`
- `ExternalDataCoverageError`
- `ExternalDataUnitError`
- `ExternalDataDatumError`
- `NumericalInstabilityError`
- `NonFiniteStateError`
- `CollisionSolverError`
- `PolicyContractMismatchError`
- `ObservationContractMismatchError`
- `DeterminismViolationError`
- `ComparisonPreconditionError`

Every fatal run error is reflected in the run manifest termination reason.

---

# Appendix B — Minimal event taxonomy

Structured nonfatal events should include at least:

- actuator clamp (only when explicitly permitted);
- operating-envelope warning/exit when configured to continue;
- sensor sample produced/dropped/late if relevant;
- collision/contact event;
- agent disabled/reactivated if supported;
- environment data source selected;
- wake-field buffer swap/debug event in trace mode;
- logging backpressure/drop event;
- policy action held between policy ticks;
- validation gate result.

---

# Appendix C — What the rewrite must not inherit

The following are specifically prohibited because they were observed historically:

- unresolved vehicle IDs mapping to demo geometry;
- downstream metadata claiming validation independently of loaded parameters;
- unknown config fields accepted;
- silent actuator clipping;
- missing sensor plugins silently dropped;
- unknown tasks replaced with a default waypoint;
- malformed mass/inertia accepted;
- environment values returned outside coverage without a signal;
- NWS values consumed without unit conversion;
- distributed frame/sign conversions;
- sensor mount offsets and lever-arm kinematics ignored;
- sensor rate/FOV/resolution configs treated as decorative;
- comparison metrics computed on misaligned/unequal data;
- validation on a different execution path from production;
- shared mutable action objects across environments;
- retry-based simulator double stepping;
- coefficient tuning hidden inside a validation harness;
- historical policy artifacts without producer/trainer provenance.

These are acceptance requirements, not merely historical notes.
