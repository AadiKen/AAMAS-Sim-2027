# Qualification gates (MVP)

`POST /v1/qualify` returns and persists a report keyed by the resolved experiment hash. Each check has category, name, status, message, metrics, path, and code. The report has timestamp, schema version, and experiment hash.

- Schema: resolve `ExperimentConfig` and all versioned definitions.
- Semantic: validate vessel runtime payload and require symmetric positive definite inertia.
- Cross-object: require GPS, IMU, and sonar for the supported waypoint navigation slice; reject a spawn overlapping a static obstacle or the flat seabed.
- T0: construct and reset the authoritative `EpisodeEngine`.
- T1: run zero, positive surge, positive yaw, and deterministic repeat trials with the actual controller/plant.
- T2: run a bounded scripted waypoint baseline, requiring success without contact.
- T3: reported `NOT_RUN`; a learning smoke is outside this evaluation-only MVP.

The supported runtime gates require one high-level vessel. Other core configurations may validate at schema level but receive `NOT_RUN` for unsupported response tests. A failure during an action trial is categorized as `CONTROL_INTERFACE`, while a construction/reset failure is `SIMULATOR_CONSTRUCTION`.
