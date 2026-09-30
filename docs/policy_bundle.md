# Policy bundle for web evaluation

Portable inference uses the existing seven-file ONNX `PolicyBundle`: `manifest.json`, `model.onnx`, `observation_contract.json`, `action_contract.json`, `preprocessing.json`, `normalization.json`, and `training_provenance.json`. Upload validates SHA-256 hashes and manifest/contract consistency through `PolicyBundle.load`. Evaluation uses `ONNXPolicyRuntime` and checks its input/output bindings before a simulator step.

The MVP adapter is `navigation_sensors_v1` → `high_level_v1`. It requires GPS, IMU, sonar, an eight-element float observation `[north_m, east_m, velocity_north_mps, velocity_east_mps, yaw_rate_radps, sonar_min_range_m, goal_north_m, goal_east_m]`, and a two-element action `[desired_speed_mps, desired_heading_rad]`. The navigation values come from delivered sensor packets and the canonical task goal. Allowed outputs are speed 0–2 m/s and heading within ±π radians. Any mismatch fails before policy execution.

`tests/fixtures/web/short_policy_bundle` is a synthetic constant-action policy for a deterministic smoke test. Its provenance is marked as a fixture; it is not a trained checkpoint or performance claim. Generic `.pt` uploads and trainer checkpoint promotion are unsupported.
