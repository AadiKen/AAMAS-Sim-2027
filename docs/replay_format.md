# Replay format (MVP)

Every local job writes `frames.jsonl` with one row per simulator frame, starting at step zero. Rows include `master_step`, `sim_time_s`, per-agent NED and display positions, quaternion, body velocity, reward, termination flags, policy/scripted action, the observation supplied to the policy (if any), actuator commands, contact event IDs, and control diagnostics. The companion `metrics.jsonl` and `logs.jsonl` are paged separately. The browser uses the same `WorldView` renderer for editing and replay and requests a selected frame by step offset; it does not rerun physics to seek.

`GET /v1/jobs/{id}/trajectory?max_points=1000` returns downsampled display positions for the path overlay. It never sends an entire long telemetry history into browser state. Frame seek remains paged from the original JSONL.

The schema leaves action, actuator, observation, event, and diagnostic fields separate so later causal overlays can connect policy output to controller, actuator, force, response, and next observation. The current recorder does not yet persist every force/wrench or full sensor packet in replay JSONL. The canonical `RunRecorder` artifact remains the source for detailed run provenance and optional sensor payloads.
