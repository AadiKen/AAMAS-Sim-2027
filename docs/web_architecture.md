# Web control plane architecture (current state)

The backend imports core simulator types. The web config service wraps, validates, hashes, and diffs those types. `resolve` binds `id@version` registry definitions and `build_engine` constructs `EpisodeEngine`; neither operation is reimplemented in TypeScript. The React client provides visual and exact inspector editing, config and diff tabs, qualification, local job controls, and replay. The older synchronous `/v1/runs` endpoint remains for compatibility; the MVP uses `/v1/jobs`.

Dependency direction: React UI → web API → config/qualification/run services → core config, engine, task, sensor, actuator, and logging packages. The local process adapter owns jobs independently from browser connections. It executes only fixed, validated job modes.

## Implemented MVP slice

A versioned transport wraps existing canonical models. The React workspace imports/exports complete experiment bundles and supported component payloads, and uses one `WorldView` for build and replay. Qualification calls the authoritative `EpisodeEngine`; local jobs run in a separate worker process and record canonical `RunRecorder` artifacts. Uploaded ONNX bundles are evaluated through the existing policy runtime with a constrained navigation adapter. The benchmark training schemas remain separate from this core experiment slice.
