# MANTA web MVP quickstart

From the repository root, install the package and web dependencies, build the client, then start the loopback API and local process runner:

```bash
python3 -m pip install -e '.[test]'
npm --prefix web_client ci
npm --prefix web_client run build
manta-runner --project-root "$PWD" --artifact-root "$PWD/.bcod_runs" start --port 8000
```

Open `http://127.0.0.1:8000`. The navigation preset loads automatically. Import a complete experiment JSON/YAML or import the supported vessel, environment, scenario, and task payloads separately. Inspector edits update the shared scene; export JSON/YAML to run headlessly. Run qualification, then start the scripted baseline or upload a compatible seven-file ONNX policy bundle and select policy evaluation. Runs survive page reload. Use Runs for metrics, logs, and artifacts; Replay for timeline seek and action/response inspection.

The same exported experiment can be run without a browser:

```bash
manta-runner --project-root "$PWD" --artifact-root "$PWD/.bcod_runs" run \
  tests/fixtures/web/experiment_navigation.yaml --run-id headless-smoke --wait
```

For tests:

```bash
python3 -m pytest -q tests/web tests/integration tests/unit
npm --prefix web_client test
npm --prefix web_client run build
```

The MVP supports one high-level vessel, flat parametric world, fixed thrusters, GPS/IMU/sonar, a waypoint task, a local runner, scripted baseline, and compatible policy evaluation. It does not train a policy, produce checkpoints, or submit SLURM jobs.
