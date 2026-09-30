import copy
import base64
import json
from pathlib import Path
import time
import yaml

from bcod_sim.web.api import create_app
from bcod_sim.web.qualification import qualify
from tests.web.test_phase11_web import ASGIClient


def preset():
    return json.loads((Path(__file__).parents[1]/"fixtures/web/experiment_navigation.json").read_text())


def test_navigation_qualification_and_missing_sensor(tmp_path):
    payload = preset()
    report = qualify(**payload, artifact_root=tmp_path)
    assert report["status"] == "PASS"
    assert (tmp_path/"qualifications"/f"{report['experiment_hash']}.json").exists()
    broken = copy.deepcopy(payload)
    broken["config"]["vessels"][0]["sensors"].remove("sonar@1")
    report = qualify(**broken)
    assert report["status"] == "FAIL"
    assert any(c["code"] == "MISSING_NAVIGATION_SENSOR" for c in report["checks"])
    invalid_spawn = copy.deepcopy(payload)
    invalid_spawn["config"]["vessels"][0]["spawn"]["ned_m"] = [7, 3, 0]
    report = qualify(**invalid_spawn)
    assert any(c["code"] == "INVALID_SPAWN" for c in report["checks"])
    bad = yaml.safe_load((Path(__file__).parents[1]/"fixtures/web/vessel_bad_actuator.yaml").read_text())
    report = qualify(**bad)
    assert any(c["code"] == "CONTROL_INTERFACE" for c in report["checks"])


def test_complete_experiment_json_yaml_round_trip(tmp_path):
    client = ASGIClient(create_app(artifact_root=tmp_path, project_root=Path(__file__).parents[2]))
    payload = preset()
    for format in ("json", "yaml"):
        exported = client.post("/v1/experiments/serialize", json={**payload, "format": format})
        assert exported.status_code == 200, exported.text
        imported = client.post("/v1/experiments/parse", json={"content": exported.json()["content"], "format": format})
        assert imported.status_code == 200, imported.text
        assert client.post("/v1/validate", json=imported.json()).json()["config_hash"] == exported.json()["experiment_hash"]
    duplicate = client.post("/v1/experiments/parse", json={"content": "config: {}\nconfig: {}\ndefinitions: []\n", "format": "yaml"})
    assert duplicate.status_code == 422
    components = {
        "vessel": next(d for d in payload["definitions"] if d["kind"] == "vessel")["payload"],
        "environment": payload["config"]["world"]["environment"],
        "scenario": payload["config"],
        "task": next(d for d in payload["definitions"] if d["kind"] == "task")["payload"],
    }
    for kind, component in components.items():
        imported = client.post(f"/v1/configs/{kind}/import", json={"content": yaml.safe_dump(component), "format": "yaml"})
        assert imported.status_code == 200, imported.text
        exported = client.post(f"/v1/configs/{kind}/export", json={"document": imported.json(), "format": "yaml"})
        assert exported.status_code == 200, exported.text
        again = client.post(f"/v1/configs/{kind}/import", json={"content": exported.json()["content"], "format": "yaml"})
        assert again.json() == imported.json()
    assembly_text = (Path(__file__).parents[1]/"fixtures/web/vessel_assembly.yaml").read_text()
    assembly = client.post("/v1/vessel-assembly/parse", json={"content": assembly_text, "format": "yaml"})
    assert assembly.status_code == 200, assembly.text
    assert len(assembly.json()["definitions"]) == 6
    assembled = client.post("/v1/vessel-assembly/serialize", json={"document": assembly.json(), "format": "yaml"})
    assert assembled.status_code == 200
    assert client.post("/v1/vessel-assembly/parse", json={"content": assembled.json()["content"], "format": "yaml"}).json() == assembly.json()


def test_local_runner_reconnect_metrics_replay_stop_and_artifact(tmp_path):
    project = Path(__file__).parents[2]
    app = create_app(artifact_root=tmp_path, project_root=project)
    client = ASGIClient(app)
    payload = preset()
    assert client.get("/v1/presets/navigation").json() == payload
    assert any(x["type"] == "sonar" for x in client.get("/v1/component-types/sensors").json())
    assert client.get("/v1/runners").json()[0]["resources"]["cpu_count"] >= 1
    request = {**payload, "run_id": "navigation-test", "step_delay_s": .001}
    assert client.post("/v1/jobs", json=request).status_code == 200
    # A fresh app instance reads the same durable job directory after page/backend reconnect.
    other = ASGIClient(create_app(artifact_root=tmp_path, project_root=project))
    for _ in range(120):
        status = other.get("/v1/jobs/navigation-test").json()
        if status["state"] not in {"QUEUED", "RUNNING"}: break
        time.sleep(.1)
    assert status["state"] == "COMPLETED", status
    assert status["step"] > 0
    assert other.get("/v1/jobs/navigation-test/metrics").json()["items"]
    frames = other.get("/v1/jobs/navigation-test/frames?limit=2").json()["items"]
    assert frames[1]["action"]["agent"]["desired_speed_mps"] == 1
    assert set(frames[1]["actuator_commands"]) == {"port", "starboard"}
    assert Path(status["artifact_path"], "manifest.json").is_file()
    assert any(a["path"] == "manifest.json" for a in other.get("/v1/jobs/navigation-test/artifacts").json()["artifacts"])
    assert other.get("/v1/jobs/navigation-test/logs").json()["items"]
    expected = json.loads((project/"tests/fixtures/web/replay_golden.json").read_text())
    replay = other.get("/v1/jobs/navigation-test/frames?limit=4").json()["items"]
    for gold, actual in zip(expected, replay):
        assert gold["step"] == actual["master_step"]
        assert [round(x, 9) for x in actual["states"]["agent"]["position_ned_m"]] == gold["position_ned_m"]
    trajectory = other.get("/v1/jobs/navigation-test/trajectory?max_points=10").json()
    assert len(trajectory["frames"]) <= 10
    assert trajectory["frames"][0]["master_step"] == 0


def test_graceful_stop_and_hard_kill(tmp_path):
    project = Path(__file__).parents[2]
    client = ASGIClient(create_app(artifact_root=tmp_path, project_root=project))
    payload = preset(); payload["config"]["simulation"]["max_master_steps"] = 1000
    for name, control in (("stop-test", "stop"), ("kill-test", "kill")):
        assert client.post("/v1/jobs", json={**payload, "run_id": name, "step_delay_s": .05}).status_code == 200
        for _ in range(100):
            if client.get(f"/v1/jobs/{name}").json()["state"] == "RUNNING": break
            time.sleep(.05)
        assert client.post(f"/v1/jobs/{name}/{control}").status_code == 200
        for _ in range(100):
            status = client.get(f"/v1/jobs/{name}").json()
            if status["state"] not in {"QUEUED", "RUNNING"}: break
            time.sleep(.05)
        assert status["state"] == ("STOPPED" if control == "stop" else "KILLED")


def test_uploaded_policy_evaluates_and_replays(tmp_path):
    project = Path(__file__).parents[2]
    client = ASGIClient(create_app(artifact_root=tmp_path, project_root=project))
    bundle = project/"tests/fixtures/web/short_policy_bundle"
    manifest = json.loads((bundle/"manifest.json").read_text())
    files = {path.name: base64.b64encode(path.read_bytes()).decode() for path in bundle.iterdir()}
    upload = client.post("/v1/policies", json={"policy_id": manifest["policy_id"],
        "files_base64": files, "observation_contract_hash": manifest["observation_contract_hash"],
        "action_contract_hash": manifest["action_contract_hash"]})
    assert upload.status_code == 200, upload.text
    assert manifest["policy_id"] in client.get("/v1/policies").json()
    response = client.post("/v1/jobs", json={**preset(), "run_id": "policy-eval",
        "mode": "policy_evaluation", "policy_id": manifest["policy_id"]})
    assert response.status_code == 200, response.text
    for _ in range(120):
        status = client.get("/v1/jobs/policy-eval").json()
        if status["state"] not in {"QUEUED", "RUNNING"}: break
        time.sleep(.1)
    assert status["state"] == "COMPLETED", status
    assert status["success"] is True
    replay = client.get("/v1/jobs/policy-eval/frames?offset=1&limit=1").json()["items"][0]
    assert len(replay["observation"]) == 8
    assert replay["action"]["agent"]["desired_speed_mps"] == 1
