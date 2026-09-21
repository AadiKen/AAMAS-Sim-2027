from pathlib import Path
import asyncio

import httpx

from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.web.api import create_app
from bcod_sim.web.runtime_factory import build_engine
from bcod_sim.web.service import build_registry, serialize_frame


class ASGIClient:
    def __init__(self, app): self.app = app
    def request(self, method, path, **kwargs):
        async def call():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
                return await client.request(method, path, follow_redirects=True, **kwargs)
        return asyncio.run(call())
    def get(self, path, **kwargs): return self.request("GET", path, **kwargs)
    def post(self, path, **kwargs): return self.request("POST", path, **kwargs)


def web_payload():
    vessel = {"mass_kg": 10, "cg_frd_m": [0, 0, 0],
        "inertia_cg_kg_m2": [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
        "added_mass_kg": [[0, 0, 0, 0, 0, 0]]*6,
        "linear_damping": [0, 0, 0, 0, 0, 0], "quadratic_damping": [0, 0, 0, 0, 0, 0],
        "buoyancy_n": 98.0665, "center_buoyancy_frd_m": [0, 0, 0],
        "max_abs_nu": [100, 100, 100, 100, 100, 100], "max_substep_s": 0.2,
        "collision": {"kind": "sphere", "radius_m": 0.2},
        "environment_loads": [0, 0, 0, 0]}
    actuator = {"kind": "fixed_thruster", "mount_frd_m": [0, 0, 0],
        "mount_q_to_frd": [1, 0, 0, 0], "thrust_bounds_n": [-100, 100]}
    task = {"kind": "waypoint", "agent_id": "agent", "target_ned_m": [100, 0, 0], "radius_m": 0.1}
    definitions = [
        {"kind": "vessel", "id": "web-vessel", "version": "1", "payload": vessel, "source": "web"},
        {"kind": "actuator", "id": "prop", "version": "1", "payload": actuator, "source": "web"},
        {"kind": "task", "id": "waypoint", "version": "1", "payload": task, "source": "web"}]
    config = {"schema_version": 1, "experiment": {"id": "web", "seed": 7},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.1, "dynamics_substeps": 1,
                       "policy_every_n_master_steps": 1, "max_master_steps": 2},
        "world": {"source": {"kind": "parametric"}, "environment": {
            "current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]}, "waves": {"kind": "calm"},
            "visibility_m": 1000}, "bathymetry": {"kind": "flat", "bottom_ned_z_m": 20,
                                                   "vertical_datum": "MSL"}},
        "vessels": [{"instance_id": "agent", "definition": "web-vessel@1", "actuators": ["prop@1"],
            "controller": {"mode": "direct_actuator"}, "spawn": {"ned_m": [0, 0, 0], "rpy_rad": [0, 0, 0]}}],
        "task": {"type": "waypoint@1", "reward": {"individual_weight": 1, "team_weight": 0},
                 "disabled_agent_behavior": "deactivate_keep_physical"},
        "logging": {"metrics": ["reward", "distance_traveled"], "states": True,
                    "sensor_payloads": False, "queue_capacity": 16, "backpressure": "block"}}
    actions = [{"agent": {"prop": 10}}, {"agent": {"prop": 10}}]
    return config, definitions, actions


def test_web_export_runs_identically_through_api_and_headless(tmp_path):
    project = Path(__file__).parents[2]
    client = ASGIClient(create_app(artifact_root=tmp_path/"runs", project_root=project))
    config, definitions, actions = web_payload()
    validated = client.post("/v1/validate", json={"config": config, "definitions": definitions})
    assert validated.status_code == 200
    response = client.post("/v1/runs", json={"run_id": "web", "config": config,
        "definitions": definitions, "actions": actions})
    assert response.status_code == 200, response.text
    web_run = response.json()
    resolved = resolve(config, build_registry(definitions)); engine = build_engine(resolved)
    frames = [serialize_frame(engine.reset())]
    for payload in actions:
        typed = client.app.state.service._actions(engine, payload)
        frames.append(serialize_frame(engine.step(typed)))
    assert web_run["config_hash"] == resolved.content_hash
    assert web_run["frames"] == frames
    assert web_run["frames"][-1]["termination_reason"] == "time_limit"
    manifest = client.get("/v1/artifacts/web/manifest.json")
    assert manifest.status_code == 200 and manifest.json()["config_hash"] == resolved.content_hash


def test_authoring_resources_streaming_and_fail_closed_validation(tmp_path):
    client = ASGIClient(create_app(artifact_root=tmp_path/"runs", project_root=Path(__file__).parents[2]))
    for family in ("vessels", "worlds", "scenarios"):
        assert client.post(f"/v1/{family}", json={"id": "draft", "document": {"name": family}}).status_code == 200
        assert client.get(f"/v1/{family}").json()["draft"] == {"name": family}
        assert client.post(f"/v1/{family}", json={"id": "draft", "document": {}}).status_code == 422
    config, definitions, actions = web_payload()
    bad = {**config, "unknown": True}
    response = client.post("/v1/validate", json={"config": bad, "definitions": definitions})
    assert response.status_code == 422 and response.json()["error"] == "ConfigSchemaError"
    client.post("/v1/runs", json={"run_id": "stream", "config": config,
        "definitions": definitions, "actions": actions})
    stream = client.get("/v1/runs/stream/stream")
    assert stream.status_code == 200 and stream.headers["content-type"].startswith("text/event-stream")
    assert stream.text.count("event: frame") == 3
    assert client.get("/v1/artifacts/stream/../pyproject.toml").status_code == 404


def test_built_client_is_served_and_contains_no_simulation_path(tmp_path):
    project = Path(__file__).parents[2]; frontend = project/"web_client"
    client = ASGIClient(create_app(artifact_root=tmp_path/"runs", project_root=project,
                                   frontend_dist=frontend/"dist"))
    assert client.get("/").status_code == 200
    source = "\n".join(path.read_text() for path in (frontend/"src").glob("*.tsx"))
    forbidden = ("integrate", "dynamics", "collision response", "wrench", "nu_body", "position_ned_m")
    assert all(term not in source for term in forbidden)
    assert "position_display_m" in source and "Run backend" in source


def test_authored_sensor_is_constructed_by_backend_runtime():
    config, definitions, _ = web_payload()
    definitions.append({"kind": "sensor", "id": "truth", "version": "1", "source": "web",
                        "payload": {"kind": "ground_truth_state", "rate_hz": 10}})
    config["vessels"][0]["sensors"] = ["truth@1"]
    engine = build_engine(resolve(config, build_registry(definitions)))
    assert engine.vessels["agent"].sensors[0].config.instance_id == "truth"
