import copy

import pytest

from bcod_sim.core.errors import ConfigSchemaError
from bcod_sim.web.config_service import diff, parse, serialize, validate
from tests.web.test_phase11_web import ASGIClient, web_payload
from bcod_sim.web.api import create_app
from pathlib import Path


def document(payload):
    return {"schema_version": 1, "payload": payload}


def test_four_canonical_round_trips():
    config, definitions, _ = web_payload()
    payloads = {
        "vessel": definitions[0]["payload"],
        "environment": config["world"]["environment"],
        "scenario": config,
        "task": definitions[2]["payload"],
    }
    for kind, payload in payloads.items():
        original = document(payload)
        normalized = serialize(parse(kind, original))
        assert serialize(parse(kind, normalized)) == normalized
        assert validate(kind, original)["content_hash"] == validate(kind, normalized)["content_hash"]


def test_rejects_unknown_and_invalid_without_losing_fields():
    config, definitions, _ = web_payload()
    bad = copy.deepcopy(definitions[0]["payload"])
    bad["unknown_extension"] = 42
    with pytest.raises(ConfigSchemaError):
        parse("vessel", document(bad))
    with pytest.raises(ConfigSchemaError):
        parse("environment", document({**config["world"]["environment"], "visibility_m": -1}))
    with pytest.raises(ConfigSchemaError):
        parse("task", {"schema_version": 2, "payload": definitions[2]["payload"]})


def test_diff_reports_spatial_edit():
    config, _, _ = web_payload()
    changed = copy.deepcopy(config)
    changed["vessels"][0]["spawn"]["ned_m"][0] = 3
    assert diff("scenario", document(config), document(changed)) == [
        {"path": "/vessels/0/spawn/ned_m/0", "before": 0.0, "after": 3.0}]


def test_config_api_normalize_validate_diff_and_reject(tmp_path):
    client = ASGIClient(create_app(artifact_root=tmp_path, project_root=Path(__file__).parents[2]))
    config, definitions, _ = web_payload()
    vessel = document(definitions[0]["payload"])
    assert client.post("/v1/configs/vessel/normalize", json={"document": vessel}).status_code == 200
    assert client.post("/v1/configs/vessel/validate", json={"document": vessel}).json()["valid"]
    assert client.post("/v1/configs/vessel/migrate", json={"document": vessel}).status_code == 200
    changed = copy.deepcopy(config)
    changed["vessels"][0]["spawn"]["ned_m"][0] = 3
    response = client.post("/v1/configs/scenario/diff", json={"left": document(config),
        "right": document(changed)})
    assert response.json()["changes"][0]["path"] == "/vessels/0/spawn/ned_m/0"
    assert client.post("/v1/configs/vessel/validate", json={"document": document({})}).status_code == 422
