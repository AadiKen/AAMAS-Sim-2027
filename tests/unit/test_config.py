import copy
import json

import pytest

from bcod_sim.config.hashing import content_hash
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import load_config, resolve
from bcod_sim.core.errors import ConfigSchemaError, DuplicateIdentityError, UnknownReferenceError, ExternalDataUnavailableError
from bcod_sim.logging.manifest import manifest_skeleton


def example():
    return {
        "schema_version": 1,
        "experiment": {"id": "trial", "seed": 42},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.02, "dynamics_substeps": 2, "policy_every_n_master_steps": 5},
        "world": {"source": {"kind": "parametric"}, "environment": {
            "current": {"kind": "uniform", "ned_mps": [0.1, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [2, 0, 0]},
            "waves": {"kind": "regular", "height_m": 0.2, "period_s": 3.0, "direction_rad": 0},
            "visibility_m": 5000}, "obstacles": []},
        "vessels": [{"instance_id": "leader", "definition": "test_vessel@1", "controller": {"mode": "direct_actuator"}, "spawn": {"ned_m": [0, 0, 0], "rpy_rad": [0, 0, 0]}}],
        "task": {"type": "test_task@1", "reward": {"individual_weight": 0.5, "team_weight": 0.5}, "disabled_agent_behavior": "deactivate_keep_physical"},
    }


def registry():
    r = Registry()
    r.register("vessel", "test_vessel", "1", {"mass_kg": 10}, "test-fixture")
    r.register("task", "test_task", "1", {"kind": "test"}, "test-fixture")
    return r


def test_materialized_defaults_and_hash():
    raw = example()
    resolved = resolve(raw, registry())
    assert resolved.config.logging.states is True
    assert resolved.canonical_payload["config"]["logging"]["states"] is True
    assert resolved.content_hash == content_hash({"config": resolved.config.model_dump(mode="json"), "definitions": [
        {"kind": d.kind, "id": d.id, "version": d.version, "content_hash": d.content_hash, "source": d.source} for d in resolved.definitions]})
    raw["experiment"]["seed"] = 99
    assert resolved.config.experiment.seed == 42
    with pytest.raises(TypeError):
        resolved.canonical_payload["config"]["experiment"]["seed"] = 99


@pytest.mark.parametrize("change", [
    lambda d: d.update(unknown=1),
    lambda d: d["simulation"].update(master_dt_s=float("nan")),
    lambda d: d["simulation"].update(master_dt_s=0),
    lambda d: d["simulation"].update(dynamics_substeps=0),
    lambda d: d["world"]["environment"]["current"].update(ned_mps=[0, 0]),
    lambda d: d["vessels"].append(copy.deepcopy(d["vessels"][0])),
    lambda d: d["vessels"][0]["spawn"].update(rpy_rad=[0, float("inf"), 0]),
    lambda d: d["task"].update(extra_key=True),
])
def test_malformed_config_fails(change):
    raw = example()
    change(raw)
    with pytest.raises(ConfigSchemaError):
        resolve(raw, registry())


def test_unknown_reference_and_duplicate_identity():
    raw = example()
    raw["vessels"][0]["definition"] = "vehicle-a-otter@1"
    with pytest.raises(UnknownReferenceError):
        resolve(raw, registry())
    r = registry()
    with pytest.raises(DuplicateIdentityError):
        r.register("vessel", "test_vessel", "1", {}, "duplicate")
    with pytest.raises(UnknownReferenceError):
        r.resolve("vessel", "test_vessel")


def test_identity_hash_changes_with_payload():
    a = Registry().register("vessel", "a", "1", {"m": 1}, "source")
    b = Registry().register("vessel", "a", "1", {"m": 2}, "source")
    assert a.content_hash != b.content_hash


def test_real_world_unavailable_fails_closed():
    raw = example()
    raw["world"]["source"] = {"kind": "real_world", "data_product": "product@1"}
    raw["world"].pop("environment")
    r = registry()
    r.register("data_product", "product", "1", {}, "test")
    with pytest.raises(ExternalDataUnavailableError):
        resolve(raw, r)


def test_duplicate_serialized_keys_rejected(tmp_path):
    for name, text in [("x.json", '{"a":1,"a":2}'), ("x.yaml", "a: 1\na: 2\n")]:
        path = tmp_path / name
        path.write_text(text)
        with pytest.raises(ConfigSchemaError):
            load_config(path)


def test_manifest_uses_resolved_hash_and_marks_unknown_runtime_facts():
    resolved = resolve(example(), registry())
    manifest = manifest_skeleton(resolved, run_id="test-run")
    assert manifest.config_hash == resolved.content_hash
    assert manifest.definition_hashes["vessel:test_vessel@1"] == resolved.definitions[0].content_hash
    assert manifest.backend is None
    assert manifest.termination_reason is None
