"""Self-contained resolved config/scenario artifact serialization and validation."""

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from bcod_sim.config.hashing import canonical_json, content_hash
from bcod_sim.config.models import ExperimentConfig
from bcod_sim.config.registry import Definition, _freeze
from bcod_sim.config.resolver import ResolvedExperiment
from bcod_sim.core.errors import ConfigSchemaError
from bcod_sim.scenario.generator import ResolvedScenario, SpawnState


def _thaw(value):
    if isinstance(value, Mapping): return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple): return [_thaw(child) for child in value]
    return value


def resolved_artifact(resolved: ResolvedExperiment) -> dict[str, Any]:
    return {"config_hash": resolved.content_hash, "config": resolved.config.model_dump(mode="json"),
            "definitions": [{"kind": d.kind, "id": d.id, "version": d.version,
                "payload": _thaw(d.payload), "source": d.source,
                "content_hash": d.content_hash} for d in resolved.definitions]}


def load_resolved_artifact(path: str | Path) -> ResolvedExperiment:
    try:
        document = yaml.safe_load(Path(path).read_text())
        if set(document) != {"config_hash", "config", "definitions"}:
            raise ValueError("sections")
        config = ExperimentConfig.model_validate(document["config"])
        definitions = []
        for item in document["definitions"]:
            if set(item) != {"kind", "id", "version", "payload", "source", "content_hash"}:
                raise ValueError("definition fields")
            if content_hash(item["payload"]) != item["content_hash"]:
                raise ValueError("definition hash")
            definitions.append(Definition(item["kind"], item["id"], item["version"],
                                          _freeze(item["payload"]), item["source"], item["content_hash"]))
        canonical = {"config": config.model_dump(mode="json"), "definitions": [
            {"kind": d.kind, "id": d.id, "version": d.version, "content_hash": d.content_hash,
             "source": d.source} for d in definitions]}
        detached = json.loads(canonical_json(canonical))
        if content_hash(detached) != document["config_hash"]:
            raise ValueError("config hash")
        return ResolvedExperiment(config, tuple(definitions), _freeze(detached), document["config_hash"])
    except Exception as exc:
        if isinstance(exc, ConfigSchemaError): raise
        raise ConfigSchemaError("Resolved config artifact failed integrity validation") from exc


def scenario_artifact(scenario: ResolvedScenario) -> dict[str, Any]:
    return {**scenario.payload(), "content_hash": scenario.content_hash}


def load_scenario_artifact(path: str | Path) -> ResolvedScenario:
    try:
        document = yaml.safe_load(Path(path).read_text())
        expected = document.pop("content_hash")
        if set(document) != {"config_hash", "template_hash", "episode_seed", "episode_index",
                             "max_master_steps", "spawns", "task_reference"}:
            raise ValueError("scenario fields")
        if content_hash(document) != expected:
            raise ValueError("scenario hash")
        spawns = tuple(SpawnState(row["instance_id"], tuple(row["position_ned_m"]), tuple(row["rpy_rad"]),
                                  tuple(row["nu_body"])) for row in document["spawns"])
        return ResolvedScenario(document["config_hash"], document["template_hash"], document["episode_seed"],
            document["episode_index"], document["max_master_steps"], spawns, document["task_reference"], expected)
    except Exception as exc:
        raise ConfigSchemaError("Resolved scenario artifact failed integrity validation") from exc
