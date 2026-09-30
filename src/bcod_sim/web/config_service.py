"""Versioned web transport for existing simulator config types.

The envelope is a transport wrapper. Its payload remains a MANTA/BCOD-Sim
model, so the web layer does not own simulation semantics.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from bcod_sim.config.hashing import content_hash
from bcod_sim.config.models import Environment, ExperimentConfig
from bcod_sim.core.errors import ConfigSchemaError
from bcod_sim.tasks.primitives import (CoverageParams, FormationParams,
    TeamWaypointParams, WaypointParams)
from bcod_sim.web.runtime_factory import VesselRuntime

ConfigKind = Literal["vessel", "environment", "scenario", "task"]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConfigDocument:
    kind: ConfigKind
    payload: BaseModel


def _model(kind: ConfigKind, payload: dict[str, Any]) -> BaseModel:
    if kind == "vessel":
        return VesselRuntime.model_validate(payload)
    if kind == "environment":
        return Environment.model_validate(payload)
    if kind == "scenario":
        # A runnable scenario is the core experiment config. A standalone
        # World lacks vessel spawns and task references.
        return ExperimentConfig.model_validate(payload)
    if kind == "task":
        task_kind = payload.get("kind")
        task_models = {"waypoint": WaypointParams, "waypoint_team": TeamWaypointParams,
                       "formation": FormationParams, "coverage": CoverageParams}
        if task_kind in task_models:
            return task_models[task_kind].model_validate(payload)
        raise ConfigSchemaError(f"Unsupported core task definition: {task_kind}")
    raise ConfigSchemaError(f"Unsupported config kind: {kind}")


def parse(kind: ConfigKind, document: dict[str, Any]) -> ConfigDocument:
    if not isinstance(document, dict):
        raise ConfigSchemaError("Config document must be an object")
    if set(document) != {"schema_version", "payload"}:
        raise ConfigSchemaError("Config document requires schema_version and payload only")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ConfigSchemaError("Unsupported web config schema_version")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise ConfigSchemaError("Config payload must be an object")
    try:
        return ConfigDocument(kind, _model(kind, payload))
    except ValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc


def serialize(value: ConfigDocument) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "payload": value.payload.model_dump(mode="json")}


def validate(kind: ConfigKind, document: dict[str, Any]) -> dict[str, Any]:
    value = parse(kind, document)
    return {"valid": True, "kind": kind, "content_hash": content_hash(serialize(value))}


def diff(kind: ConfigKind, left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    a, b = serialize(parse(kind, left))["payload"], serialize(parse(kind, right))["payload"]
    changes: list[dict[str, Any]] = []
    def walk(path: str, old: Any, new: Any) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(old.keys() | new.keys()):
                walk(f"{path}/{key}", old.get(key), new.get(key))
        elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            for index, (old_item, new_item) in enumerate(zip(old, new)):
                walk(f"{path}/{index}", old_item, new_item)
        elif old != new:
            changes.append({"path": path or "/", "before": old, "after": new})
    walk("", a, b)
    return changes


def migrate(kind: ConfigKind, document: dict[str, Any]) -> dict[str, Any]:
    # There is only one supported transport version. Validate explicitly;
    # never guess how a future or legacy payload should be converted.
    return serialize(parse(kind, document))


def parse_vessel(document: dict[str, Any]) -> ConfigDocument: return parse("vessel", document)
def serialize_vessel(value: ConfigDocument) -> dict[str, Any]: return serialize(value)
def validate_vessel(document: dict[str, Any]) -> dict[str, Any]: return validate("vessel", document)
def parse_environment(document: dict[str, Any]) -> ConfigDocument: return parse("environment", document)
def serialize_environment(value: ConfigDocument) -> dict[str, Any]: return serialize(value)
def validate_environment(document: dict[str, Any]) -> dict[str, Any]: return validate("environment", document)
def parse_scenario(document: dict[str, Any]) -> ConfigDocument: return parse("scenario", document)
def serialize_scenario(value: ConfigDocument) -> dict[str, Any]: return serialize(value)
def validate_scenario(document: dict[str, Any]) -> dict[str, Any]: return validate("scenario", document)
def parse_task(document: dict[str, Any]) -> ConfigDocument: return parse("task", document)
def serialize_task(value: ConfigDocument) -> dict[str, Any]: return serialize(value)
def validate_task(document: dict[str, Any]) -> dict[str, Any]: return validate("task", document)
