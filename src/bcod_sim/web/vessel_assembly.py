"""Lossless transport of a core vessel instance and its registry definitions."""

from typing import Any

from bcod_sim.config.models import Vessel
from bcod_sim.config.registry import Registry
from bcod_sim.core.errors import ConfigSchemaError
from bcod_sim.web.runtime_factory import FixedThrusterRuntime, SensorRuntime, VesselRuntime


def normalize(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != {"schema_version", "vessel", "definitions"}:
        raise ConfigSchemaError("Vessel assembly requires schema_version, vessel and definitions")
    if document["schema_version"] != 1 or type(document["schema_version"]) is not int:
        raise ConfigSchemaError("Unsupported vessel assembly version")
    vessel = Vessel.model_validate(document["vessel"])
    definitions = document["definitions"]
    if not isinstance(definitions, list):
        raise ConfigSchemaError("Vessel assembly definitions must be a list")
    registry = Registry()
    for item in definitions:
        if not isinstance(item, dict) or set(item) != {"kind", "id", "version", "payload", "source"}:
            raise ConfigSchemaError("Malformed vessel assembly definition")
        if item["kind"] not in {"vessel", "actuator", "sensor"}:
            raise ConfigSchemaError("Unexpected vessel assembly definition kind")
        registry.register(**item)
        if item["kind"] == "vessel": VesselRuntime.model_validate(item["payload"])
        elif item["kind"] == "actuator": FixedThrusterRuntime.model_validate(item["payload"])
        else: SensorRuntime.model_validate(item["payload"])
    expected = {("vessel", vessel.definition)} | {("actuator", x) for x in vessel.actuators} | {
        ("sensor", x) for x in vessel.sensors}
    actual = {(item["kind"], f"{item['id']}@{item['version']}") for item in definitions}
    if expected != actual:
        raise ConfigSchemaError(f"Vessel assembly reference mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return {"schema_version": 1, "vessel": vessel.model_dump(mode="json"),
            "definitions": definitions}
