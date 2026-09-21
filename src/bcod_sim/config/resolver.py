"""One-way raw config to immutable, identity-resolved experiment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

import yaml
from pydantic import ValidationError

from bcod_sim.config.hashing import canonical_json, content_hash
from bcod_sim.config.models import ExperimentConfig
from bcod_sim.config.registry import Definition, Registry, _freeze
from bcod_sim.core.errors import ConfigSchemaError, ExternalDataUnavailableError


@dataclass(frozen=True)
class ResolvedExperiment:
    config: ExperimentConfig
    definitions: tuple[Definition, ...]
    canonical_payload: Mapping[str, Any]
    content_hash: str


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigSchemaError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


class _UniqueLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    return _no_duplicate_pairs(loader.construct_pairs(node, deep=True))


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            result = json.loads(text, object_pairs_hook=_no_duplicate_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ConfigSchemaError(f"Nonfinite: {x}")))
        elif path.suffix.lower() in (".yaml", ".yml"):
            result = yaml.load(text, Loader=_UniqueLoader)
        else:
            raise ConfigSchemaError(f"Unsupported config format: {path.suffix}")
    except (OSError, yaml.YAMLError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigSchemaError(str(exc)) from exc
    if not isinstance(result, dict):
        raise ConfigSchemaError("Configuration root must be an object")
    return result


def resolve(raw: Mapping[str, Any], registry: Registry) -> ResolvedExperiment:
    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc
    refs: list[tuple[str, str]] = []
    for vessel in config.vessels:
        refs.append(("vessel", vessel.definition))
        if vessel.dynamics:
            refs.append(("dynamics", vessel.dynamics))
        refs.extend(("actuator", ref) for ref in vessel.actuators)
        refs.extend(("sensor", ref) for ref in vessel.sensors)
        if vessel.policy:
            refs.append(("policy", vessel.policy))
    refs.append(("task", config.task.type))
    refs.extend(("asset", ref) for ref in config.world.obstacles)
    if config.scenario_template:
        refs.append(("scenario_template", config.scenario_template))
    if config.world.source.data_product:
        refs.append(("data_product", config.world.source.data_product))
    definitions = tuple(registry.resolve(kind, ref) for kind, ref in refs)
    if config.world.source.kind == "real_world":
        products = [definition for definition in definitions if definition.kind == "data_product"]
        bundle_hash = products[0].payload.get("bundle_hash") if len(products) == 1 else None
        if (not isinstance(bundle_hash, str) or len(bundle_hash) != 64 or
                any(character not in "0123456789abcdef" for character in bundle_hash)):
            raise ExternalDataUnavailableError("Real-world data product requires a resolved bundle_hash")
    payload = {
        "config": config.model_dump(mode="json"),
        "definitions": [
            {"kind": d.kind, "id": d.id, "version": d.version, "content_hash": d.content_hash, "source": d.source}
            for d in definitions
        ],
    }
    detached = json.loads(canonical_json(payload))
    return ResolvedExperiment(config, definitions, _freeze(detached), content_hash(detached))
