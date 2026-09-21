"""Group compatible component schemas without a global padded tensor."""

from dataclasses import dataclass
from typing import Mapping

from bcod_sim.core.engine import EpisodeEngine


@dataclass(frozen=True, order=True)
class ComponentIdentity:
    env_id: int
    owner_vessel_id: int
    component_id: str


@dataclass(frozen=True, order=True)
class SchemaKey:
    category: str
    implementation: str
    output_shape: tuple[int, ...]
    version: str


@dataclass(frozen=True)
class ComponentGroup:
    schema: SchemaKey
    identities: tuple[ComponentIdentity, ...]
    components: tuple[object, ...]


def _schema(category: str, component: object) -> SchemaKey:
    shape: tuple[int, ...] = ()
    if hasattr(component, "ray_count"):
        shape = (int(component.ray_count),)
    elif hasattr(component, "beam_count"):
        shape = (int(component.beam_count),)
    version = component.config.version if category == "sensor" else "1"
    return SchemaKey(category, f"{type(component).__module__}.{type(component).__qualname__}", shape, version)


def group_components(engines: Mapping[int, EpisodeEngine]) -> tuple[ComponentGroup, ...]:
    grouped: dict[SchemaKey, list[tuple[ComponentIdentity, object]]] = {}
    for env_id, engine in sorted(engines.items()):
        for category, components in (("actuator", engine.all_actuators), ("sensor", engine.all_sensors)):
            for component in components:
                config = component.config
                identity = ComponentIdentity(env_id, config.owner_vessel_id, config.instance_id)
                grouped.setdefault(_schema(category, component), []).append((identity, component))
    result = []
    for schema in sorted(grouped):
        rows = sorted(grouped[schema], key=lambda item: item[0])
        identities = tuple(row[0] for row in rows)
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate component identity within compatible group")
        result.append(ComponentGroup(schema, identities, tuple(row[1] for row in rows)))
    return tuple(result)
