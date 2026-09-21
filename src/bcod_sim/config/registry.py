"""Identity-bearing definitions resolved without fallback."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from bcod_sim.config.hashing import content_hash
from bcod_sim.core.errors import DuplicateIdentityError, UnknownReferenceError


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True)
class Definition:
    kind: str
    id: str
    version: str
    payload: Mapping[str, Any]
    source: str
    content_hash: str


class Registry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], Definition] = {}

    def register(self, kind: str, id: str, version: str, payload: Mapping[str, Any], source: str) -> Definition:
        if not all((kind, id, version, source)) or "@" in id:
            raise ValueError("Definition requires kind, id, version, and source; id cannot contain @")
        key = (kind, id, version)
        if key in self._entries:
            raise DuplicateIdentityError(f"Duplicate definition: {kind}:{id}@{version}")
        # Round-trip to detach caller-owned mutable input.
        from json import loads
        from bcod_sim.config.hashing import canonical_json

        detached = loads(canonical_json(dict(payload)))
        definition = Definition(kind, id, version, _freeze(detached), source, content_hash(detached))
        self._entries[key] = definition
        return definition

    def resolve(self, kind: str, reference: str) -> Definition:
        id, marker, version = reference.partition("@")
        if not marker or not id or not version:
            raise UnknownReferenceError(f"Explicit id@version required for {kind}: {reference}")
        try:
            return self._entries[(kind, id, version)]
        except KeyError as exc:
            raise UnknownReferenceError(f"Unknown {kind}: {reference}") from exc
