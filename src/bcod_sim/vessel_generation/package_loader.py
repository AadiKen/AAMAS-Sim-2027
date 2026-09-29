"""Strict, path-independent loading for generated vessel packages."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from bcod_sim.web.runtime_factory import VesselRuntime


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class VesselPackage:
    root: Path
    id: str
    version: str
    runtime_payload: Mapping[str, Any]
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "VesselPackage":
        root = Path(path).expanduser().resolve()
        try:
            manifest_path = root / "coefficient_package.yaml"
            from .coefficient_package import load_coefficient_package
            manifest = load_coefficient_package(manifest_path)
            if not isinstance(manifest, dict) or manifest.get("schema_version") != "manta-hydrodynamics-v1":
                raise ValueError("Unsupported or missing coefficient package manifest")
            runtime_path = root / "runtime_payload.json"
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            VesselRuntime.model_validate(payload)
            if payload != manifest.get("runtime_payload"):
                raise ValueError("runtime_payload.json disagrees with validated package manifest")
            # The package validator is the canonical validator for the generation contract.
            hashes = manifest.get("artifact_hashes", {})
            for relative, expected in hashes.items():
                artifact = (root / relative).resolve()
                if root not in artifact.parents:
                    raise ValueError(f"Package artifact escapes root: {relative}")
                if not artifact.is_file():
                    raise ValueError(f"Missing package artifact: {relative}")
                actual = sha256(artifact.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Package artifact hash mismatch: {relative}")
            # Resolve asset-valued paths against the package location before freezing.
            resolved = json.loads(json.dumps(payload))
            for parent, key in ((resolved.get("hydrostatics") or {}, "hull_mesh"),):
                if parent.get(key):
                    parent[key] = str((root / parent[key]).resolve())
            try:
                vessel_doc = yaml.safe_load((root / "vessel.yaml").read_text(encoding="utf-8"))
            except OSError as exc:
                raise ValueError("Missing vessel.yaml") from exc
            identity = vessel_doc if isinstance(vessel_doc, dict) else {}
            return cls(root, str(identity.get("id", root.name)), str(identity.get("version", "1")),
                       _freeze(resolved), _freeze(manifest))
        except Exception as exc:
            raise ValueError(f"Invalid vessel package at {root}: {exc}") from exc

    def definition(self) -> dict[str, Any]:
        # Registry performs a canonical detach/freeze; thaw tuples and mappings here.
        def thaw(value):
            if isinstance(value, Mapping): return {key: thaw(item) for key, item in value.items()}
            if isinstance(value, tuple): return [thaw(item) for item in value]
            return value
        return {"kind": "vessel", "id": self.id, "version": self.version,
                "payload": thaw(self.runtime_payload), "source": str(self.root)}


def load_vessel_package(path: str | Path) -> dict[str, Any]:
    return VesselPackage.load(path).definition()
