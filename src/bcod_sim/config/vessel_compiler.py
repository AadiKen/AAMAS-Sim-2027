"""Fail-closed canonical vessel artifact compilation and fingerprinting."""
from dataclasses import dataclass
from typing import Any,Mapping
from bcod_sim.config.hashing import content_hash
from bcod_sim.core.errors import PhysicalValidationError

REQUIRED=("canonical_origin_frd_m","mass","cg","inertia","added_mass","coriolis_model","damping","hydrostatics","crossflow","equilibrium","actuators","sensors","validity","asset_hashes","provenance")

@dataclass(frozen=True)
class CompiledVessel:
    payload:Mapping[str,Any]
    plant_fingerprint:str

def compile_vessel(source:Mapping[str,Any])->CompiledVessel:
    missing=[key for key in REQUIRED if key not in source]
    if missing: raise PhysicalValidationError(f"Unresolved vessel fields: {missing}")
    if source.get("canonical_frame") not in (None,"FRD") or tuple(source["canonical_origin_frd_m"])!=(0,0,0): raise PhysicalValidationError("Compiled plant must use canonical FRD plant origin")
    authorities=source["hydrostatics"]
    if not isinstance(authorities,dict) or "model" not in authorities: raise PhysicalValidationError("Exactly one explicit hydrostatic authority is required")
    hashes=source["asset_hashes"]
    if any(not isinstance(v,str) or len(v)!=64 for v in hashes.values()): raise PhysicalValidationError("Assets require SHA-256 hashes")
    detached=dict(source); fingerprint=content_hash(detached)
    return CompiledVessel(detached,fingerprint)
