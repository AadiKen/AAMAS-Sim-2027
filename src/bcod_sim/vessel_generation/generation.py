"""Manual, procedural, geometry, imported, and CFD canonical producers."""

import hashlib
import json
import math
import random
from typing import Any

from .models import CanonicalVessel, GeometryReference, ParameterLineage


def _hash(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


class VesselFactory:
    @staticmethod
    def manual(document: dict[str, Any]) -> CanonicalVessel:
        return CanonicalVessel.model_validate(document)

    @staticmethod
    def imported(data: bytes) -> CanonicalVessel:
        vessel = CanonicalVessel.model_validate_json(data)
        return vessel

    @staticmethod
    def procedural(identity: str, *, seed: int, length_m: float, beam_m: float, mass_kg: float,
                   actuator_strength_n: float = 200) -> CanonicalVessel:
        if not (1 <= length_m <= 100 and .2 <= beam_m <= length_m / 2 and mass_kg >= 10 * length_m):
            raise ValueError("procedural dimensions or mass are physically implausible")
        rng = random.Random(seed); draft = beam_m * (.3 + .1 * rng.random())
        inertia = (mass_kg*(beam_m**2+draft**2)/12, mass_kg*(length_m**2+draft**2)/12,
                   mass_kg*(length_m**2+beam_m**2)/12)
        added = tuple(tuple((mass_kg*(.04 + .02*i) if i == j else 0) for j in range(6)) for i in range(6))
        geom_data = json.dumps([seed, length_m, beam_m, draft], separators=(",", ":")).encode()
        source = ParameterLineage(source_kind="geometry-derived", source_id=f"procedural:{seed}", uncertainty=.15)
        components = ({"kind":"actuator", "id":"port", "version":"1", "source":"procedural",
            "payload":{"kind":"fixed_thruster", "mount_frd_m":[-length_m*.35,-beam_m*.35,0],
            "mount_q_to_frd":[1,0,0,0], "thrust_bounds_n":[-actuator_strength_n,actuator_strength_n]}},
            {"kind":"actuator", "id":"starboard", "version":"1", "source":"procedural",
            "payload":{"kind":"fixed_thruster", "mount_frd_m":[-length_m*.35,beam_m*.35,0],
            "mount_q_to_frd":[1,0,0,0], "thrust_bounds_n":[-actuator_strength_n,actuator_strength_n]}})
        return CanonicalVessel(id=identity, version="1", geometry=GeometryReference(format="procedural",
            content_hash=_hash(geom_data), length_m=length_m, beam_m=beam_m, draft_m=draft),
            collision={"kind":"box", "half_extents_m":[length_m/2,beam_m/2,draft/2]}, mass_kg=mass_kg,
            cg_frd_m=(0,0,0), inertia_cg_kg_m2=((inertia[0],0,0),(0,inertia[1],0),(0,0,inertia[2])),
            added_mass_kg=added, linear_damping=tuple(mass_kg*x for x in (.08,.12,.15,.04,.05,.1)),
            quadratic_damping=tuple(mass_kg*x for x in (.02,.04,.05,.01,.01,.03)), buoyancy_n=mass_kg*9.80665,
            center_buoyancy_frd_m=(0,0,-draft*.1), equilibrium_heave_roll_pitch=(0,0,0),
            max_abs_nu=(15,10,8,3,3,3), max_substep_s=.05, components=components,
            provenance={name:source for name in ("mass_kg","cg_frd_m","inertia_cg_kg_m2","added_mass_kg",
                "linear_damping","quadratic_damping","buoyancy_n")})

    @staticmethod
    def geometry(identity: str, geometry: bytes, *, format: str, length_m: float, beam_m: float,
                 draft_m: float, seed: int = 0) -> CanonicalVessel:
        if len(geometry) < 16 or format not in {"stl", "obj", "step"}:
            raise ValueError("geometry is malformed or unsupported")
        vessel = VesselFactory.procedural(identity, seed=seed, length_m=length_m, beam_m=beam_m,
            mass_kg=max(10*length_m, 350*length_m*beam_m*draft_m))
        return vessel.model_copy(update={"geometry": GeometryReference(format=format, content_hash=_hash(geometry),
            length_m=length_m, beam_m=beam_m, draft_m=draft_m)})

    @staticmethod
    def from_cfd(base: CanonicalVessel, added_mass: tuple, linear: tuple, quadratic: tuple,
                 *, run_id: str, uncertainty: float) -> CanonicalVessel:
        lineage = ParameterLineage(source_kind="CFD-derived", source_id=run_id, uncertainty=uncertainty)
        provenance = dict(base.provenance)
        for key in ("added_mass_kg", "linear_damping", "quadratic_damping"): provenance[key] = lineage
        return base.model_copy(update={"added_mass_kg": added_mass, "linear_damping": linear,
            "quadratic_damping": quadratic, "provenance": provenance})
