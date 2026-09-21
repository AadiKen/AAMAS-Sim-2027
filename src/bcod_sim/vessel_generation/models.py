"""The single versioned vessel artifact accepted from every creation path."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ParameterLineage(Strict):
    source_kind: Literal["manual", "geometry-derived", "CFD-derived", "calibration-adjusted", "imported"]
    source_id: str = Field(min_length=1)
    original_source: str | None = None
    original_value: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)


class GeometryReference(Strict):
    format: Literal["procedural", "stl", "obj", "step", "canonical"]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    length_m: float = Field(gt=0)
    beam_m: float = Field(gt=0)
    draft_m: float = Field(gt=0)


class ComponentDefinition(Strict):
    kind: Literal["actuator", "sensor"]
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    payload: dict[str, Any]
    source: str = Field(min_length=1)


class CanonicalVessel(Strict):
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    geometry: GeometryReference
    collision: dict[str, Any]
    mass_kg: float = Field(gt=0)
    cg_frd_m: tuple[float, float, float]
    inertia_cg_kg_m2: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    added_mass_kg: tuple[tuple[float, float, float, float, float, float], ...]
    linear_damping: tuple[float, float, float, float, float, float]
    quadratic_damping: tuple[float, float, float, float, float, float]
    buoyancy_n: float = Field(gt=0)
    center_buoyancy_frd_m: tuple[float, float, float]
    equilibrium_heave_roll_pitch: tuple[float, float, float]
    max_abs_nu: tuple[float, float, float, float, float, float]
    min_substep_s: float = Field(default=1e-6, gt=0)
    max_substep_s: float = Field(gt=0)
    environment_loads: tuple[float, float, float, float] = (0, 0, 0, 0)
    components: tuple[ComponentDefinition, ...] = ()
    provenance: dict[str, ParameterLineage]
    validation_claim: Literal["synthetic/reference validation only"] = "synthetic/reference validation only"
    validation_state: Literal["unvalidated"] = "unvalidated"

    @model_validator(mode="after")
    def physical_checks(self):
        import numpy as np
        inertia = np.asarray(self.inertia_cg_kg_m2); added = np.asarray(self.added_mass_kg)
        if added.shape != (6, 6) or not np.allclose(added, added.T) or np.linalg.eigvalsh(added).min() < 0:
            raise ValueError("added mass must be a positive semidefinite symmetric 6x6 matrix")
        eig = np.linalg.eigvalsh(inertia)
        if eig.min() <= 0 or eig.max() > np.sort(eig)[:2].sum() + 1e-9:
            raise ValueError("inertia must be positive definite and satisfy the triangle inequality")
        if min(self.linear_damping) < 0 or min(self.quadratic_damping) < 0 or min(self.max_abs_nu) <= 0:
            raise ValueError("damping and operating envelope must be physically valid")
        if self.collision.get("kind") not in {"sphere", "box"}:
            raise ValueError("collision geometry must use a supported shape")
        required = {"mass_kg", "cg_frd_m", "inertia_cg_kg_m2", "added_mass_kg",
                    "linear_damping", "quadratic_damping", "buoyancy_n"}
        if not required.issubset(self.provenance):
            raise ValueError("all physical parameter groups require provenance")
        return self

    def simulator_definitions(self) -> list[dict[str, Any]]:
        payload = self.model_dump(include={"mass_kg", "cg_frd_m", "inertia_cg_kg_m2", "added_mass_kg",
            "linear_damping", "quadratic_damping", "buoyancy_n", "center_buoyancy_frd_m", "max_abs_nu",
            "min_substep_s", "max_substep_s", "collision", "environment_loads"}, mode="json")
        payload.update({"geometry": self.geometry.model_dump(mode="json"),
                        "equilibrium_heave_roll_pitch": list(self.equilibrium_heave_roll_pitch),
                        "provenance": {k: v.model_dump(mode="json") for k, v in self.provenance.items()},
                        "validation_claim": self.validation_claim,"validation_state":self.validation_state})
        definitions = [{"kind": "vessel", "id": self.id, "version": self.version,
                        "payload": payload, "source": "canonical-vessel"}]
        definitions.extend(component.model_dump(mode="json") for component in self.components)
        return definitions
