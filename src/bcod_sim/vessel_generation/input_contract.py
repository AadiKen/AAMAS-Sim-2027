"""Configuration-driven adapter for the established passive generator."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from .simple_geometry import prepare_geometry
from .simple_pipeline import generate_simple_vessel
from .simple_sections import hydrostatic_state


class GenerationFailure(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _classify_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "invalid_scale" in message or "scale" in message or "unit" in message:
        return "INVALID_SCALE"
    if "waterline" in message or "displacement" in message or "equilibrium" in message:
        return "NO_EQUILIBRIUM_WATERLINE"
    if "mass" in message or "draft" in message or "loading" in message:
        return "INVALID_LOADING"
    if "section" in message:
        return "SECTION_EXTRACTION_FAILED"
    if "frame" in message or "axis" in message:
        return "FRAME_AMBIGUITY"
    if "mesh" in message or "geometry" in message or "surface" in message:
        return "INVALID_GEOMETRY"
    return "COEFFICIENT_VALIDATION_FAILED"


def generate_from_config(config_path: str | Path, output: str | Path) -> Path:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not all(k in config for k in ("geometry", "loading", "operating_envelope")):
        raise ValueError("config requires geometry, loading and operating_envelope")
    geometry, loading, envelope = (config[k] for k in ("geometry", "loading", "operating_envelope"))
    settings = config.get("generation", {})
    source = Path(geometry["file"])
    if not source.is_absolute():
        source = path.parent / source
    units = geometry["units"]
    source_frame = geometry.get("source_frame", "FRD")
    geometry_mode = geometry.get("mode", "full_hull")
    reference_draft = geometry.get("reference_draft")
    density = float(settings.get("fluid_density_kg_m3", 1025.))
    viscosity = float(settings.get("fluid_kinematic_viscosity_m2_s", 1.05e-6))
    gravity = float(settings.get("gravity_mps2", 9.80665))
    if abs(gravity - 9.80665) > 1e-9:
        raise ValueError("current Plant6 generator requires gravity 9.80665 m/s²")
    mass = loading.get("mass_kg")
    draft = loading.get("draft_m")
    if mass is None and draft is None:
        raise ValueError("loading requires mass_kg or draft_m")
    if mass is None:
        try:
            prepared = prepare_geometry(source, units=units, source_frame=source_frame)
            if geometry_mode == "design_waterline_submerged_hull":
                if reference_draft is None:
                    raise ValueError("design_waterline_submerged_hull requires geometry.reference_draft")
                state = hydrostatic_state(prepared.mesh, float(prepared.mesh.bounds[0, 2]),
                    density=density, waterline_is_boundary=True, include_wetted=False)
            else:
                waterline = float(prepared.mesh.bounds[1, 2] - float(draft))
                state = hydrostatic_state(prepared.mesh, waterline, density=density)
        except ValueError as exc:
            raise GenerationFailure(_classify_failure(exc), str(exc)) from exc
        mass = state["displacement_kg"]
        if mass <= 0:
            raise ValueError("draft produces zero displacement")
        mass_source = "geometry_derived"
    else:
        mass = float(mass)
        mass_source = "supplied"
    cg = loading.get("cg_frd_m")
    cg_source = "supplied" if cg is not None else "estimated"
    if cg is None:
        cg = (0., 0., 0.)
    inertia = loading.get("inertia_cg_kg_m2")
    if inertia is not None:
        inertia = np.asarray(inertia, dtype=float)
        if inertia.shape != (3, 3):
            raise ValueError("inertia_cg_kg_m2 must be 3x3")
    try:
        root = generate_simple_vessel(
            geometry=source, output=output, mass_kg=mass, cg_frd_m=tuple(cg),
            units=units, source_frame=source_frame, draft_m=draft,
            water_density_kg_m3=density, inertia_cg_kg_m2=inertia,
            water_kinematic_viscosity_m2_s=viscosity,
            speed_range_mps=(0., float(envelope["max_forward_speed_mps"])),
            geometry_mode=geometry_mode,
            reference_draft_m=None if reference_draft is None else float(reference_draft),
            reference_length_m=None if geometry.get("reference_length") is None else float(geometry["reference_length"]),
            disable_bem=bool(settings.get("disable_bem", False)),
            lut_samples=int(settings.get("lut_samples", 9)))
    except ValueError as exc:
        raise GenerationFailure(_classify_failure(exc), str(exc)) from exc
    normalized = {"schema": "manta-generation-input-v1", "original_config_sha256": sha256(path.read_bytes()).hexdigest(),
                  "geometry_file": str(source), "geometry_sha256": sha256(source.read_bytes()).hexdigest(),
                  "mass_kg": mass, "draft_m": draft, "cg_frd_m": list(cg),
                  "geometry_mode": geometry_mode, "reference_draft_m": reference_draft,
                  "reference_length_m": geometry.get("reference_length"),
                  "sources": {"mass": mass_source, "draft": "supplied" if draft is not None else "geometry_derived",
                              "cg": cg_source, "inertia": "supplied" if inertia is not None else "estimated"},
                  "operating_envelope": envelope, "generation": settings,
                  "fluid_kinematic_viscosity_m2_s": viscosity}
    (root / "input_definition.yaml").write_text(yaml.safe_dump(normalized, sort_keys=True))
    from .coefficient_package import write_coefficient_package
    write_coefficient_package(root)
    return root
