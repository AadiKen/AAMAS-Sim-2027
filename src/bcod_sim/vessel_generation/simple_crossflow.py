"""Geometry-derived lateral sections and bounded cross-flow/lift estimates.

The drag construction follows the sectional approach of Hughes et al. (2011):
local transverse velocity, local projected depth, and a shape-dependent Cd.
The rounded-section curve is the existing Hoerner tabulation used elsewhere
in BCOD. A flat-plate Cd=2 is the bluff-section upper envelope. Blending by
section fullness interpolates the ideal half-ellipse (pi/4) and rectangle (1)
without vessel-specific fitting. The linear lift estimate is the Inoue et al.
(1981) low-aspect-ratio/merchant-ship expression, only within its guarded
displacement-monohull domain. Multihulls use its potential low-AR term only.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from bcod_sim.dynamics.crossflow import HOERNER_RATIO, HOERNER_CD
from .simple_sections import section_properties, polygon_properties

DRAG_SOURCE = "Hughes et al., ISSW 2011, sectional Cd from Hoerner 1965 ellipse/rectangle curves"
LIFT_SOURCE = "Inoue, Hirano & Kijima, International Shipbuilding Progress 28 (1981), low-AR linear derivative"


def _clip_section(loop: np.ndarray, waterline: float) -> np.ndarray:
    points = np.asarray(loop, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-10:
        points = points[:-1]
    out = []
    for p, q in zip(points, np.roll(points, -1, axis=0)):
        inside_p, inside_q = p[2] >= waterline, q[2] >= waterline
        if inside_p:
            out.append(p)
        if inside_p != inside_q:
            out.append(p + (waterline-p[2])/(q[2]-p[2])*(q-p))
    if len(out) < 3:
        return np.empty((0, 3))
    clipped = np.asarray(out)
    keep = np.r_[True, np.linalg.norm(np.diff(clipped, axis=0), axis=1) > 1e-9]
    return clipped[keep]


def section_drag_coefficient(beam: float, depth: float, fullness: float) -> tuple[float, dict]:
    if min(beam, depth) <= 0 or not np.isfinite((beam, depth, fullness)).all():
        raise ValueError("Cross-flow section geometry must be positive and finite")
    ratio = beam/(2*depth)
    rounded = float(np.interp(ratio, HOERNER_RATIO, HOERNER_CD))
    # pi/4 is the submerged area fraction of a half-ellipse in its B*T box.
    shape = float(np.clip((fullness-math.pi/4)/(1-math.pi/4), 0., 1.))
    cd = (1-shape)*rounded + shape*2.0
    return float(np.clip(cd, min(HOERNER_CD), 2.0)), {
        "method": "hoerner_ellipse_to_bluff_fullness_v1", "source": DRAG_SOURCE,
        "applicability": "high-Re separated transverse flow; ideal-section shape interpolation",
        "confidence": "low" if ratio < HOERNER_RATIO[0] or ratio > HOERNER_RATIO[-1] else "medium",
        "beam_over_twice_depth": ratio, "rectangle_weight": shape,
        "outside_tabulated_ratio": not HOERNER_RATIO[0] <= ratio <= HOERNER_RATIO[-1]}


def extract_crossflow_stations(mesh: trimesh.Trimesh, waterline: float,
                               *, count: int = 31, max_froude: float = 0.,
                               density: float = 1025.) -> tuple[list[dict], dict]:
    """Keep every watertight hull component and clipped submerged section."""
    components = mesh.split(only_watertight=True)
    stations = []
    methods = []
    for hull_id, component in enumerate(components):
        length = float(component.extents[0])
        dx = length/count
        local = []
        for x in np.linspace(component.bounds[0, 0]+dx/2, component.bounds[1, 0]-dx/2, count):
            loops = section_properties(component, 0, float(x))["loops"]
            for loop_id, loop in enumerate(loops):
                clipped = _clip_section(loop, waterline)
                if len(clipped) < 3:
                    continue
                area, cy, cz, _, _ = polygon_properties(clipped, 1, 2)
                beam = float(np.ptp(clipped[:, 1]))
                depth = float(np.ptp(clipped[:, 2]))
                if area <= 1e-12 or beam <= 1e-6 or depth <= 1e-6:
                    continue
                fullness = float(np.clip(area/(beam*depth), 0., 1.))
                perimeter = float(np.linalg.norm(np.diff(np.vstack((clipped, clipped[0])), axis=0)[:, 1:], axis=1).sum())
                cd, provenance = section_drag_coefficient(beam, depth, fullness)
                local.append({"x_m": float(x), "y_m": float(cy), "z_m": float(cz),
                    "x_reference_m": 0., "hull_id": hull_id, "loop_id": loop_id,
                    "lateral_force_method": "translation_shear_v5",
                    "beam_m": beam, "draft_m": depth, "lateral_projected_depth_m": depth,
                    "submerged_section_area_m2": float(area), "section_perimeter_m": perimeter,
                    "section_fullness": fullness, "section_aspect_ratio": beam/depth,
                    "section_family": "bluff" if provenance["rectangle_weight"] >= .5 else "rounded",
                    "dx_m": float(dx), "cd": cd, "cd_provenance": provenance,
                    "lift_base_kg_per_m": 0.})
        if not local:
            continue
        total_projected = sum(s["lateral_projected_depth_m"]*s["dx_m"] for s in local)
        max_beam = max(s["beam_m"] for s in local)
        max_draft = max(s["draft_m"] for s in local)
        volume = sum(s["submerged_section_area_m2"]*s["dx_m"] for s in local)
        block = volume/max(length*max_beam*max_draft, 1e-12)
        slenderness = length/max_beam
        k = 2*max_draft/length
        if (len(components) == 1 and 4 <= slenderness <= 12 and
            .4 <= block <= .85 and max_froude <= .4):
            lift_coefficient = math.pi/2*k + 1.4*block*max_beam/length
            method, confidence = "inoue_guarded_displacement_v1", "medium"
        elif len(components) > 1 and slenderness >= 4 and max_froude <= .4:
            lift_coefficient = math.pi/2*k
            method, confidence = "low_aspect_potential_multihull_v1", "low"
        else:
            lift_coefficient = 0.
            method, confidence = "unsupported_lift_crossflow_only", "low"
        for s in local:
            weight = s["lateral_projected_depth_m"]*s["dx_m"]/total_projected
            s["lift_base_kg_per_m"] = .5*density * length*max_draft*lift_coefficient*weight
            s["lift_method"] = method
        methods.append({"hull_id": hull_id, "method": method, "confidence": confidence,
            "source": LIFT_SOURCE if lift_coefficient else "No extrapolation beyond guarded domain",
            "length_m": length, "beam_m": max_beam, "draft_m": max_draft,
            "sectional_block_coefficient": block, "slenderness": slenderness,
            "maximum_froude": max_froude, "lift_coefficient": lift_coefficient,
            "station_count": len(local)})
        stations.extend(local)
    if len(stations) < 3:
        raise ValueError("Hull cannot be sectioned for cross-flow")
    return stations, {"drag": {"method": "hoerner_ellipse_to_bluff_fullness_v1",
        "source": DRAG_SOURCE, "confidence": "low", "applicability": "high-Re separated cross-flow"},
        "transition": {"method": "translation_shear_v5", "weight": "q_mean^2/(mean_abs_u^2+q_mean^2)",
                       "source": "uniform-drift incidence saturation with separate rotational-shear cross-flow; no fitted constant",
                       "confidence": "low"},
        "linear_lift": methods, "hull_components": len(components), "stations": len(stations),
        "reference_point_frd_m": [0., 0., 0.]}
