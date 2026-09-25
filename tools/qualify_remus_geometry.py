#!/usr/bin/env python3
"""Fail-closed qualification of the public NPS SAVAGE REMUS visual model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MSS = {"length_m": 1.6, "diameter_m": 0.19, "mass_kg": 31.9}
SCALE = 0.2145


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_obj_sections(path):
    vertices, sections, current = [], [], None
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(value) for value in line.split()[1:4]))
        elif line.startswith("usemtl "):
            current = {"material": line.split()[1], "faces": []}; sections.append(current)
        elif line.startswith("f ") and current is not None:
            polygon = tuple(int(token.split("/")[0]) - 1 for token in line.split()[1:])
            current["faces"].extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1))
    return np.asarray(vertices), sections


def transformed_candidate(vertices, sections, include_fins):
    selected = list(range(8)) + [12] + (list(range(8, 12)) if include_fins else [])
    points, triangles = [], []
    for section_index in selected:
        section = sections[section_index]
        mapping = {}
        for face in section["faces"]:
            converted = []
            for old in face:
                if old not in mapping:
                    point = vertices[old].copy()
                    # Assimp 6 incorrectly drops the enclosing transform for the X3D TailSection.
                    if section_index == 12:
                        point = np.asarray([1.0 - SCALE * point[1], SCALE * point[0], SCALE * point[2]])
                    mapping[old] = len(points); points.append(point)
                converted.append(mapping[old])
            triangles.append(tuple(converted))
    return np.asarray(points), triangles


def audit(points, triangles):
    welded, canonical_points, faces = {}, [], []
    for triangle in triangles:
        converted = []
        for index in triangle:
            key = tuple(round(float(value), 7) for value in points[index])
            if key not in welded:
                welded[key] = len(canonical_points); canonical_points.append(np.asarray(key))
            converted.append(welded[key])
        faces.append(tuple(converted))
    duplicates = sum(count - 1 for count in Counter(tuple(sorted(face)) for face in faces).values())
    degenerates = sum(len(set(face)) < 3 for face in faces)
    edge_counts = Counter(tuple(sorted(edge)) for a, b, c in faces for edge in ((a, b), (b, c), (c, a)))
    directed = Counter(edge for a, b, c in faces for edge in ((a, b), (b, c), (c, a)))
    inconsistent = sum(count == 2 and not (directed[(a, b)] == 1 and directed[(b, a)] == 1) for (a, b), count in edge_counts.items())
    adjacency = defaultdict(set)
    for a, b, c in faces:
        adjacency[a].update((b, c)); adjacency[b].update((a, c)); adjacency[c].update((a, b))
    unseen, components = set(adjacency), 0
    while unseen:
        components += 1; stack = [unseen.pop()]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen: unseen.remove(neighbor); stack.append(neighbor)
    area = signed_volume = 0.0
    for a, b, c in faces:
        pa, pb, pc = canonical_points[a], canonical_points[b], canonical_points[c]
        area += np.linalg.norm(np.cross(pb - pa, pc - pa)) / 2
        signed_volume += float(np.dot(pa, np.cross(pb, pc))) / 6
    array = np.asarray(canonical_points)
    return {"triangles": len(faces), "vertices_after_weld": len(array), "boundary_edges": sum(count == 1 for count in edge_counts.values()),
            "nonmanifold_edges": sum(count > 2 for count in edge_counts.values()), "duplicate_faces": duplicates,
            "degenerate_faces": degenerates, "normal_inconsistent_shared_edges": inconsistent, "connected_components": components,
            "watertight": bool(edge_counts) and all(count == 2 for count in edge_counts.values()), "surface_area_m2": float(area),
            "enclosed_volume_m3": abs(signed_volume) if edge_counts and all(count == 2 for count in edge_counts.values()) else None,
            "signed_volume_m3_unreliable_when_open": signed_volume,
            "bounds_min_m": array.min(axis=0).tolist(), "bounds_max_m": array.max(axis=0).tolist(),
            "extents_m": np.ptp(array, axis=0).tolist(), "self_intersections": "NOT_RUN_DIMENSION_AND_WATERTIGHTNESS_GATE"}


def symmetry(points):
    unique = np.unique(np.round(points, 7), axis=0)
    def hausdorff(axis):
        reflected = unique.copy(); reflected[:, axis] *= -1
        return float(max(np.min(np.linalg.norm(unique - point, axis=1)) for point in reflected))
    return {"body_centerline_axes": "world Y/Z; world X is longitudinal", "max_reflection_mismatch_y_m": hausdorff(1),
            "max_reflection_mismatch_z_m": hausdorff(2)}


def write_stl(path, points, triangles, label):
    with path.open("w") as stream:
        stream.write(f"solid {label}\n")
        for a, b, c in triangles:
            pa, pb, pc = points[a], points[b], points[c]
            normal = np.cross(pb - pa, pc - pa); length = np.linalg.norm(normal)
            if length: normal /= length
            stream.write(f" facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n  outer loop\n")
            for point in (pa, pb, pc): stream.write(f"   vertex {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
            stream.write("  endloop\n endfacet\n")
        stream.write(f"endsolid {label}\n")


def idealized_profile_volume():
    axial = [1.8525, 1.9525, 2.0525, 2.1525, 2.2525, 6.5065, 6.7715, 7.0275, 7.2675, 7.4845, 7.6715, 7.8235, 7.9375]
    radius = [.267*.48, .267*.76, .267*.88, .267*.95, .267, .267, .267*.985, .267*.94, .267*.866, .267*.766, .267*.643, .267*.5, .267*.342]
    volume = sum(math.pi*(b-a)*(ra*ra+ra*rb+rb*rb)/3 for a,b,ra,rb in zip(axial,axial[1:],radius,radius[1:])) * SCALE**3
    return volume


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("source", type=Path); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    output = args.output; geometry = output/"geometry"; geometry.mkdir(parents=True, exist_ok=True)
    preserved = geometry/"original"/args.source.name; preserved.parent.mkdir(exist_ok=True); shutil.copy2(args.source, preserved)
    obj = geometry/"assimp_scene.obj"; subprocess.run(["assimp", "export", str(args.source), str(obj), "-f", "objnomtl"], cwd=ROOT, check=True, capture_output=True)
    vertices, sections = parse_obj_sections(obj)
    body = transformed_candidate(vertices, sections, False); body_fins = transformed_candidate(vertices, sections, True)
    write_stl(geometry/"remus_body_unqualified.stl", *body, "remus_body_unqualified")
    write_stl(geometry/"remus_body_fins_unqualified.stl", *body_fins, "remus_body_fins_unqualified")
    root = ET.parse(args.source).getroot(); metadata = {element.attrib.get("name"): element.attrib.get("content") for element in root.findall("./head/meta")}
    defs = [{"tag": element.tag, "name": element.attrib["DEF"]} for element in root.findall(".//*[@DEF]")]
    native_length = (7.9375 - 1.8525) * SCALE; native_diameter = .534 * SCALE
    volume = idealized_profile_volume()
    report = {"qualification_status": "FAIL_GEOMETRY", "classification": "PUBLIC_RESEARCH_VISUAL_MODEL", "authoritative_cad": False,
      "provenance": {"source_url": "https://savage.nps.edu/Savage/Robots/UnmannedUnderwaterVehicles/Remus.x3d", "preserved_original": str(preserved), "sha256": file_hash(args.source), "file_size_bytes": args.source.stat().st_size, "metadata": metadata,
                     "native_units": "X3D metres after explicit uniform scale=0.2145; comment claims English-to-metric conversion", "coordinate_convention": "authored longitudinal +Y, transformed longitudinal -X; Y/Z radial"},
      "scene_inventory": {"shape_nodes": len(root.findall(".//Shape")), "named_nodes": defs, "assimp_mesh_instances": len(sections), "categories": {"main_body":"7 open cylinders", "nose":"capped extrusion ending at nonzero radius", "tail":"extrusion with endCap=false ending at nonzero radius", "control_fins":4, "propulsor":"3 stator blades and crankshaft inside LOD", "accessories":"two side-scan arrays, top hook, GPS transceiver"}},
      "dimensions": {"native_body_length_m": native_length, "native_body_diameter_m": native_diameter, "body_fineness_ratio": native_length/native_diameter,
                     "mss": MSS, "length_difference_percent": 100*(native_length/MSS["length_m"]-1), "diameter_difference_percent": 100*(native_diameter/MSS["diameter_m"]-1), "rescaling_performed": False},
      "topology": {"BODY_ONLY": audit(*body), "BODY_PLUS_FINS": audit(*body_fins)}, "symmetry": {"BODY_ONLY": symmetry(body[0]), "BODY_PLUS_FINS": symmetry(body_fins[0])},
      "volume_check": {"enclosed_mesh_volume_m3": None, "reason":"candidate is not watertight", "idealized_authored_profile_if_end-capped_m3_diagnostic_only": volume,
                       "equivalent_seawater_mass_kg_at_1025": volume*1025, "difference_from_mss_mass_percent": 100*(volume*1025/MSS["mass_kg"]-1)},
      "repairs": [], "openfoam_surfaceCheck": "NOT_RUN_FAIL_GEOMETRY; native executable unavailable and no cleaned CFD surface was approved",
      "shape_plausibility": {"body_axisymmetry":"strong at the authored low polygon resolution", "nose":"smooth taper but terminates at nonzero radius", "tail":"smooth taper but explicitly open and terminates at nonzero radius", "fins":"four approximately cruciform box fins; placement qualitatively plausible but coarse", "propeller":"LOD-only three-blade visual assembly, excluded from passive candidates", "public_reference":"WHOI describes REMUS 100 as 19 cm diameter and at least 160 cm long"},
      "major_limitations": ["Native length and diameter materially disagree with pinned MSS remus100 physical dimensions.", "The passive body is an assembly of open primitives, not a unified watertight pressure hull.", "The authored nose and tail terminate at nonzero radii; closing them would invent end geometry.", "The explicit scale=0.2145 conflicts with the nearby comment describing 39.3 inches per metre.", "The visual mesh is very low resolution and fins are intersecting box primitives."],
      "recommended_cfd_geometry": "NONE"}
    (output/"qualification.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"provenance.json").write_text(json.dumps(report["provenance"], indent=2)+"\n")
    (output/"scene_inventory.json").write_text(json.dumps(report["scene_inventory"], indent=2)+"\n")
    (output/"topology.json").write_text(json.dumps(report["topology"], indent=2)+"\n")
    body_audit=report["topology"]["BODY_ONLY"]; fins_audit=report["topology"]["BODY_PLUS_FINS"]
    lines=["# NPS SAVAGE REMUS geometry qualification","","**FAIL_GEOMETRY**","","## Provenance","",f"- Source: {report['provenance']['source_url']}",f"- SHA256: `{report['provenance']['sha256']}`",f"- Size: {report['provenance']['file_size_bytes']} bytes",f"- Classification: PUBLIC_RESEARCH_VISUAL_MODEL; AUTHORITATIVE_CAD=false","","## Dimensional gate","",f"- Native body: {native_length:.6f} m long × {native_diameter:.6f} m diameter",f"- Pinned MSS remus100: 1.600 m long × 0.190 m diameter; mass 31.9 kg",f"- Differences: {report['dimensions']['length_difference_percent']:.1f}% length, {report['dimensions']['diameter_difference_percent']:.1f}% diameter",f"- No rescaling was performed.","","## Topology","",f"- BODY_ONLY: {body_audit['triangles']} triangles, {body_audit['boundary_edges']} boundary edges, {body_audit['nonmanifold_edges']} non-manifold edges, {body_audit['connected_components']} components, watertight={body_audit['watertight']}",f"- BODY_PLUS_FINS: {fins_audit['triangles']} triangles, {fins_audit['boundary_edges']} boundary edges, {fins_audit['nonmanifold_edges']} non-manifold edges, {fins_audit['connected_components']} components, watertight={fins_audit['watertight']}","- Self-intersection testing was not promoted past the dimensional/watertightness gate.","","## Volume gate","",f"The open mesh has no valid enclosed volume. Even an idealized diagnostic that caps the authored truncated profile gives {volume:.6f} m³, or {volume*1025:.2f} kg of seawater—{abs(report['volume_check']['difference_from_mss_mass_percent']):.1f}% below the MSS mass.","","## Decision","","The body scale is materially wrong for the pinned REMUS 100, and producing a closed pressure hull would require inventing nose/tail closures. No repair was performed, native OpenFOAM `surfaceCheck` was not run, and neither candidate is recommended for CFD.",""]
    (output/"report.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
