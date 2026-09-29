"""Replace the Surveyor v2 scaled aft cap with surviving STEP section contours.

The source-derived contours are recorded separately as immutable input.  All
coordinates in this B-rep stage are FRD millimetres.  The cut is source Z=5..95
mm; the tiny existing terminal cap before Z=5 is retained.  Source contours
at Z=18.8,30,45,60,75,90 mm are lofted to the unchanged Z=95 mm section.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v2_targeted_patch'
OUT = ROOT / 'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'
SOURCE_CONTOURS = OUT / 'provenance/aft_source_contours_frd_mm.json'
CUT_START, CUT_END = -910., -820.


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def adaptive_mass_properties(shape: cq.Shape) -> dict:
    # CadQuery Shape.Volume() uses OCC's nonadaptive integration and can
    # undercount this spline loft by ~0.26%.  Adaptive 2D Gauss integration
    # converges against two independent conformal STEP tessellations.
    properties = GProp_GProps()
    estimated_error = BRepGProp.VolumeProperties_s(shape.wrapped, properties,
                                                    1e-6, True, False)
    centre = properties.CentreOfMass()
    return {'volume_mm3': properties.Mass(),
            'center_frd_mm': [centre.X(), centre.Y(), centre.Z()],
            'estimated_relative_error': estimated_error}


def source_wire(row: dict) -> cq.Wire:
    """Eight smooth edges aligned with the untouched native section topology."""
    x = row['frd_x_mm']
    # The source contour starts at the outer top; shift one eighth turn to the
    # top midpoint so the two lid edges and six hull edges correspond to the
    # surviving eight-edge native section at the forward join.
    points = np.roll(np.asarray(row['points_frd_yz_mm']), 16, axis=0)
    closed = np.vstack((points, points[0]))
    edges = []
    for index in range(8):
        section = [cq.Vector(x, *yz) for yz in closed[index*16:index*16+17]]
        edges.append(cq.Edge.makeLine(section[0], section[-1]) if index in (0, 7)
                     else cq.Edge.makeSpline(section))
    return cq.Wire.assembleEdges(edges)


def native_wire(solid: cq.Solid, x: float, *, split_lid: bool = False) -> cq.Wire:
    wire = cq.Workplane('YZ', origin=(x, 0, 0)).newObject([solid]).section().wires().val()
    if not split_lid:
        return wire
    # The tiny existing aft section has seven edges.  Split its top lid into
    # two exact trimmed pieces, preserving the original B-spline geometry.
    lid = wire.Edges()[0]
    first, last = lid._geomAdaptor().FirstParameter(), lid._geomAdaptor().LastParameter()
    middle = (first+last)/2
    return cq.Wire.assembleEdges([lid.trim(first, middle), lid.trim(middle, last)] +
                                 wire.Edges()[1:])


def patch_one(solid: cq.Solid, contours: list[dict]) -> tuple[cq.Solid, dict]:
    cut = cq.Solid.makeBox(CUT_END-CUT_START, 2000, 2000,
                           cq.Vector(CUT_START, -1000, -1000))
    outside = solid.cut(cut)
    wires = [native_wire(solid, CUT_START, split_lid=True)]
    for row in contours[::3]:
        wires.append(source_wire(row))
    wires.append(native_wire(solid, CUT_END))
    # Piecewise ruled spans avoid the high-order tip overshoot found in the
    # first rejected trial.  Exact native boundary wires avoid the planar lip
    # found in the second rejected trial.
    loft = cq.Solid.makeLoft(wires, ruled=True)
    if loft.Volume() <= 0 or not BRepCheck_Analyzer(loft.wrapped).IsValid():
        raise RuntimeError('Aft replacement loft is invalid')
    result = outside.fuse(loft).clean()
    if len(result.Solids()) != 1:
        raise RuntimeError(f'Aft replacement did not form one pontoon: {len(result.Solids())}')
    repaired = result.Solids()[0]
    if not repaired.isValid() or not BRepCheck_Analyzer(repaired.wrapped).IsValid():
        raise RuntimeError('Aft repaired B-rep is invalid')
    seam_faces = [face for face in repaired.Faces() if any(
        abs(face.BoundingBox().xmin-x)<1e-5 and
        abs(face.BoundingBox().xmax-x)<1e-5 and face.Area()>1e-6
        for x in (CUT_START, CUT_END))]
    if seam_faces:
        raise RuntimeError(f'Aft repaired B-rep retains {len(seam_faces)} planar seam faces')
    final_outside = repaired.cut(cut)
    audit = {'before_volume_mm3': solid.Volume(), 'after_volume_mm3': repaired.Volume(),
             'added_outside_mask_mm3': max(0., final_outside.cut(outside).Volume()),
             'removed_outside_mask_mm3': max(0., outside.cut(final_outside).Volume()),
             'loft_volume_mm3': loft.Volume(), 'brepcheck_valid': True,
             'planar_seam_face_count': 0,
             'old_mask_face_count': len(solid.intersect(cut).Faces()),
             'new_mask_face_count': len(repaired.intersect(cut).Faces()),
             'old_mask_vertex_count': len(solid.intersect(cut).Vertices()),
             'new_mask_vertex_count': len(repaired.intersect(cut).Vertices())}
    return repaired, audit


def run() -> dict:
    contours = json.loads(SOURCE_CONTOURS.read_text())
    (OUT / 'cad').mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'surveyor-v3-aft-cap-repair-1',
                'input_status': 'pre_fallback_targeted_repair',
                'source_contours_sha256': digest(SOURCE_CONTOURS),
                'mask_frd_x_mm': [CUT_START, CUT_END],
                'mask_source_z_mm': [CUT_START+915, CUT_END+915],
                'contour_source_z_mm': [row['source_z_mm'] for row in contours[::3]],
                'variants': {}}
    for name in ('nominal', 'fuller', 'finer'):
        prior = BASE / 'cad' / f'surveyor_{name}_v2_bridge_refit.step'
        root = cq.importers.importStep(str(prior)).val()
        port = max(root.Solids(), key=lambda s: s.Center().y)
        repaired, audit = patch_one(port, contours)
        opposite = repaired.mirror('XZ')
        if not BRepCheck_Analyzer(opposite.wrapped).IsValid():
            raise RuntimeError('Mirrored starboard B-rep is invalid')
        output = OUT / 'cad' / f'surveyor_{name}_v3_aft_repair.step'
        cq.exporters.export(cq.Compound.makeCompound([repaired, opposite]), str(output))
        exported = cq.importers.importStep(str(output)).val().Solids()
        if len(exported) != 2:
            raise RuntimeError('Exported STEP does not contain two pontoon solids')
        exported_properties = [adaptive_mass_properties(solid) for solid in exported]
        exported_volumes = [item['volume_mm3'] for item in exported_properties]
        total_volume = sum(exported_volumes)
        combined_center = [sum(item['volume_mm3']*item['center_frd_mm'][axis]
                               for item in exported_properties)/total_volume
                           for axis in range(3)]
        manifest['variants'][name] = {'input_step': str(prior),
                                      'input_step_sha256': digest(prior),
                                      'output_step': str(output),
                                      'output_step_sha256': digest(output),
                                      'port': audit,
                                      'exported_step_adaptive_properties': exported_properties,
                                      'combined_step_volume_m3': total_volume/1e9,
                                      'combined_step_center_frd_m': [x/1000 for x in combined_center]}
        print(name, 'STEP', output, 'volume', total_volume/1e9, flush=True)
    (OUT / 'cad_repair_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    run()
