"""Prepare a local-layer KCS +6 degree mesh from the qualified smoke inputs."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

root = Path("stage3_results/kcs-validation/track_b_hmri")
source = root / "mesh_smoke"
target = root / "mesh_wall_layers_v2"
if target.exists():
    raise SystemExit(f"preserving existing mesh attempt: {target}")
for name in ("system/blockMeshDict", "system/controlDict", "constant/triSurface/hull.stl"):
    path = target / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / name, path)
snappy = (source / "system/snappyHexMeshDict").read_text()
assert "addLayers false" in snappy
assert "addLayersControls {relativeSizes true; layers {}; }" not in snappy
snappy = snappy.replace("addLayers false", "addLayers true")
snappy = snappy.replace("refinementSurfaces {hull {level (3 3);", "refinementSurfaces {hull {level (4 4);")
snappy = snappy.replace(
    "addLayersControls {relativeSizes true; layers {};}",
    """addLayersControls {
    relativeSizes false;
    layers {hull {nSurfaceLayers 6;}}
    expansionRatio 1.2;
    firstLayerThickness 0.002;
    minThickness 0.0005;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}""",
)
snappy = snappy.replace(
    'meshQualityControls {#includeEtc "caseDicts/mesh/generation/meshQualityDict"}',
    'meshQualityControls {#includeEtc "caseDicts/mesh/generation/meshQualityDict" relaxed {maxNonOrtho 65;}}',
)
(target / "system/snappyHexMeshDict").write_text(snappy)
speed, length, reynolds = 1.953, 5.75, 9.851e6
nu = speed * length / reynolds
cf = 0.075 / (math.log10(reynolds) - 2) ** 2
u_tau = speed * math.sqrt(cf / 2)
first = 0.002
(target / "wall_plan.json").write_text(json.dumps({
    "model": "kOmegaSST",
    "hull_wall_functions": {"k": "kqRWallFunction", "omega": "omegaWallFunction", "nut": "nutkWallFunction"},
    "target_y_plus": [50, 100],
    "nu_m2_s": nu, "Re": reynolds,
    "estimated_Cf_ITTC1957": cf, "estimated_u_tau_mps": u_tau,
    "estimated_first_cell_center_m_for_y_plus_50_to_100": [50 * nu / u_tau, 100 * nu / u_tau],
    "planned_first_layer_thickness_m": first,
    "estimated_first_cell_y_plus": u_tau * first / (2 * nu),
    "planned_layer_count": 6, "growth_ratio": 1.2,
    "planned_total_layer_thickness_m": first * sum(1.2 ** i for i in range(6)),
    "far_field_grid_changed": False,
}, indent=2) + "\n")
print(target)
