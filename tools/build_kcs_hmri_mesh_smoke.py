"""Create a deliberately coarse, reproducible KCS Track B mesh smoke case."""
from pathlib import Path
import numpy as np
import trimesh

root = Path("stage3_results/kcs-validation/track_b_hmri")
case = root / "mesh_smoke"
(case / "system").mkdir(parents=True, exist_ok=True)
(case / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
hull = trimesh.load(root / "kcs_hmri_full_hull.stl", process=True)
hull.vertices = hull.vertices * np.array([1.0, -1.0, -1.0])
hull.fix_normals()
hull.export(case / "constant" / "triSurface" / "hull.stl")
header = 'FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n'
(case / "system" / "blockMeshDict").write_text(header % "blockMeshDict" + '''
convertToMeters 1;
vertices ((-15 -8 -5)(20 -8 -5)(20 8 -5)(-15 8 -5)
          (-15 -8 0)(20 -8 0)(20 8 0)(-15 8 0)
          (-15 -8 4)(20 -8 4)(20 8 4)(-15 8 4));
blocks (
 hex (0 1 2 3 4 5 6 7) (70 32 11) simpleGrading (1 1 1)
 hex (4 5 6 7 8 9 10 11) (70 32 9) simpleGrading (1 1 1)
);
edges ();
boundary (
 inletWater {type patch; faces ((1 2 6 5));}
 inletAir {type patch; faces ((5 6 10 9));}
 outletWater {type patch; faces ((0 4 7 3));}
 outletAir {type patch; faces ((4 8 11 7));}
 sides {type wall; faces ((0 1 5 4)(3 7 6 2)(4 5 9 8)(7 11 10 6));}
 bottom {type wall; faces ((0 3 2 1));}
 atmosphere {type patch; faces ((8 9 10 11));}
);
''')
(case / "system" / "snappyHexMeshDict").write_text(header % "snappyHexMeshDict" + '''
castellatedMesh true; snap true; addLayers false;
geometry {
 hull.stl {type triSurfaceMesh; name hull;}
 bowBox {type searchableBox; min (-3.15 -0.5 -0.15); max (-2.75 0.5 0.15);}
}
castellatedMeshControls {
 maxLocalCells 300000; maxGlobalCells 600000; minRefinementCells 0;
 nCellsBetweenLevels 3; features ();
 refinementSurfaces {hull {level (3 3); patchInfo {type wall;}}}
 resolveFeatureAngle 30;
 refinementRegions {bowBox {mode inside; levels ((1e15 4));}}
 locationInMesh (10 0 1); allowFreeStandingZoneFaces true;
}
snapControls {nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8;}
addLayersControls {relativeSizes true; layers {};}
meshQualityControls {#includeEtc "caseDicts/mesh/generation/meshQualityDict"}
mergeTolerance 1e-6;
''')
(case / "system" / "controlDict").write_text(header % "controlDict" + '''
application foamRun; startFrom startTime; startTime 0; stopAt endTime;
endTime 0; deltaT 1; writeControl timeStep; writeInterval 1;
''')
print(case)
