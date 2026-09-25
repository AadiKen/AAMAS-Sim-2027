import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bcod_sim.vessel_generation.cfd import OpenFOAMAdapter
from bcod_sim.vessel_generation.identification import (
    DOF, FluidModel, IdentificationCase, MotionType, SolverSettings,
)
from tools.cfd_smoke.generate_smoke_hull import generate
from tools.verify_openfoam_motion import verify_motion


@pytest.mark.parametrize("motion,dof,expected", [
    (MotionType.FORCED_TRANSLATION, DOF.SURGE, "amplitude (0.1 0.0 0.0)"),
    (MotionType.FORCED_TRANSLATION, DOF.SWAY, "amplitude (0.0 -0.1 0.0)"),
    (MotionType.FORCED_TRANSLATION, DOF.HEAVE, "amplitude (0.0 0.0 -0.1)"),
    (MotionType.FORCED_ROTATION, DOF.ROLL, "axis (1.0 0.0 0.0)"),
    (MotionType.FORCED_ROTATION, DOF.PITCH, "axis (0.0 -1.0 0.0)"),
    (MotionType.FORCED_ROTATION, DOF.YAW, "axis (0.0 0.0 -1.0)"),
])
def test_all_six_motion_definitions(tmp_path, motion, dof, expected):
    geometry = generate(tmp_path / "hull.stl")
    digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
    case = IdentificationCase(digest, motion, dof, amplitude=.1, fluid_model=FluidModel.FREE_SURFACE,
        frequency_rad_s=20., reference_point_frd_m=(.1,.2,.3),
        solver_settings=SolverSettings(write_interval_s=.02))
    root = OpenFOAMAdapter(tmp_path / "cases").generate_identification_case(
        SimpleNamespace(path=geometry, content_hash=digest), case)
    point = (root / "0/pointDisplacement").read_text()
    assert expected in point
    assert "mover {type motionSolver" in (root / "constant/dynamicMeshDict").read_text()
    assert "cellDisplacementFinal" in (root / "system/fvSolution").read_text()
    assert "movingWallVelocity" in (root / "0/U").read_text()
    assert "writeInterval 0.02" in (root / "system/controlDict").read_text()


@pytest.mark.parametrize("name", ["forced_surge", "forced_yaw", "steady_yaw"])
def test_saved_openfoam_hull_points_follow_command(name):
    root = Path(__file__).resolve().parents[2] / "stage3_results/frame-contract-motion" / name
    report = verify_motion(root, .04)
    assert report["accepted"]
    assert report["hull_point_count"] > 0


def test_motion_verifier_rejects_stationary_mesh(tmp_path):
    source = Path(__file__).resolve().parents[2] / "stage3_results/frame-contract-motion/forced_surge"
    import shutil
    for rel in ["case_metadata.json", "constant/polyMesh/boundary",
                "constant/polyMesh/faces", "constant/polyMesh/points"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, target)
    target = tmp_path / "0.04/polyMesh/points"
    target.parent.mkdir(parents=True)
    shutil.copyfile(source / "constant/polyMesh/points", target)
    assert not verify_motion(tmp_path, .04)["accepted"]
