import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
import torch

from bcod_sim.vessel_generation.cfd import CFDExecutionError, OpenFOAMAdapter
from bcod_sim.vessel_generation.identification import DOF, IdentificationCase, MotionType
from bcod_sim.vessel_generation.frame_contract import (
    body_velocity_to_fixed_hull_inlet, foam_wrench_to_body, point_to_foam,
    vector_to_body, vector_to_foam,
)
from bcod_sim.vessel_generation.fitting import CoefficientFitter, ForceMomentDataset
from bcod_sim.vessel_generation.workflow import AcceptedObservation, identify_package
from bcod_sim.vessel_generation.models import CanonicalVessel
from bcod_sim.dynamics.damping import Damping
from tools.cfd_smoke.generate_smoke_hull import generate


@pytest.mark.parametrize("body,expected", [
    ((1, 0, 0), (-1, 0, 0)),
    ((-1, 0, 0), (1, 0, 0)),
    ((0, 1, 0), (0, 1, 0)),
    ((0, 0, 1), (0, 0, 1)),
])
def test_signed_fixed_hull_inlet(body, expected):
    assert body_velocity_to_fixed_hull_inlet(body) == expected


def test_source_body_and_angular_axes():
    assert vector_to_foam((1, 2, 3)) == (1, -2, -3)
    assert vector_to_body((1, -2, -3)) == (1, 2, 3)
    assert [vector_to_foam(tuple(int(i == j) for i in range(3))) for j in range(3)] == [
        (1, 0, 0), (0, -1, 0), (0, 0, -1)]
    assert point_to_foam((0, 0, 0), .1) == (0, 0, -.1)


@pytest.mark.parametrize("dof,expected", [
    (DOF.SURGE, (-1, 0, 0)), (DOF.SWAY, (0, 1, 0)),
    (DOF.HEAVE, (0, 0, 1)),
])
def test_generated_case_records_signed_inlet_and_center(tmp_path, dof, expected):
    geometry = generate(tmp_path / "hull.stl")
    import hashlib
    digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
    case = IdentificationCase(digest, MotionType.STEADY_VELOCITY, dof,
        magnitude=1., waterline_z_m=.02, cg_frd_m=(.3, .2, .1),
        reference_point_frd_m=(.3, .2, .1))
    reference = SimpleNamespace(path=geometry, content_hash=digest)
    root = OpenFOAMAdapter(tmp_path / "cases").generate_identification_case(reference, case)
    metadata = json.loads((root / "case_metadata.json").read_text())
    assert tuple(metadata["inlet_velocity_foam_mps"]) == expected
    assert metadata["solver_moment_reference_point_m"] == pytest.approx([.3, -.2, -.12])
    control = (root / "system/controlDict").read_text()
    assert "CofR (0.3 -0.2 -0.12000000000000001)" in control or "CofR (0.3 -0.2 -0.12)" in control


def test_force_sign_and_moment_shift():
    # Fluid force at A=(1,0,0), reported about B=(0,0,0).
    force, moment = foam_wrench_to_body((-10, 0, 0), (0, 0, 0),
        foam_reference=(0, 0, 0), body_reference=(0, 0, 0), waterline_z_m=0)
    assert force == (10, 0, 0)  # positive resistance to positive surge
    force, moment = foam_wrench_to_body((0, 10, 0), (0, 0, 0),
        foam_reference=(1, 0, 0), body_reference=(0, 0, 0), waterline_z_m=0)
    assert force == (0, 10, 0)
    assert moment == (0, 0, 10)


def test_parser_requires_window_and_transforms_synthetic_wrench(tmp_path: Path):
    forces = tmp_path / "postProcessing/forces/0/forces.dat"
    forces.parent.mkdir(parents=True)
    forces.write_text("# Time forces(pressure viscous) moments(pressure viscous)\n"
        "0 (0 0 0) (0 0 0) (0 0 0) (0 0 0)\n"
        "1 (-10 0 0) (0 0 0) (0 0 0) (0 0 0)\n")
    (tmp_path / "solver.log").write_text("\nEnd\n")
    (tmp_path / "case_metadata.json").write_text(json.dumps({
        "case_id": "synthetic", "frame_contract": "bcod-openfoam-frd-v1",
        "velocity_body_frd": [1, 0, 0, 0, 0, 0],
        "solver_moment_reference_point_m": [0, 0, 0],
        "moment_reference_point_frd_m": [0, 0, 0],
        "source_waterline_z_m": 0, "solver": "synthetic"}))
    adapter = OpenFOAMAdapter(tmp_path)
    with pytest.raises(CFDExecutionError, match="qualified fitting-window"):
        adapter.parse_case(tmp_path)
    result = adapter.parse_case(tmp_path, debug_last_sample=True)
    assert result.force_body_frd_n == (10, 0, 0)


def test_synthetic_foam_to_package_to_plant_damping(tmp_path: Path):
    rng = np.random.default_rng(7)
    velocities = rng.uniform(-1, 1, (48, 6))
    accelerations = rng.uniform(-1, 1, (48, 6))
    linear = np.array([10., 12., 14., 16., 18., 20.])
    added = np.array([2., 3., 4., 5., 6., 7.])
    quadratic = np.array([1., 2., 3., 4., 5., 6.])
    observations = []
    for i, (v, a) in enumerate(zip(velocities, accelerations)):
        expected = added*a + linear*v + quadratic*np.abs(v)*v
        foam_force = -np.asarray(vector_to_foam(expected[:3]))
        foam_moment = -np.asarray(vector_to_foam(expected[3:]))
        force, moment = foam_wrench_to_body(foam_force, foam_moment,
            foam_reference=(0, 0, 0), body_reference=(0, 0, 0), waterline_z_m=0)
        observations.append(AcceptedObservation(str(i), tuple(v), tuple(a), force+moment,
            True, "bcod-openfoam-frd-v1", (1., 2.)))
    geometry = generate(tmp_path / "hull.stl")
    output = identify_package(name="synthetic", geometry_path=geometry, mass_kg=5.,
        cg_frd_m=(0., 0., 0.), observations=tuple(observations), output=tmp_path / "vessel",
        solver={"name":"synthetic", "version":"test"},unsafe_debug_observations=True)
    vessel = CanonicalVessel.model_validate(yaml.safe_load((output / "vessel.yaml").read_text()))
    assert np.allclose(np.diag(vessel.linear_damping_matrix), linear, atol=1e-8)
    assert np.allclose(np.diag(vessel.added_mass_kg), added, atol=1e-8)
    assert np.allclose(vessel.quadratic_damping, quadratic, atol=1e-8)
    damping = Damping(torch.tensor(vessel.linear_damping, dtype=torch.float64),
        torch.tensor(vessel.quadratic_damping, dtype=torch.float64),
        linear_matrix=torch.tensor(vessel.linear_damping_matrix, dtype=torch.float64))
    for axis in range(6):
        velocity = torch.zeros(6, dtype=torch.float64)
        velocity[axis] = 1
        assert damping.wrench(velocity)[axis] == pytest.approx(-linear[axis]-quadratic[axis])
