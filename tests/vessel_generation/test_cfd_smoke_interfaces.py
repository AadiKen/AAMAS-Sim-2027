from pathlib import Path

import numpy as np
import pytest

from bcod_sim.vessel_generation.artifacts import write_hash_manifest, verify_hash_manifest
from bcod_sim.vessel_generation.cfd import CFDExecutionError, OpenFOAMAdapter
from bcod_sim.vessel_generation.geometry import import_ascii_stl
from tools.cfd_smoke.generate_smoke_hull import generate


def test_smoke_geometry_is_deterministic_closed_and_dimensioned(tmp_path):
    a=generate(tmp_path/"a.stl"); b=generate(tmp_path/"b.stl")
    first=import_ascii_stl(a,tmp_path/"a-normalized.stl",source_frame="body_FRD_m")
    second=import_ascii_stl(b,tmp_path/"b-normalized.stl",source_frame="body_FRD_m")
    assert first.content_hash==second.content_hash
    assert np.allclose(first.bounds_m,((-1,-.2,-.075),(1,.2,.075)))
    with pytest.raises(CFDExecutionError):
        import_ascii_stl(Path(__file__),tmp_path/"bad.stl",source_frame="body_FRD_m")


def test_production_case_generator_has_real_mesh_solver_and_force_contract(tmp_path):
    source=generate(tmp_path/"source.stl")
    geometry=import_ascii_stl(source,tmp_path/"normalized.stl",source_frame="body_FRD_m")
    root=OpenFOAMAdapter(tmp_path/"cases").generate_operating_case(geometry,case_id="surge_1p0",surge_mps=1.)
    assert "snappyHexMesh" in (root/"system"/"snappyHexMeshDict").read_text()
    control=(root/"system"/"controlDict").read_text()
    assert "incompressibleFluid" in control and "type forces" in control and "patches (hull)" in control
    assert (root/"0"/"U").exists() and (root/"0"/"p").exists()


def test_integrity_manifest_rejects_tampering(tmp_path):
    target=tmp_path/"raw.dat"; target.write_text("actual solver evidence")
    manifest=write_hash_manifest(tmp_path,tmp_path/"manifest.json",exclude=("manifest.json",))
    verify_hash_manifest(tmp_path,manifest); target.write_text("tampered")
    with pytest.raises(ValueError,match="integrity failure"): verify_hash_manifest(tmp_path,manifest)
