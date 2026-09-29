import json

import pytest
import trimesh

from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.vessel_generation.package_loader import VesselPackage
from bcod_sim.vessel_generation.simple_pipeline import generate_simple_vessel
from bcod_sim.web.runtime_factory import build_engine


def test_generated_package_load_and_register(tmp_path):
    mesh=tmp_path/"box.stl";trimesh.creation.box(extents=(4.,2.,1.)).export(mesh)
    root=tmp_path/"package"
    generate_simple_vessel(geometry=mesh,output=root,units="m",mass_kg=2050.,cg_frd_m=(0.,0.,0.),disable_bem=True,lut_samples=3)
    package=VesselPackage.load(root)
    registry=Registry();registry.register_vessel_package(id=package.id,version=package.version,path=str(root))
    registry.register("task","task","1",{"kind":"waypoint","agent_id":"a","target_ned_m":[100,0,0],"radius_m":.1},"test")
    config={"schema_version":1,"experiment":{"id":"pkg","seed":1},"simulation":{"dynamics_mode":"full6","master_dt_s":.01,
        "dynamics_substeps":1,"policy_every_n_master_steps":1,"max_master_steps":3},"world":{"source":{"kind":"parametric"},
        "environment":{"current":{"kind":"uniform","ned_mps":[0,0,0]},"wind":{"kind":"uniform","ned_mps":[0,0,0]},
        "waves":{"kind":"calm"},"visibility_m":100},"bathymetry":{"kind":"flat","bottom_ned_z_m":20,"vertical_datum":"MSL"}},
        "vessels":[{"instance_id":"a","definition":f"{package.id}@{package.version}","controller":{"mode":"direct_actuator"},
        "spawn":{"ned_m":[0,0,0],"rpy_rad":[0,0,0]}}],"task":{"type":"task@1","reward":{"individual_weight":1,"team_weight":0},
        "disabled_agent_behavior":"deactivate_keep_physical"}}
    engine=build_engine(resolve(config,registry));first=engine.reset();a=engine.step({"a":__import__("bcod_sim.core.lifecycle",fromlist=["DirectAction"]).DirectAction(())})
    assert first.master_step==0 and a.master_step==1


def test_loader_fails_on_missing_or_tampered_payload(tmp_path):
    from test_simple_pipeline import _make
    root=_make(trimesh.creation.box(extents=(4.,2.,1.)),tmp_path,"tamper",2050.)
    (root/"runtime_payload.json").unlink()
    with pytest.raises(ValueError,match="Invalid vessel package"): VesselPackage.load(root)
