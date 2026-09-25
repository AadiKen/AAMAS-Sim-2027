import importlib.util,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load():
    path=ROOT/"tools/stage3_wamv_validation.py";spec=importlib.util.spec_from_file_location("stage3_wamv_validation",path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def qualified(module):
    vertices,faces,semantics=module.load_collada(ROOT/"stage3_inputs/wamv/vrx-41f2df50/original/WAM-V-Base.dae")
    points,wfaces=module.welded_mesh(vertices,faces);components=sorted(module.face_components(points,wfaces),key=lambda mesh:len(mesh[1]),reverse=True)
    return semantics,components,module.combine(components[:2])

def test_collada_units_transform_and_hull_topology():
    module=load();semantics,components,combined=qualified(module);stats=[module.mesh_stats(*mesh) for mesh in components[:2]]
    assert semantics["unit_meter"]==.01 and semantics["up_axis"]=="Z_UP"
    assert all(item["watertight"] and item["boundary_edges"]==0 for item in stats)
    assert stats[0]["extents_m"]==pytest.approx([4.931483,.432987,.559184],abs=1e-6)
    assert module.mesh_stats(*combined)["enclosed_volume_m3"]==pytest.approx(1.231104334,rel=1e-8)

def test_geometry_derived_hydrostatic_equilibrium():
    module=load();_,_,combined=qualified(module);result=module.hydrostatics(*combined)
    assert result["equivalent_mass_kg"]==pytest.approx(180,abs=.01)
    assert result["draft_from_lowest_point_m"]==pytest.approx(.099915,abs=2e-4)
    assert result["waterplane_area_m2"]>0 and result["heave_restoring_N_m"]>0

def test_reference_leakage_guard_and_cache_invalidation():
    module=load();module.guard_pre_freeze_reference({"mass_kg":180,"geometry":"vrx mesh"})
    with pytest.raises(ValueError):module.guard_pre_freeze_reference({"vrx_added_mass":[1,2,3]})
    base=module.cfd_cache_key("a",{"level":"medium"},{"solver":"interFoam"},{"rho":1025},{"u":1})
    assert base!=module.cfd_cache_key("a",{"level":"fine"},{"solver":"interFoam"},{"rho":1025},{"u":1})
