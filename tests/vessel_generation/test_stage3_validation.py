import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest

def load():
    path=Path(__file__).resolve().parents[2]/"tools"/"stage3_openfoam_validation.py";spec=importlib.util.spec_from_file_location("stage3_openfoam_validation",path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def valid_provenance(module): return {key:{"kind":"KNOWN_PHYSICAL","source":"measurement"} for key in module.REQUIRED_PROVENANCE}

def test_mss_leakage_guard_and_provenance_completeness():
    module=load();module.validate_provenance(valid_provenance(module))
    with pytest.raises(ValueError):module.validate_provenance({})
    with pytest.raises(ValueError):module.leakage_guard({"source":"MSS added mass"})

def test_freeze_hash_is_canonical_and_changes_with_inputs():
    module=load();p=valid_provenance(module);a=module.freeze_config({"mass":1,"geometry":"abc"},p);b=module.freeze_config({"geometry":"abc","mass":1},p);c=module.freeze_config({"mass":2,"geometry":"abc"},p)
    assert a==b and a!=c

def test_cached_cfd_invalidation_covers_all_inputs():
    module=load();base=dict(geometry_hash="a"*64,mesh_settings={"level":"medium"},solver_settings={"residual":1e-6},openfoam_identity="sha256:x");key=module.cache_key(**base);assert module.cache_valid({"cache_key":key,"complete":True},key)
    for field,value in (("geometry_hash","b"*64),("mesh_settings",{"level":"fine"}),("solver_settings",{"residual":1e-7}),("openfoam_identity","sha256:y")):
        changed=dict(base);changed[field]=value;assert module.cache_key(**changed)!=key

def test_metrics_and_stage1_scenario_reuse():
    module=load();reference=np.zeros((4,12));candidate=reference.copy();candidate[-1,0]=1;metrics=module.comparison_metrics(reference,candidate)
    assert metrics["terminal_max_abs"]==1 and metrics["position_rmse"]>0
    assert {"surge","combined_6dof","straight_thrust","differential_thrust"}.issubset(module.STAGE1_SCENARIOS)

def test_missing_otter_geometry_fails_closed_with_artifacts(tmp_path):
    module=load();summary=module.run(tmp_path/"stage3",None)
    assert summary["stage3_status"]=="BLOCKED_GEOMETRY" and summary["cfd_cases"]["completed"]==0 and not summary["model_frozen"]
    for name in ("manifest.json","summary.json","provenance.json","report.md"):assert (tmp_path/"stage3"/name).is_file()

def test_open_visual_hulls_are_rejected(tmp_path,monkeypatch):
    module=load();dae=tmp_path/"otter.dae";dae.write_text('<COLLADA><asset><unit meter="1"/><up_axis>Z_UP</up_axis></asset></COLLADA>')
    obj="v 0 0 0\nv 1 0 0\nv 0 1 0\ng left\nf 1 2 3\nv 2 0 0\nv 3 0 0\nv 2 1 0\ng right\nf 4 5 6\n"
    def convert(command,**kwargs): Path(command[3]).write_text(obj)
    monkeypatch.setattr(module.subprocess,"run",convert)
    result=module.qualify_collada(dae,tmp_path/"geometry")
    assert result["status"]=="BLOCKED_GEOMETRY"
    assert [h["boundary_edges"] for h in result["hull_candidates"]]==[3,3]
