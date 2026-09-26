import json
import pytest
from bcod_sim.vessel_generation.spec_a.mesh_profiles import PROFILES, background_grid, normalized_difference, select_profile
from bcod_sim.vessel_generation.spec_a.run_matrix import run_matrix


def result(y, n, profile="fast"):
    return dict(status="converged", cell_count=400000, mesh_profile=profile,
                qualification={"mean_foam": [0,y,0,0,0,n]}, state={"beta_deg":8,"r_prime":0,"family":"drift"})


def test_resolution_budget():
    grids = [background_grid((6,6,1.5), p["background_cells"]) for p in PROFILES.values()]
    assert grids[0][1] > grids[1][1] > grids[2][1]
    for (counts, spacing), p in zip(grids, PROFILES.values()):
        assert counts[0]*counts[1]*counts[2] <= p["background_cells"]
        assert counts[0]*counts[1]*counts[2] > .9*p["background_cells"]


@pytest.mark.parametrize("y,n,selected", [(90,90,"fast"),(110,110,"fast"),(89,100,"standard"),(100,111,"standard")])
def test_selection(y,n,selected):
    meta=select_profile(result(y,n), result(100,100,"standard"),force_floor=1,moment_floor=1)
    assert meta["mesh_profile_selected"] == selected
    assert json.loads(json.dumps(meta)) == meta
    assert meta["sentinel_threshold"] == .1


def test_nearzero():
    assert normalized_difference(1e-9,0,1) == 1e-9
    with pytest.raises(ValueError): normalized_difference(0,0,0)


def test_reuse(tmp_path,monkeypatch):
    case=tmp_path/'02_drift';case.mkdir()
    saved=result(1,2);saved['sentinel_reused']=True
    (case/'case_config.json').write_text(json.dumps(dict(state=saved['state'],mesh_profile='fast')))
    (case/'run_result.json').write_text(json.dumps(saved))
    def forbidden(*a,**kw): raise AssertionError('sentinel rerun')
    monkeypatch.setattr('bcod_sim.vessel_generation.spec_a.run_matrix.run_case',forbidden)
    assert run_matrix([case]) == [saved]


def test_yaml_metadata(tmp_path):
    import yaml
    from bcod_sim.vessel_generation.spec_a.fit import CoefficientSurface, write_coefficients_yaml, fit_cases
    surface=CoefficientSurface(5,.3,2,1025,"cubic","cubic",(0,)*6,(0,)*6,(0,)*3)
    meta=select_profile(result(100,100),result(100,100),force_floor=1,moment_floor=1)
    path=tmp_path/'coefficients.yaml'
    write_coefficients_yaml(path,surface,froude={},mesh_selection=meta)
    assert yaml.safe_load(path.read_text())['mesh_selection'] == meta
    with pytest.raises(ValueError,match='mixed mesh'):
        fit_cases([{'mesh_profile':'fast'},{'mesh_profile':'standard'}],length_m=5,draft_m=.3,reference_speed_mps=2)


def test_failed_sentinel_rejected():
    failed=result(1,1);failed['status']='failed'
    with pytest.raises(ValueError,match='qualification'):
        select_profile(failed,result(1,1),force_floor=1,moment_floor=1)


def test_sentinel_pair_only_and_metadata(tmp_path,monkeypatch):
    from bcod_sim.vessel_generation.spec_a.selection import run_sentinels
    calls=[]
    for profile in ('fast','standard'):
        case=tmp_path/'sentinels'/profile;case.mkdir(parents=True)
        (case/'case_config.json').write_text(json.dumps({'case_spec':dict(density_kg_m3=1025,length_m=5,speed_mps=2)}))
    def mesh(cases):
        calls.append(('mesh',cases[0].name));return {'mesh_seconds':1}
    def run(case):
        calls.append(('solve',case.name))
        row=result(100,100,case.name)
        row.update(wall_seconds=1,core_count=1,core_hours=1/3600,iterations=3000)
        return row
    monkeypatch.setattr('bcod_sim.vessel_generation.spec_a.selection.mesh_once',mesh)
    monkeypatch.setattr('bcod_sim.vessel_generation.spec_a.selection.run_case',run)
    meta=run_sentinels(tmp_path)
    assert calls == [('mesh','fast'),('solve','fast'),('mesh','standard'),('solve','standard')]
    assert json.loads((tmp_path/'mesh_selection.json').read_text()) == meta
    assert not list(tmp_path.glob('[0-9][0-9]_*'))


def test_materialization_keeps_one_profile_and_reuses_selected(tmp_path,monkeypatch):
    from bcod_sim.vessel_generation.spec_a.selection import materialize_matrix
    from bcod_sim.vessel_generation.spec_a.matrix import case_matrix
    for profile in ('fast','standard'):
        source=tmp_path/'sentinels'/profile
        (source/'constant/polyMesh').mkdir(parents=True)
        (source/'constant/polyMesh/points').write_text(profile)
        (source/'log.checkMesh').write_text('Mesh OK.')
        (source/'case_config.json').write_text(json.dumps({'mesh_profile':profile,'state':result(1,1)['state']}))
        (source/'run_result.json').write_text(json.dumps(result(1,1,profile)))
    def write(case,spec,state,**kwargs):
        (case/'constant').mkdir(parents=True)
        (case/'case_config.json').write_text(json.dumps({'mesh_profile':kwargs['mesh_profile'],'state':state.__dict__}))
    monkeypatch.setattr('bcod_sim.vessel_generation.spec_a.selection.write_case',write)
    materialize_matrix(tmp_path,None,case_matrix(),mesh_family='displacement_monohull',selection={'mesh_profile_selected':'standard'})
    production=sorted(tmp_path.glob('[0-9][0-9]_*'))
    assert len(production)==10
    assert all(json.loads((p/'case_config.json').read_text())['mesh_profile']=='standard' for p in production)
    assert all((p/'constant/polyMesh/points').read_text()=='standard' for p in production)
    saved=list(tmp_path.glob('[0-9][0-9]_*/run_result.json'))
    assert len(saved)==1
    assert json.loads(saved[0].read_text())['sentinel_reused']
