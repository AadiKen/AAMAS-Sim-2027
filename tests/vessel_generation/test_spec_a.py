import math

import pytest
import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.vessel_generation.spec_a.empirical import clarke_linear, compare_v5
from bcod_sim.vessel_generation.spec_a.fit import CoefficientSurface, fit_cases
from bcod_sim.vessel_generation.spec_a.matrix import case_matrix, froude_gate
from bcod_sim.vessel_generation.spec_a.make_case import VesselCaseSpec, write_case
from bcod_sim.vessel_generation.spec_a.extract_forces import (
    parse_force_history, qualify_force_tail, physical_frd_row)
import trimesh
import numpy as np


def test_matrix_drift_sign_and_froude_gate():
    states = case_matrix()
    assert len(states) == 10
    u, v, r = states[1].body_velocity(2., 5.)
    assert u == pytest.approx(2*math.cos(math.radians(4)))
    assert v == pytest.approx(-2*math.sin(math.radians(4)))
    assert r == 0
    assert froude_gate(0, 5)["status"] == "eligible"
    speed = math.sqrt(9.80665*5)
    assert froude_gate(.35*speed, 5)["status"] == "extrapolated_fr"
    assert froude_gate(.5*speed, 5)["status"] == "out_of_envelope"


def test_clarke_formula_and_v5_comparison():
    result = clarke_linear(100., 15., 6., .7)
    scale = math.pi*(6/100)**2
    assert result["Y_v"] == pytest.approx(-scale*(1+.4*.7*15/6))
    assert result["N_v"] < 0
    decision = compare_v5(result, length_m=100, beam_m=15, draft_m=6,
                          block_coefficient=.7, max_beta_deg=10, max_abs_r_prime=.3)
    assert decision["decision"] == "ship_v5"


def test_synthetic_fit_and_physical_sign():
    target = CoefficientSurface(5., .3, 2., 1025., "cubic", "cubic",
                                (-.2, -.03, -1., .1, .2, -.05),
                                (-.1, -.04, -.3, .2, -.1, -.02), (-.2, -.1, .03))
    rows = []
    for state in case_matrix():
        u, v, r = state.body_velocity(2., 5.)
        dx, y, n = target.evaluate(u, v, r)
        rows.append(dict(u_mps=u, v_mps=v, r_rad_s=r, X_n=dx-100., Y_n=y, N_nm=n))
    fitted, diagnostics = fit_cases(rows, length_m=5., draft_m=.3, reference_speed_mps=2.)
    assert fitted.y_basis == fitted.n_basis == "cubic"
    assert max(abs(z) for z in diagnostics["Y"]["residuals"]) < 1e-10
    assert diagnostics["Y"]["rank_deficient_loo_folds"] == [8, 9]
    u, v, r = case_matrix()[1].body_velocity(2., 5.)
    assert v < 0
    assert fitted.evaluate(u, v, r)[1] > 0
    assert fitted.evaluate(u, v, r)[2] > 0
    assert fitted.evaluate(u, -v, -r)[1] == pytest.approx(-fitted.evaluate(u, v, r)[1])


def _plant(surface=None, added_mass_coriolis_enabled=True):
    dtype = torch.float64
    tensor = lambda x: torch.tensor(x, dtype=dtype)
    added = torch.diag(tensor((1., 4., 0., 0., 0., 2.)))
    mass = MassProperties(10., tensor((0., 0., 0.)), torch.diag(tensor((3., 4., 5.))), added)
    return Plant6(mass, Damping(tensor((0.,)*6), tensor((0.,)*6)),
                  Hydrostatics(10*9.80665, tensor((0., 0., 0.))),
                  OperatingEnvelope(tensor((100.,)*6), max_substep_s=1.),
                  maneuvering_surface=surface,
                  added_mass_coriolis_enabled=added_mass_coriolis_enabled)


def _state(u, v, r=0.):
    return VesselState(torch.zeros(3, dtype=torch.float64),
                       torch.tensor((1., 0., 0., 0.), dtype=torch.float64),
                       torch.tensor((u, v, 0., 0., 0., r), dtype=torch.float64))


def _external():
    from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
    return {name: torch.zeros(6, dtype=torch.float64) for name in EXTERNAL_TERMS}


def test_munk_once_and_surface_disables_added_coriolis():
    baseline = _plant()
    ledger = baseline.diagnostics(_state(2., .3), _external())
    assert ledger.terms["added_mass_coriolis"][5].item() == pytest.approx(-(4.-1.)*2.*.3)
    surface = CoefficientSurface(5., .3, 2., 1025., "cubic", "cubic",
                                 (0.,)*6, (0.,)*6, (0., 0., 0.))
    with pytest.raises(PhysicalValidationError, match="Coriolis disabled"):
        _plant(surface)
    hybrid = _plant(surface, added_mass_coriolis_enabled=False)
    active = hybrid.diagnostics(_state(2., .3), _external())
    assert torch.count_nonzero(active.terms["added_mass_coriolis"]) == 0
    assert active.component_diagnostics["maneuvering_surface_active"]
    # Reverse uses the historical V5/C_A path.
    reverse = hybrid.diagnostics(_state(-2., .3), _external())
    assert reverse.terms["added_mass_coriolis"][5].item() == pytest.approx((4.-1.)*2.*.3)


def test_case_generator_uses_verified_frame_and_one_hull(tmp_path):
    hull = trimesh.creation.box(extents=(2., .5, .3))
    hull.apply_translation((0, 0, .15))
    path = tmp_path/"hull.stl"
    hull.export(path)
    spec = VesselCaseSpec(path, 2., .5, .3, .75, 1025., 1e-6, .5, (0., 0., .1))
    drift = write_case(tmp_path/"drift", spec, case_matrix()[1])
    assert drift["body_velocity_frd"][1] < 0
    assert drift["inlet_foam"][1] < 0
    assert "symmetryPlane" in (tmp_path/"drift/system/blockMeshDict").read_text()
    assert "rhoInf 1025" in (tmp_path/"drift/system/controlDict").read_text()
    yaw = write_case(tmp_path/"turn", spec, case_matrix()[5])
    assert yaw["rotation_method"] == "OpenFOAM11-MRF-all"
    assert "MRFFreestreamVelocity" in (tmp_path/"turn/0/U").read_text()


def test_force_extraction_physical_frd_and_tail(tmp_path):
    line = "1 ((1 2 0) (0.5 0 0)) ((0 0 3) (0 0 1))\n"
    path = tmp_path/"forces.dat"
    path.write_text("# Time forces(pressure viscous) moments(pressure viscous)\n"+line)
    values = parse_force_history(path)
    assert values.shape == (1, 7)
    assert values[0, 1] == pytest.approx(1.5)
    qualified = {"status": "converged", "mean_foam": values[0, 1:].tolist()}
    row = physical_frd_row(case_id="drift", u=1, v=-.1, r=0,
                           qualified=qualified, foam_reference=(0,0,0),
                           body_reference=(0,0,0), waterline_z_m=0)
    assert (row["X_n"], row["Y_n"], row["N_nm"]) == pytest.approx((1.5, -2., -4.))
    history = np.column_stack((np.arange(1, 3001), np.tile(values[0, 1:], (3000, 1))))
    log = "\n".join(f"Initial residual = {max(1e-8, .01*math.exp(-i/100)):.9g}" for i in range(3000))
    result = qualify_force_tail(history, residual_log=log)
    assert result["status"] == "converged"


def test_large_periodic_band_cannot_pass_by_equal_block_means():
    time = np.arange(1, 3001)
    values = np.ones((3000, 6))
    values[:, 1] = 1+.2*np.sin(2*np.pi*time/50)
    values[:, 5] = 1+.2*np.sin(2*np.pi*time/50)
    history = np.column_stack((time, values))
    log = "Initial residual = 1e-8\n"*3000
    assert qualify_force_tail(history, residual_log=log)["status"] == "failed"
    values[:, 1] = 1+.02*np.sin(2*np.pi*time/50)
    values[:, 5] = 1+.02*np.sin(2*np.pi*time/50)
    result = qualify_force_tail(np.column_stack((time, values)), residual_log=log)
    assert result["status"] == "oscillatory"
    assert result["window"] == 1000


def test_yaml_is_portable_and_invalid_coefficients_rejected(tmp_path):
    import yaml
    from bcod_sim.vessel_generation.spec_a.fit import write_coefficients_yaml
    surface = CoefficientSurface(5,.3,2,1025,"cubic","cubic", np.zeros(6), np.zeros(6), np.zeros(3))
    write_coefficients_yaml(tmp_path/"coefficients.yaml", surface, froude=froude_gate(2,5))
    data = yaml.safe_load((tmp_path/"coefficients.yaml").read_text())
    assert CoefficientSurface(**data["coefficients"]) == surface
    with pytest.raises(ValueError):
        CoefficientSurface(5,.3,2,1025,"cubic","cubic", (float("nan"),)*6, (0,)*6, (0,)*3)


def test_runner_retries_once_and_halves_relaxation(tmp_path, monkeypatch):
    import json
    from importlib import import_module
    runner = import_module("bcod_sim.vessel_generation.spec_a.run_matrix")
    (tmp_path/"system").mkdir()
    (tmp_path/"constant/polyMesh").mkdir(parents=True)
    (tmp_path/"3000").mkdir()
    (tmp_path/"case_config.json").write_text(json.dumps({"state": {"beta_deg": 8, "r_prime": 0}}))
    (tmp_path/"log.checkMesh").write_text("Mesh OK.\n")
    (tmp_path/"system/controlDict").write_text("startFrom startTime; endTime 3000;")
    (tmp_path/"system/fvSolution").write_text("relaxationFactors {fields {p 0.3;} equations {U 0.7; k 0.7; omega 0.7;}}")
    calls = []
    def solver(*args, **kwargs):
        calls.append(args)
        kwargs["stdout"].write("End\n")
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(runner.subprocess, "run", solver)
    monkeypatch.setattr(runner, "case_force_history", lambda case: np.ones((3001,7)))
    outcomes = iter([{"status":"failed", "reason":"band"},
                     {"status":"converged", "mean_foam":[1]*6}])
    monkeypatch.setattr(runner, "qualify_force_tail", lambda *a, **kw: next(outcomes))
    result = runner.run_case(tmp_path)
    assert len(calls) == 2
    assert result["status"] == "converged"
    assert "p 0.15" in (tmp_path/"system/fvSolution").read_text()
    assert "U 0.35" in (tmp_path/"system/fvSolution").read_text()
    assert "endTime 6000" in (tmp_path/"system/controlDict").read_text()


def test_matrix_stops_after_third_failure(tmp_path, monkeypatch):
    import json
    from importlib import import_module
    runner = import_module("bcod_sim.vessel_generation.spec_a.run_matrix")
    cases = []
    for i in range(5):
        p = tmp_path/str(i); p.mkdir()
        (p/"case_config.json").write_text(json.dumps({"state":{"beta_deg":4*i,"r_prime":0}}))
        cases.append(p)
    called = []
    def fail(case, **kwargs):
        called.append(case)
        return {"status":"failed", "case":str(case)}
    monkeypatch.setattr(runner, "run_case", fail)
    with pytest.raises(RuntimeError, match="More than two"):
        runner.run_matrix(cases)
    assert len(called) == 3
    assert called[0].name == "2"


def test_capped_matrix_keeps_ten_distinct_design_states():
    states = case_matrix(max_beta_deg=5, max_abs_r_prime=.15)
    assert len(states) == len(set(states)) == 10
    assert max(row.beta_deg for row in states) == 5
    assert max(abs(row.r_prime) for row in states) == .15


def test_current_relative_surface_and_authoritative_surge():
    surface = CoefficientSurface(5,.3,2,1025,"cubic","cubic",
                                 (-.1,)*6, (-.02,)*6, (-.1,0,0))
    plant = _plant(surface, added_mass_coriolis_enabled=False)
    plant.damping = Damping(torch.tensor((2.,0,0,0,0,0),dtype=torch.float64),
                            torch.zeros(6,dtype=torch.float64))
    state = _state(2., .3)
    water = torch.tensor((.5, .1, 0.), dtype=torch.float64)
    ledger = plant.diagnostics(state, _external(), water)
    dx, y, n = surface.evaluate(1.5,.2,0)
    assert ledger.terms["linear_damping"][0].item() == pytest.approx(-3.)
    assert ledger.terms["nonlinear_damping"][0].item() == pytest.approx(dx)
    assert ledger.terms["crossflow"][[1,5]].tolist() == pytest.approx([y,n])


def test_episode_randomization_is_deterministic_and_preserves_stability():
    from bcod_sim.vessel_generation.spec_a.randomize import sample_episode, stability_index
    surface = CoefficientSurface(5,.3,2,1025,"cubic","cubic",
                                 (-.1,.01,-.1,.01,.01,-.1), (-.02,-.03,-.1,.01,.01,-.1), (-.1,0,0))
    one, log = sample_episode(surface, mass_kg=100, seed=17)
    two, repeated = sample_episode(surface, mass_kg=100, seed=17)
    assert one == two and log == repeated
    assert np.sign(stability_index(one,100)) == np.sign(stability_index(surface,100))
    assert .9 <= log["added_mass_multiplier"] <= 1.1


def test_sysid_holds_out_one_maneuver():
    from bcod_sim.vessel_generation.spec_a.sysid import fit_output_error
    maneuvers = [{"velocity_uvw_r":np.ones((3,3))*value, "scale":value}
                 for value in (1.,2.,3.)]
    def simulator(parameters, maneuver):
        return np.ones((3,3))*maneuver["scale"]*parameters[0]
    result = fit_output_error([.8], maneuvers, simulator=simulator, holdout_index=2, ridge_weight=0.)
    assert result["coefficients"] == pytest.approx([1.])
    assert result["held_out_rmse"] < 1e-8


def test_nonzero_waterline_and_turn_center_geometry(tmp_path):
    import re
    hull = trimesh.creation.box(extents=(2.,.5,.3));hull.apply_translation((0,0,.65))
    path = tmp_path/'hull.stl';hull.export(path)
    spec = VesselCaseSpec(path,2.,.5,.3,.75,1025.,1e-6,.5,(.1,.02,.7),waterline_frd_z_m=.5)
    state = case_matrix()[-1]
    result = write_case(tmp_path/'turn',spec,state)
    assert result['moment_reference_foam'] == pytest.approx((.1,-.02,0.))
    mrf = (tmp_path/'turn/constant/MRFProperties').read_text()
    center = np.array(list(map(float,re.search(r'origin \(([^)]+)\)',mrf)[1].split())))
    u,v,r = state.body_velocity(.5,2.)
    velocity = np.cross([0,0,-r],np.array(result['moment_reference_foam'])-center)
    assert velocity == pytest.approx([u,-v,0])
    assert result['inlet_foam'] == (0.,0.,0.)


def test_prolate_spheroid_analytic_munk_once():
    # Potential-flow ellipsoid added-mass factors (a>b=c). The physical
    # moment is -(m_y-m_x)uv = -0.5(m_y-m_x)U² sin(2 beta).
    a,b,rho = 2.,.5,1000.
    eccentricity = math.sqrt(1-(b/a)**2)
    alpha = 2*(1-eccentricity**2)/eccentricity**3*(math.atanh(eccentricity)-eccentricity)
    beta_factor = 1-alpha/2
    displacement = rho*4*math.pi*a*b*b/3
    mx = displacement*alpha/(2-alpha)
    my = displacement*beta_factor/(2-beta_factor)
    plant = _plant()
    plant.added_mass = torch.diag(torch.tensor([mx,my,my,0.,0.,0.],dtype=torch.float64))
    angle = .1; speed = 2.
    state = _state(speed*math.cos(angle),speed*math.sin(angle))
    ledger = plant.diagnostics(state,_external())
    expected = -.5*(my-mx)*speed**2*math.sin(2*angle)
    assert ledger.total[5].item() == pytest.approx(expected)
    assert ledger.terms['added_mass_coriolis'][5].item() == pytest.approx(expected)


def test_runtime_payload_requires_physical_force_contract():
    from bcod_sim.vessel_generation.spec_a.fit import surface_from_payload
    with pytest.raises(ValueError,match='physical fluid-on-hull'):
        surface_from_payload({'schema':'bcod-maneuvering-spec-a-v1',
                              'force_convention':'bcod_resisting_wrench'})


def test_restart_replaces_invalid_superseded_force_samples(tmp_path):
    from bcod_sim.vessel_generation.spec_a.extract_forces import case_force_history
    root = tmp_path/'postProcessing/forces'
    (root/'0').mkdir(parents=True); (root/'500').mkdir()
    row = '((1 2 3) (0 0 0)) ((0 0 4) (0 0 0))\n'
    (root/'0/forces.dat').write_text('1 '+row+'500 ((nan 2 3) (0 0 0)) ((0 0 4) (0 0 0))\n')
    (root/'500/forces.dat').write_text('500 '+row+'501 '+row)
    history = case_force_history(tmp_path)
    assert history[:,0].tolist() == [1,500,501]
    assert np.isfinite(history).all()


def test_surface_reference_velocity_and_moment_transform():
    surface = CoefficientSurface(5,.3,2,1025,"cubic","cubic",
                                 (-.1,0,0,0,0,0), (-.02,0,0,0,0,0), (0,0,0),
                                 moment_reference_frd_m=(.3,.1,0.))
    model = _plant(surface, added_mass_coriolis_enabled=False)
    state = _state(2.,.2,.1)
    expected_u,expected_v = 2.-.1*.1, .2+.3*.1
    dx,y,n = surface.evaluate(expected_u,expected_v,.1)
    ledger = model.diagnostics(state,_external())
    assert ledger.terms['crossflow'][1].item() == pytest.approx(y)
    assert ledger.terms['crossflow'][5].item() == pytest.approx(n+.3*y-.1*dx)


def test_output_error_adapter_executes_plant6_with_fixed_mass():
    from bcod_sim.vessel_generation.spec_a.sysid import plant6_simulator, fit_output_error
    def factory(parameters):
        surface = CoefficientSurface(5,.3,2,1025,'cubic','cubic',
                                     (float(parameters[0]),0,0,0,0,0),(0,)*6,(0,)*3)
        return _plant(surface, added_mass_coriolis_enabled=False)
    simulator = plant6_simulator(factory)
    maneuvers = []
    for sway in (.1,.2):
        row = {'time_s':np.linspace(0,.1,6),'initial_state':_state(1.,sway),
               'actuation_wrench_frd':np.zeros((6,6)),
               'fixed_actuator_model_id':'synthetic-zero-actuator'}
        row['velocity_uvw_r'] = simulator(np.array([-.001]),row)
        maneuvers.append(row)
    fitted = fit_output_error([-.0007],maneuvers,simulator=simulator,holdout_index=1,ridge_weight=0.)
    assert fitted['coefficients'] == pytest.approx([-.001],rel=1e-4)
    assert fitted['held_out_rmse'] < 1e-6
