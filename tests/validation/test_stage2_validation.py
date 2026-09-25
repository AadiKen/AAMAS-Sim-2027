from pathlib import Path
import importlib.util
import sys


def load_runner():
    path = Path(__file__).resolve().parents[2]/"tools"/"stage2_validation.py"
    spec = importlib.util.spec_from_file_location("stage2_validation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage2_field_reference_cases(tmp_path):
    runner = load_runner()
    for function in (runner.env_000, runner.cur_001, runner.cur_002, runner.wind_001,
                     runner.wind_003, runner.wreg_001, runner.wreg_002,
                     runner.wirr_001, runner.bath_001, runner.bath_002, runner.comb_001):
        result = function(tmp_path/result_name(function))
        assert result.status == "PASS", result


def result_name(function):
    return function.__name__.replace("_", "-")


def test_stage2_contact_reference_cases(tmp_path):
    runner = load_runner()
    for function in (runner.collision_case("COLL-002"), runner.collision_case("COLL-004", oblique=True),
                     runner.collision_case("VVC-001", dynamic_target=True)):
        result = function(tmp_path/"contact")
        assert result.status == "PASS", result


def test_campaign_executes_grounding_and_writes_artifacts(tmp_path):
    runner = load_runner(); results = runner.run_campaign(tmp_path/"results")
    grounding = next(item for item in results if item.id == "GROUND-001")
    assert grounding.status == "PASS"
    assert all(next(item for item in results if item.id == f"GROUND-{index:03d}").status == "PASS"
               for index in range(1, 6))
    assert (tmp_path/"results"/"manifest.json").is_file()
    assert (tmp_path/"results"/"summary.json").is_file()
    assert (tmp_path/"results"/"cases.csv").is_file()
    assert (tmp_path/"results"/"report.md").is_file()

def test_required_registry_is_complete_and_has_no_blocked_fallbacks():
    runner=load_runner(); required={case for ids in runner.REQUIRED_CASES.values() for case in ids}; registered={case for case,_,_ in runner.CASES}
    assert registered==required
    assert all(function.__name__!="<lambda>" for _,_,function in runner.CASES)

def test_generic_dt_convergence_helper():
    runner=load_runner(); converged=runner.run_case_at_dt(lambda dt:{"terminal":1+dt**4},.04); divergent=runner.run_case_at_dt(lambda dt:{"terminal":dt},.04)
    assert converged["passed"]
    assert not divergent["passed"]

def test_collision_sweep_mirror_and_momentum_helpers(tmp_path):
    runner=load_runner(); sweep=runner.collision_matrix_case("COLL-003")(tmp_path/"sweep");mirror=runner.collision_matrix_case("COLL-005")(tmp_path/"mirror")
    assert sweep.status=="PASS" and mirror.status=="PASS"
    assert runner.vessel_pair(1,5)[3] <= 1e-8
    assert runner.mirrored_error(runner.tensor((1,2)),runner.tensor((1,-2)),(1,-1))==0

def test_controller_and_wave_metric_helpers():
    runner=load_runner(); rows,effort,_=runner.response_run(dt_s=.1,duration_s=.3,controlled=True)
    controller=runner.controller_metrics(rows,effort,200); wave=runner.wave_metrics(rows,0)
    assert set(controller)=={"tracking_rmse","heading_rmse","max_excursion","control_effort","saturation_fraction"}
    assert set(wave)=={"heave_rms","roll_rms","pitch_rms","heading_drift"}
