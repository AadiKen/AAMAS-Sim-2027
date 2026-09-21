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
