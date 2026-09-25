"""Stage 5C benchmark execution and configuration integrity."""

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import stage5c_scaling_validation as campaign


def test_agent_benchmark_returns_finite_metrics_and_stable_identity():
    rows = campaign.bench_agents(2, "medium", reps=1, steps=3)
    assert len(rows) == 1
    row = rows[0]
    assert row["count"] == 2 and row["steps"] == 3
    assert all(math.isfinite(row[key]) and row[key] > 0 for key in
               ("steps_per_s", "mean_step_ms", "p50_ms", "p95_ms", "p99_ms"))
    assert row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"]


def test_vector_benchmark_keeps_environment_count_and_reward_mapping():
    rows = campaign.bench_vector(2, 3, reps=1, steps=3)
    assert len(rows) == 1
    row = rows[0]
    assert row["agent_count"] == 2 and row["env_count"] == 3
    assert math.isclose(row["aggregate_env_steps_per_s"],
                        3 * row["per_env_steps_per_s"], rel_tol=1e-12)


def test_sensor_and_environment_configurations_execute():
    for kind in ("dynamics", "gps_imu", "lidar", "sonar", "mixed"):
        row = campaign.bench_custom(lambda kind=kind: campaign.sensor_case(kind),
                                    kind, reps=1, steps=2)[0]
        assert row["steps"] == 2 and row["steps_per_s"] > 0
    for kind in ("still", "current", "wind", "regular", "irregular",
                 "combined", "bathymetry", "obstacles"):
        row = campaign.bench_custom(lambda kind=kind: campaign.environment_case(kind),
                                    kind, reps=1, steps=2)[0]
        assert row["steps"] == 2 and row["steps_per_s"] > 0


def test_memory_sampling_and_reset_cycle_artifacts_are_complete(tmp_path):
    memory_path = tmp_path / "memory.csv"
    memory = campaign.memory_test(memory_path, steps=100, sample_every=20)
    assert memory["steps"] == 100
    with memory_path.open() as file:
        assert len(list(csv.DictReader(file))) == 5
    cycle_path = tmp_path / "resets.csv"
    cycles = campaign.reset_cycles(cycle_path, cycles=21)
    assert cycles["cycles"] == 21
    with cycle_path.open() as file:
        assert len(list(csv.DictReader(file))) == 2


STAGE5C_CASES = (
    "test_agent_benchmark_returns_finite_metrics_and_stable_identity",
    "test_vector_benchmark_keeps_environment_count_and_reward_mapping",
    "test_sensor_and_environment_configurations_execute",
    "test_memory_sampling_and_reset_cycle_artifacts_are_complete",
)


def test_registry_complete():
    assert set(STAGE5C_CASES) == {name for name in globals()
                                   if name.startswith("test_") and name != "test_registry_complete"}
