from pathlib import Path

import pytest

pytest.importorskip("benchmarl")

from bcod_sim.benchmark_v3.marl.curriculum import (
    Curriculum, CurriculumController, CurriculumStage, stage_sampler, write_stage)
from bcod_sim.benchmark_v3.marl.scenarios import (
    bank_hash, m0_curriculum_bank, reserved_m0_curriculum_hashes,
    sample_m0_curriculum_train)
from bcod_sim.benchmark_v3.marl.curriculum_train import load_recipe
from bcod_sim.benchmark_v2.backend import VesselReading
from bcod_sim.benchmark_v3.config import TaskConfig
from bcod_sim.benchmark_v3.observations import build_observation


def test_curriculum_banks_are_unique_and_fresh():
    banks = [m0_curriculum_bank(stage, split) for stage in "ABC"
             for split in ("dev", "test")]
    assert [len(bank) for bank in banks] == [50, 100] * 3
    assert len(reserved_m0_curriculum_hashes()) == 450
    assert len({bank_hash(bank) for bank in banks}) == 6
    assert all(len(case.starts) == 2 and not case.obstacles
               for bank in banks for case in bank)
    import numpy as np
    rng = np.random.default_rng(913)
    reserved = reserved_m0_curriculum_hashes()
    for stage in "ABC":
        assert all(sample_m0_curriculum_train(rng, stage).geometry_hash() not in reserved
                   for _ in range(100))


def test_promotion_requires_two_consecutive_evaluations_and_preserves_state(tmp_path):
    path = Path("configs/benchmark_v3/marl/m0_highlevel_mappo_curriculum_v1.yaml")
    _, curriculum = load_recipe(path)
    controller = CurriculumController(curriculum)
    assert controller.record(6000, success_rate=.92, collision_rate=.01) == "continue"
    assert controller.record(12000, success_rate=.89, collision_rate=.01) == "continue"
    assert controller.record(18000, success_rate=.91, collision_rate=.01) == "continue"
    assert controller.record(24000, success_rate=.93, collision_rate=.01) == "promote"
    assert controller.stage.name == "B" and controller.entry_step == 24000
    assert controller.record(30000, success_rate=.9, collision_rate=.2) == "continue"
    assert controller.record(36000, success_rate=.86, collision_rate=.08) == "continue"
    assert controller.record(42000, success_rate=.87, collision_rate=.09) == "promote"
    assert controller.stage.name == "C"
    stage_path = tmp_path / "stage.json"
    write_stage(stage_path, "A")
    sample = stage_sampler(stage_path, lambda _rng, stage: stage)
    assert sample(None) == "A"
    write_stage(stage_path, "B")
    assert sample(None) == "B"


def test_local_actor_observation_distinguishes_four_neighbor_motions():
    cfg = TaskConfig(agent_count=2)
    ego = VesselReading(0., 0., 0., 0., 0.)
    other = VesselReading(5., 0., 0., 0., 0.)
    now = {"vessel_0": ego, "vessel_1": other}
    prior_positions = {"approaching": (5.2, 0.), "receding": (4.8, 0.),
                       "crossing_left": (5., -.2), "crossing_right": (5., .2)}
    observations = {}
    for motion, (x, y) in prior_positions.items():
        before = {"vessel_0": ego, "vessel_1": VesselReading(x, y, 0., 0., 0.)}
        observations[motion] = build_observation("vessel_0", ego, (20., 0.),
                                                 now, before, 1, cfg)
    assert len({tuple(obs[9:15]) for obs in observations.values()}) == 4
    assert all(obs.shape == (78,) for obs in observations.values())
