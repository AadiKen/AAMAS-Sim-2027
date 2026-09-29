import math
from dataclasses import replace

import pytest

from bcod_sim.benchmark_v3.config import TaskConfig
from bcod_sim.benchmark_v3.control.actions import decode_action, action_space
from bcod_sim.benchmark_v3.control.high_level import HeadingController
from bcod_sim.benchmark_v3.gym_env import NavigationGymEnv
from bcod_sim.benchmark_v3.runner import schema_record


def test_low_level_mapping_unchanged():
    config = TaskConfig()
    assert [(decode_action([speed, yaw], config).desired_speed_mps,
             decode_action([speed, yaw], config).desired_yaw_rate_radps)
            for speed, yaw in [(-1, -1), (0, 0), (1, 1)]] == [(0, -.25), (1, 0), (2, .25)]


def test_heading_sign_wrap_and_saturation():
    config = TaskConfig(action_mode="high_level")
    assert decode_action([0, 0], config).desired_yaw_rate_radps == 0
    assert decode_action([0, .25], config).desired_yaw_rate_radps == .25
    assert decode_action([0, -.25], config).desired_yaw_rate_radps == -.25
    assert decode_action([1, 1], config).saturated
    controller = HeadingController()
    command = controller.command(1, math.radians(-179), math.radians(179), 0, .2)
    assert command.heading_error_rad == pytest.approx(math.radians(2))
    assert command.desired_yaw_rate_radps > 0


@pytest.mark.parametrize("backend", ["kinematic", "bcod-reduced", "bcod-full"])
def test_decoder_is_backend_independent(backend):
    config = TaskConfig(action_mode="high_level", dynamics=backend)
    command = decode_action([.3, .1], config, heading_rad=.7, yaw_rate_radps=.05)
    reference = decode_action([.3, .1], replace(config, dynamics="kinematic"),
                              heading_rad=.7, yaw_rate_radps=.05)
    assert command == reference
    assert action_space(config.action_mode).shape == (2,)
    assert schema_record(config)["action_schema_version"] == "desired-speed-heading-v1"


def test_gym_high_level_command():
    env = NavigationGymEnv(TaskConfig(action_mode="high_level"))
    try:
        env.reset(seed=11)
        _, _, _, _, info = env.step([0, .25])
        assert info["physical_command"]["yaw_rate_radps"] > 0
        assert info["heading_control"]["heading_error_rad"] > 0
    finally:
        env.close()
