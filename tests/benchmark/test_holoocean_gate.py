"""HoloOcean Linux adapter wiring; native engine qualification runs on Linux."""
import math

import pytest

from bcod_sim.benchmark.core import BenchmarkConfig, generate_scenario
from bcod_sim.benchmark.holoocean_adapter import HoloOceanAdapter, HoloOceanUnavailable, vessel_scenario_config


class FakeOcean:
    def __init__(self, scenario_cfg, show_viewport):
        self.config = scenario_cfg
        self.props = []
        self.actions = {}
        self.ticks = 0

    def reset(self):
        return {agent["agent_name"]: {
            "GPSSensor": agent["location"], "RotationSensor": agent["rotation"]}
            for agent in self.config["agents"]}

    def spawn_prop(self, kind, location, **kwargs):
        self.props.append((kind, location, kwargs))

    def act(self, name, action):
        self.actions[name] = action

    def tick(self, num_ticks):
        self.ticks += num_ticks
        return self.reset()

    def close(self):
        pass


def test_four_vessel_linux_configuration_and_prop_mapping():
    scenario = generate_scenario(8)
    config = vessel_scenario_config(scenario)
    assert len(config["agents"]) == 4
    for source, vessel in zip(scenario.starts, config["agents"]):
        assert vessel["location"][:2] == [source.y_m, -source.x_m]
        assert vessel["rotation"][2] == pytest.approx(math.degrees(source.heading_rad - math.pi / 2))
    created = []
    def factory(**kwargs):
        fake = FakeOcean(**kwargs)
        created.append(fake)
        return fake
    adapter = HoloOceanAdapter(BenchmarkConfig(), environment_factory=factory)
    try:
        readings, truth = adapter.reset(scenario, scenario.seed)
        fake = created[0]
        assert len(fake.props) == len(scenario.obstacles)
        for obstacle, (kind, position, options) in zip(scenario.obstacles, fake.props):
            assert kind == "sphere"
            assert position[:2] == [obstacle.y_m, -obstacle.x_m]
            assert options["scale"] == 2 * obstacle.radius_m
        for i, source in enumerate(scenario.starts):
            measured = readings[f"vessel_{i}"]
            assert (measured.x_m, measured.y_m) == pytest.approx((source.x_m, source.y_m))
        adapter.step({f"vessel_{i}": (0.5, 0.5) for i in range(4)})
        assert fake.ticks == 10 and len(fake.actions) == 4
        assert all(action[1] > action[0] for action in fake.actions.values())
    finally:
        adapter.close()


def test_close_calls_callable_close_once_and_clears_state():
    class Closable:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1

    adapter = HoloOceanAdapter(environment_factory=lambda **_: None)
    env = Closable()
    adapter.env = env
    adapter.latest_readings = {"vessel_0": object()}
    adapter.close()
    adapter.close()
    assert env.calls == 1
    assert adapter.env is None
    assert not hasattr(adapter, "latest_readings")


def test_close_without_shutdown_method_is_safe():
    adapter = HoloOceanAdapter(environment_factory=lambda **_: None)
    adapter.env = object()
    adapter.latest_readings = {}
    adapter.close()
    adapter.close()
    assert adapter.env is None
    assert not hasattr(adapter, "latest_readings")


def test_close_uses_holoocean_context_manager_teardown():
    class ContextEnvironment:
        def __init__(self):
            self.exits = []

        def __exit__(self, exc_type, exc_value, traceback):
            self.exits.append((exc_type, exc_value, traceback))

    adapter = HoloOceanAdapter(environment_factory=lambda **_: None)
    env = ContextEnvironment()
    adapter.env = env
    adapter.close()
    adapter.close()
    assert env.exits == [(None, None, None)]
    assert adapter.env is None


def test_close_before_environment_creation():
    adapter = HoloOceanAdapter(environment_factory=lambda **_: None)
    adapter.close()
    adapter.close()
    assert adapter.env is None
    assert not hasattr(adapter, "latest_readings")
