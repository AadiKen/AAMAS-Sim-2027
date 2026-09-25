"""Linux HoloOcean SurfaceVessel bridge for the shared navigation benchmark.

The Ocean engine is installed and run on Linux. Tests can inject a small engine
fake; they do not stand in for native engine qualification.
"""

import math
import platform

from .core import BenchmarkConfig, NAMES, Scenario, Truth, VesselReading


class HoloOceanUnavailable(RuntimeError):
    pass


def vessel_scenario_config(scenario: Scenario, *, package_name: str = "Ocean",
                           world: str = "OpenWater") -> dict:
    return {"name": "FourVesselBenchmark", "package_name": package_name, "world": world,
            "main_agent": NAMES[0], "ticks_per_sec": 50, "frames_per_sec": 50,
            "agents": [{"agent_name": name, "agent_type": "SurfaceVessel", "control_scheme": 0,
                        "location": [scenario.starts[i].y_m, -scenario.starts[i].x_m, 2.],
                        "rotation": [0., 0., math.degrees(scenario.starts[i].heading_rad - math.pi / 2)],
                        "sensors": [{"sensor_type": sensor} for sensor in
                                    ("GPSSensor", "IMUSensor", "RotationSensor", "CollisionSensor")]}
                       for i, name in enumerate(NAMES)]}


class HoloOceanAdapter:
    def __init__(self, config: BenchmarkConfig = BenchmarkConfig(), *, environment_factory=None):
        self.config = config
        if not math.isclose(config.dt_s * 50, round(config.dt_s * 50), abs_tol=1e-9):
            raise ValueError("HoloOcean timestep must be an integer number of 50 Hz engine ticks")
        if environment_factory is None:
            if platform.system() != "Linux":
                raise HoloOceanUnavailable("HoloOcean benchmark requires a Linux engine host")
            try:
                import holoocean
            except ImportError as exc:
                raise HoloOceanUnavailable("Install the Linux HoloOcean client and Ocean world") from exc
            environment_factory = holoocean.make
        self.factory = environment_factory
        self.env = None

    def reset(self, scenario: Scenario, seed: int):
        if seed != scenario.seed:
            raise ValueError("Scenario seed and reset seed must match")
        self.close()
        self.scenario = scenario
        self.env = self.factory(scenario_cfg=vessel_scenario_config(scenario), show_viewport=False)
        state = self.env.reset()
        # Props are removed by reset. Scale is a diameter because the base prop
        # is 1 m across; the obstacle contract stores a radius.
        for index, obstacle in enumerate(scenario.obstacles):
            self.env.spawn_prop("sphere", [obstacle.y_m, -obstacle.x_m, 2.],
                                scale=2 * obstacle.radius_m, sim_physics=False,
                                tag=f"benchmark_obstacle_{index}")
        self.previous_positions = {}
        self.previous_headings = {}
        result = self._read(state, initial=True)
        self.latest_readings = result[0]
        return result

    def _read(self, state, *, initial=False):
        readings, truth = {}, {}
        positions = {}
        for name in NAMES:
            sensors = state[name]
            gps = sensors.get("GPSSensor")
            rotation = sensors.get("RotationSensor")
            if gps is None or rotation is None:
                raise RuntimeError(f"{name} missing GPS or rotation sensor payload")
            x, y = -float(gps[1]), float(gps[0])
            heading = math.pi / 2 + math.radians(float(rotation[2]))
            positions[name] = (x, y, heading)
            truth[name] = Truth(x, y, heading)
        for name in NAMES:
            x, y, heading = positions[name]
            prior = self.previous_positions.get(name)
            if initial or prior is None:
                surge, yaw = 0., 0.
            else:
                dx, dy = x - prior[0], y - prior[1]
                surge = (dx * math.cos(heading) + dy * math.sin(heading)) / self.config.dt_s
                delta = math.atan2(math.sin(heading - self.previous_headings[name]),
                                   math.cos(heading - self.previous_headings[name]))
                yaw = delta / self.config.dt_s
            agents = tuple((ax, ay) for other, (ax, ay, _) in positions.items()
                           if other != name and math.dist((x, y), (ax, ay)) <= self.config.visibility_m)
            obstacles = tuple((o.x_m, o.y_m, o.radius_m) for o in self.scenario.obstacles
                              if math.dist((x, y), (o.x_m, o.y_m)) <= self.config.visibility_m)
            readings[name] = VesselReading(x, y, heading, surge, yaw, agents, obstacles)
            self.previous_positions[name] = (x, y)
            self.previous_headings[name] = heading
        return readings, truth

    def step(self, actions):
        if self.env is None:
            raise RuntimeError("Reset required")
        for name in NAMES:
            speed, yaw = actions[name]
            measured = self.latest_readings[name] if hasattr(self, "latest_readings") else None
            target_speed = max(0., min(1., speed)) * self.config.max_surge_mps
            # Native force control in newtons; right-minus-left gives positive yaw.
            base = 200. * (target_speed - (measured.surge_mps if measured else 0.))
            differential = 100. * max(-1., min(1., yaw))
            self.env.act(name, [max(-500., min(500., base - differential)),
                                max(-500., min(500., base + differential))])
        state = self.env.tick(num_ticks=round(self.config.dt_s * 50))
        result = self._read(state)
        self.latest_readings = result[0]
        return result

    def close(self):
        try:
            if self.env is not None:
                close = getattr(self.env, "close", None)
                if callable(close):
                    close()
                else:
                    # HoloOcean 2.3 provides context-manager teardown, not close().
                    exit_method = getattr(self.env, "__exit__", None)
                    if callable(exit_method):
                        exit_method(None, None, None)
        finally:
            self.env = None
            if hasattr(self, "latest_readings"):
                del self.latest_readings
