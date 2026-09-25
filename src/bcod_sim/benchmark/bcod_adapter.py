"""BCOD EpisodeEngine bridge for the portable navigation benchmark."""

import math
import json
import hashlib
from pathlib import Path

import torch

from bcod_sim.actuators.autopilot import HeadingSpeedAutopilot, HighLevelCommand
from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster
from bcod_sim.collision.shapes import Sphere
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.crossflow import StripTheoryCrossflow
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import PlanarEquilibrium, Plant6
from bcod_sim.dynamics.restoring import LinearHydrostatics
from bcod_sim.frames.geodesy import geodetic_to_ned
from bcod_sim.sensors.base import SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.abstract import AbstractEntitySensor

from .core import BenchmarkConfig, NAMES, Scenario, Truth, VesselReading


OTTER_PARAMETER_SHA256 = "ee616689f78eaba6803e5571a505ebbe1fd3175fde85ae5372ef97eb37301718"


def _otter_parameters():
    path = Path(__file__).resolve().parents[3] / "artifacts/mss-6dof-validation/latest/plant/bcod_otter_parameters.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OTTER_PARAMETER_SHA256:
        raise RuntimeError("Validated MSS/Otter parameter artifact changed")
    return json.loads(raw)


class BCODAdapter:
    """Validated MSS/Otter plant with BCOD GPS, IMU, and entity detections."""

    def __init__(self, config: BenchmarkConfig = BenchmarkConfig(), *, reduced_fidelity: bool = False):
        self.config = config
        self.reduced_fidelity = reduced_fidelity
        self.engine = None

    def _build(self, scenario: Scenario):
        cfg = self.config
        registry = Registry()
        registry.register("vessel", "benchmark_vessel", "1", {}, "benchmark")
        registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": NAMES[0],
                          "target_ned_m": (1000., 1000., 0.), "radius_m": 0.1}, "benchmark")
        static = [{"id": f"obstacle_{i}", "position_ned_m": [o.y_m, o.x_m, 0.],
                   "shape": {"kind": "sphere", "radius_m": o.radius_m}}
                  for i, o in enumerate(scenario.obstacles)]
        raw = {"schema_version": 1, "experiment": {"id": "benchmark", "seed": scenario.seed},
               "simulation": {"dynamics_mode": "planar3" if self.reduced_fidelity else "full6", "master_dt_s": cfg.dt_s,
                              "dynamics_substeps": 2, "policy_every_n_master_steps": 1,
                              "max_master_steps": cfg.max_steps + 1},
               "world": {"source": {"kind": "parametric"},
                         "environment": {"current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                                         "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                                         "waves": {"kind": "calm"}, "visibility_m": cfg.visibility_m},
                         "bathymetry": {"kind": "flat", "bottom_ned_z_m": 100., "vertical_datum": "MSL"},
                         "static_entities": static},
               "vessels": [{"instance_id": name, "definition": "benchmark_vessel@1",
                            "controller": {"mode": "high_level"},
                            "spawn": {"ned_m": [scenario.starts[i].y_m, scenario.starts[i].x_m, 0.],
                                      "rpy_rad": [0., 0., math.pi / 2 - scenario.starts[i].heading_rad]}}
                           for i, name in enumerate(NAMES)],
               "task": {"type": "waypoint@1", "reward": {"individual_weight": 0., "team_weight": 0.},
                        "disabled_agent_behavior": "deactivate_keep_physical"}}
        resolved = resolve(raw, registry)
        tensor = lambda value: torch.tensor(value, dtype=torch.float64)
        params = _otter_parameters()
        vessels = []
        for i, name in enumerate(NAMES):
            vessel_id = i + 1
            hydro = params["hydrostatics"]
            cross = params["crossflow"]
            plant = Plant6(MassProperties(params["mass_kg"], tensor(params["cg_frd_m"]),
                tensor(params["inertia_cg_kg_m2"]), tensor(params["added_mass_kg"])),
                Damping(tensor(params["linear_damping"]), tensor(params["quadratic_damping"])),
                LinearHydrostatics(tensor(hydro["stiffness_6x6"]),
                    tensor(hydro["equilibrium_position_ned_m"]), tensor(hydro["equilibrium_rpy_rad"]), .5, .5),
                OperatingEnvelope(tensor(params["max_abs_nu"]), max_substep_s=0.1),
                mode="planar3" if self.reduced_fidelity else "full6",
                planar_equilibrium=PlanarEquilibrium(0., 0., 0.) if self.reduced_fidelity else None,
                crossflow=StripTheoryCrossflow.constant_section(cross["length_m"], cross["beam_m"],
                    cross["draft_m"], cross["strips"], water_density_kg_m3=params["water_density_kg_m3"],
                    include_vertical=cross["include_vertical"], dtype=torch.float64))
            propulsion = params["thrusters"]
            thrust_min = propulsion["k_negative"] * propulsion["shaft_speed_min_rad_s"] * abs(propulsion["shaft_speed_min_rad_s"])
            thrust_max = propulsion["k_positive"] * propulsion["shaft_speed_max_rad_s"] ** 2
            thrusters = tuple(FixedThruster(ActuatorConfig(f"thruster_{side}", 0, vessel_id,
                tuple(offset), (1, 0, 0, 0), Bounds(thrust_min, thrust_max),
                thrust_time_constant_s=propulsion["actuator_time_constant_s"]))
                for side, offset in zip(("left", "right"), propulsion["positions_frd_m"]))
            sensors = (GPS(SensorConfig("gps", "gps", 0, vessel_id, (0, 0, 0), (1, 0, 0, 0),
                                   1 / cfg.dt_s, 0, 0, scenario.seed + i, "1", "benchmark"),
                           origin_wgs84_rad_m=(0., 0., 0.)),
                       IMU(SensorConfig("imu", "imu", 0, vessel_id, (0, 0, 0), (1, 0, 0, 0),
                                   1 / cfg.dt_s, 0, 0, scenario.seed + i, "1", "benchmark")),
                       AbstractEntitySensor(SensorConfig("entities", "abstract_entities", 0, vessel_id,
                                   (0, 0, 0), (1, 0, 0, 0), 1 / cfg.dt_s, 0, 0,
                                   scenario.seed + i, "1", "benchmark"),
                                   max_range_m=cfg.visibility_m, horizontal_fov_rad=2 * math.pi))
            vessels.append(EpisodeVessel(name, vessel_id, plant, Sphere(cfg.vessel_radius_m),
                thrusters, sensors, ExplicitZeroLoads(), autopilot=HeadingSpeedAutopilot(100., 60.)))
        return EpisodeEngine(resolved, tuple(vessels))

    def reset(self, scenario: Scenario, seed: int):
        if seed != scenario.seed:
            raise ValueError("Scenario seed and reset seed must match")
        self.scenario = scenario
        self.engine = self._build(scenario)
        frame = self.engine.reset(seed=seed)
        self.estimated_heading = {n: scenario.starts[i].heading_rad for i, n in enumerate(NAMES)}
        return self._read(frame)

    def _read(self, frame):
        truth = {}
        readings = {}
        for i, name in enumerate(NAMES):
            state = frame.states[name]
            x, y = float(state.position_ned[1]), float(state.position_ned[0])
            yaw_ned = math.atan2(2 * (float(state.q_body_to_ned[0]) * float(state.q_body_to_ned[3]) +
                                      float(state.q_body_to_ned[1]) * float(state.q_body_to_ned[2])),
                                 1 - 2 * (float(state.q_body_to_ned[2]) ** 2 + float(state.q_body_to_ned[3]) ** 2))
            truth[name] = Truth(x, y, math.pi / 2 - yaw_ned)
        measured_positions = {}
        for i, name in enumerate(NAMES):
            gps = self.engine.latest_packets[(i + 1, "gps")].values
            north, east, _ = geodetic_to_ned(*[float(v) for v in gps["wgs84_lat_lon_alt"]], (0., 0., 0.))
            measured_positions[name] = (east, north)
        for i, name in enumerate(NAMES):
            gps = self.engine.latest_packets[(i + 1, "gps")].values
            imu = self.engine.latest_packets[(i + 1, "imu")].values
            east, north = measured_positions[name]
            yaw_rate = -float(imu["angular_rate_mount_radps"][2])
            if frame.master_step:
                self.estimated_heading[name] += yaw_rate * self.config.dt_s
            heading = self.estimated_heading[name]
            v = gps["velocity_ned_mps"]
            surge = float(v[1]) * math.cos(heading) + float(v[0]) * math.sin(heading)
            # Other vessels share their GPS positions as a measured broadcast.
            agents = tuple(position for other, position in measured_positions.items() if other != name and
                           math.dist((east, north), position) <= self.config.visibility_m)
            detections = self.engine.latest_packets[(i + 1, "entities")].values["detections"]
            obstacles = []
            for detection in detections:
                forward, right = (float(detection.relative_mount_m[0]), float(detection.relative_mount_m[1]))
                ox = east + forward * math.cos(heading) + right * math.sin(heading)
                oy = north + forward * math.sin(heading) - right * math.cos(heading)
                obstacles.append((ox, oy, 0.0))  # radius is not a policy observation field
            readings[name] = VesselReading(east, north, heading, surge, yaw_rate, agents, obstacles)
        return readings, truth

    def step(self, actions):
        commands = {}
        for name, (surge, yaw) in actions.items():
            state = self.engine.states[name]
            q = state.q_body_to_ned
            heading_ned = math.atan2(2 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                                     1 - 2 * (float(q[2]) ** 2 + float(q[3]) ** 2))
            commands[name] = HighLevelCommand(max(0., surge) * self.config.max_surge_mps,
                                               heading_ned - yaw * self.config.max_yaw_rps * self.config.dt_s)
        frame = self.engine.step(commands)
        return self._read(frame)

    def close(self):
        self.engine = None
