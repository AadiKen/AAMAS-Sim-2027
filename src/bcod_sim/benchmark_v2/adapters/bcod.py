"""BCOD EpisodeEngine bridge for the portable navigation benchmark."""

import math
import json
import hashlib
from pathlib import Path

import torch

from bcod_sim.actuators.allocation import allocate_fixed_thrusters
from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.core.lifecycle import DirectAction
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
from bcod_sim.frames.tensor import rotate_body_to_world
from bcod_sim.sensors.base import SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.abstract import AbstractEntitySensor

from ..config import TaskConfig
from ..scenarios import Scenario
from ..backend import BackendFrame, Truth, VesselReading
from bcod_sim.benchmark.control import RatePI, integrate_gyro, common_heading_rate


OTTER_PARAMETER_SHA256 = "ee616689f78eaba6803e5571a505ebbe1fd3175fde85ae5372ef97eb37301718"


def _otter_parameters():
    path = Path(__file__).resolve().parents[4] / "artifacts/mss-6dof-validation/latest/plant/bcod_otter_parameters.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OTTER_PARAMETER_SHA256:
        raise RuntimeError("Validated MSS/Otter parameter artifact changed")
    return json.loads(raw)


class BCODBackend:
    """Validated MSS/Otter plant with BCOD GPS, IMU, and entity detections."""

    def __init__(self, config: TaskConfig = TaskConfig()):
        self.config = config
        self.reduced_fidelity = True
        self.names = ()
        self.engine = None

    def _build(self, scenario: Scenario):
        cfg = self.config
        registry = Registry()
        registry.register("vessel", "benchmark_vessel", "1", {}, "benchmark")
        registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": scenario.agent_ids[0],
                          "target_ned_m": (1000., 1000., 0.), "radius_m": 0.1}, "benchmark")
        static = [{"id": f"obstacle_{i}", "position_ned_m": [o.y_m, o.x_m, 0.],
                   "shape": {"kind": "sphere", "radius_m": o.radius_m}}
                  for i, o in enumerate(scenario.obstacles)]
        raw = {"schema_version": 1, "experiment": {"id": "benchmark", "seed": scenario.seed},
               "simulation": {"dynamics_mode": "planar3" if self.reduced_fidelity else "full6", "master_dt_s": cfg.dt_s,
                              "dynamics_substeps": 2, "policy_every_n_master_steps": 1,
                              "max_master_steps": cfg.deadline_steps + 1},
               "world": {"source": {"kind": "parametric"},
                         "environment": {"current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                                         "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                                         "waves": {"kind": "calm"}, "visibility_m": cfg.visibility_m},
                         "bathymetry": {"kind": "flat", "bottom_ned_z_m": 100., "vertical_datum": "MSL"},
                         "static_entities": static},
               "vessels": [{"instance_id": name, "definition": "benchmark_vessel@1",
                            "controller": {"mode": "direct_actuator"},
                            "spawn": {"ned_m": [scenario.starts[i].y_m, scenario.starts[i].x_m, 0.],
                                      "rpy_rad": [0., 0., math.pi / 2 - scenario.starts[i].heading_rad]}}
                           for i, name in enumerate(self.names)],
               "task": {"type": "waypoint@1", "reward": {"individual_weight": 0., "team_weight": 0.},
                        "disabled_agent_behavior": "deactivate_keep_physical"}}
        resolved = resolve(raw, registry)
        tensor = lambda value: torch.tensor(value, dtype=torch.float64)
        params = _otter_parameters()
        vessels = []
        for i, name in enumerate(self.names):
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
                                   max_range_m=cfg.visibility_m + 1e-8, horizontal_fov_rad=2 * math.pi))
            vessels.append(EpisodeVessel(name, vessel_id, plant, Sphere(cfg.vessel_radius_m),
                thrusters, sensors, ExplicitZeroLoads()))
        return EpisodeEngine(resolved, tuple(vessels))

    def reset(self, scenario: Scenario, seed: int):
        if seed != scenario.seed:
            raise ValueError("Scenario seed and reset seed must match")
        self.scenario = scenario
        self.names = scenario.agent_ids
        if self.engine is not None:
            self.close()
        self.engine = self._build(scenario)
        if set(self.engine.vessels) != set(self.names) or len(self.engine.vessels) != len(self.names):
            raise AssertionError("BCOD engine vessel count differs from scenario")
        frame = self.engine.reset(seed=seed)
        self.orientation = {n: frame.states[n].q_body_to_ned.clone() for n in self.names}
        self.previous_omega = {n: frame.states[n].nu_body[3:].clone() for n in self.names}
        # Pole placement around the nominal mass/damping model: wn=1 rad/s,
        # zeta=1. Feedforward cancels nominal drag; feedback rejects coupling.
        self.controllers = {}
        for n in self.names:
            mass = self.engine.vessels[n].plant.mass.matrices()[1]
            self.controllers[n] = (RatePI(2 * float(mass[0, 0]), float(mass[0, 0])),
                                   RatePI(2 * float(mass[5, 5]), float(mass[5, 5])))
        self.diagnostics = {"sim_time_s": 0., "saturated": {}, "native_contacts": []}
        readings, truth = self._read(frame)
        return BackendFrame(readings, truth, dict(self.diagnostics))

    def _read(self, frame):
        truth = {}
        readings = {}
        for i, name in enumerate(self.names):
            state = frame.states[name]
            x, y = float(state.position_ned[1]), float(state.position_ned[0])
            yaw_ned = math.atan2(2 * (float(state.q_body_to_ned[0]) * float(state.q_body_to_ned[3]) +
                                      float(state.q_body_to_ned[1]) * float(state.q_body_to_ned[2])),
                                 1 - 2 * (float(state.q_body_to_ned[2]) ** 2 + float(state.q_body_to_ned[3]) ** 2))
            truth[name] = Truth(x, y, math.pi / 2 - yaw_ned)
        measured_positions = {}
        for i, name in enumerate(self.names):
            gps = self.engine.latest_packets[(i + 1, "gps")].values
            north, east, _ = geodetic_to_ned(*[float(v) for v in gps["wgs84_lat_lon_alt"]], (0., 0., 0.))
            measured_positions[name] = (east, north)
        for i, name in enumerate(self.names):
            gps = self.engine.latest_packets[(i + 1, "gps")].values
            imu = self.engine.latest_packets[(i + 1, "imu")].values
            east, north = measured_positions[name]
            omega = imu["angular_rate_mount_radps"]
            if frame.master_step:
                self.orientation[name] = integrate_gyro(self.orientation[name], self.previous_omega[name],
                                                       omega, self.config.dt_s)
            self.previous_omega[name] = omega.clone()
            heading, yaw_rate = common_heading_rate(self.orientation[name], omega)
            v = gps["velocity_ned_mps"]
            surge = float(v[1]) * math.cos(heading) + float(v[0]) * math.sin(heading)
            # Other vessels share their GPS positions as a measured broadcast.
            agents = tuple(position for other, position in measured_positions.items() if other != name and
                           math.dist((east, north), position) <= self.config.visibility_m)
            detections = self.engine.latest_packets[(i + 1, "entities")].values["detections"]
            obstacles = []
            for detection in detections:
                relative_ned = rotate_body_to_world(detection.relative_mount_m, self.orientation[name])
                ox = east + float(relative_ned[1])
                oy = north + float(relative_ned[0])
                obstacles.append((ox, oy, 0.0))  # radius is not a policy observation field
            readings[name] = VesselReading(east, north, heading, surge, yaw_rate, agents, obstacles)
        return readings, truth

    def step(self, physical_commands, dt_s):
        if abs(dt_s - self.config.dt_s) > 1e-12:
            raise ValueError("Backend time step differs from configuration")
        if self.engine is None:
            raise RuntimeError("Reset backend before stepping")
        if set(physical_commands) != set(self.names):
            raise ValueError("Commands must match scenario agent IDs")
        for target in physical_commands.values():
            if (len(target) != 2 or not all(math.isfinite(x) for x in target)
                    or not 0 <= target[0] <= self.config.max_surge_mps
                    or not -self.config.max_yaw_rps <= target[1] <= self.config.max_yaw_rps):
                raise ValueError("Physical speed/yaw target outside V2 bounds")
        commands = {}
        params = _otter_parameters()
        saturated = {}
        for name, (target_speed, target_yaw) in physical_commands.items():
            state = self.engine.states[name]
            _, actual_yaw = common_heading_rate(state.q_body_to_ned, state.nu_body[3:])
            speed_pi, yaw_pi = self.controllers[name]
            force = speed_pi.request(target_speed - float(state.nu_body[0]), self.config.dt_s,
                                     params["linear_damping"][0] * target_speed)
            moment_ccw = yaw_pi.request(target_yaw - actual_yaw, self.config.dt_s,
                params["linear_damping"][5] * target_yaw + params["quadratic_damping"][5] * abs(target_yaw) * target_yaw)
            thrusters = self.engine.vessels[name].actuators
            allocated = allocate_fixed_thrusters(thrusters, surge_n=force, yaw_nm=-moment_ccw)
            bounded = {}
            for thruster in thrusters:
                key = thruster.config.instance_id
                request = allocated[key].thrust_n
                limits = thruster.config.thrust_bounds_n
                bounded[key] = ThrustCommand(max(limits.minimum, min(limits.maximum, request)))
            saturated[name] = any(bounded[k].thrust_n != allocated[k].thrust_n for k in bounded)
            speed_pi.commit(saturated[name])
            yaw_pi.commit(saturated[name])
            commands[name] = DirectAction(tuple(bounded.items()))
        frame = self.engine.step(commands)
        self.diagnostics = {"sim_time_s": frame.sim_time_s, "saturated": saturated,
                            "native_contacts": [str(event) for event in frame.contact_events],
                            "native_terminated": bool(frame.terminated),
                            "native_termination_reason": str(frame.termination_reason) if frame.termination_reason else None}
        readings, truth = self._read(frame)
        return BackendFrame(readings, truth, dict(self.diagnostics))

    def close(self):
        self.engine = None
