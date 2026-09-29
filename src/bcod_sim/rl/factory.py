"""Convenience Gymnasium and PettingZoo factories over the existing core adapter."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from bcod_sim.actuators.autopilot import HighLevelCommand
from bcod_sim.config.resolver import ResolvedExperiment, load_config, resolve
from bcod_sim.config.registry import Registry
from bcod_sim.core.engine import EpisodeEngine
from bcod_sim.core.errors import ConfigSchemaError, PhysicalValidationError
from bcod_sim.core.lifecycle import DirectAction, PhysicalAction
from bcod_sim.rl.centralized_state import CentralizedStateContract
from bcod_sim.rl.observation import ObservationContract
from bcod_sim.rl.observation import ObservationField
from bcod_sim.rl.pettingzoo_env import PettingZooParallelEnv
from bcod_sim.rl.profile import RLProfile
from bcod_sim.web.runtime_factory import build_engine
from bcod_sim.sensors.base import SensorOutputField, SensorOutputSchema


@dataclass(frozen=True)
class ExperimentBundle:
    config: Mapping[str, object]
    definitions: tuple[Mapping[str, object], ...]
    rl: Mapping[str, object]
    bundle_version: int = 1

    @classmethod
    def load(cls, path):
        raw = load_config(path)
        if raw.get("bundle_version") != 1 or not isinstance(raw.get("config"), dict):
            raise ConfigSchemaError("Expected ExperimentBundle with bundle_version: 1 and config")
        definitions = raw.get("definitions")
        rl = raw.get("rl")
        if not isinstance(definitions, list) or not isinstance(rl, dict):
            raise ConfigSchemaError("ExperimentBundle requires definitions list and rl mapping")
        if set(raw) != {"bundle_version", "config", "definitions", "rl"}:
            raise ConfigSchemaError("ExperimentBundle has unknown or missing top-level fields")
        return cls(raw["config"], tuple(definitions), rl)


def _schema_for(sensor_type, parameters):
    # These are static declarations, matching built-in packet output metadata.
    specs = {
        "gps": (("wgs84_lat_lon_alt", (3,), "float64", "rad,rad,m", "WGS84"),
                ("velocity_ned_mps", (3,), "float64", "m/s", "NED")),
        "imu": (("specific_force_mount_mps2", (3,), "float64", "m/s^2", "mount"),
                ("angular_rate_mount_radps", (3,), "float64", "rad/s", "mount")),
        "ground_truth_state": (("position_ned_m", (3,), "float64", "m", "NED"),
                ("q_body_to_ned", (4,), "float64", "unitless", "FRD->NED"),
                ("nu_body", (6,), "float64", "m/s,rad/s", "FRD")),
    }
    if sensor_type in {"lidar", "sonar"}:
        key, count = ("range_m", int(parameters.get("ray_count" if sensor_type == "lidar" else "beam_count", 0)))
        if count <= 0:
            raise ConfigSchemaError(f"Sensor {sensor_type} requires declared beam count for a fixed output schema")
        specs[sensor_type] = ((key, (count,), "float64", "m", "mount"),
                              ("hit", (count,), "bool", "bool", "mount"))
    if sensor_type not in specs:
        raise ConfigSchemaError(f"Sensor type {sensor_type!r} has variable or unsupported output; supply explicit encoder/profile")
    return SensorOutputSchema(tuple(SensorOutputField(key, dtype, shape, units=units, frame=frame)
                                    for key, shape, dtype, units, frame in specs[sensor_type]))


def _contracts_for(config, registry, rl):
    observation = rl.get("observation", "sensor_dict")
    allow_truth = bool(rl.get("allow_ground_truth", False))
    contracts = {}
    for vessel in config.vessels:
        if vessel.controller.mode == "scripted":
            continue
        fields = []
        requested = rl.get("fields")
        for reference in vessel.sensors:
            definition = registry.resolve("sensor", reference)
            sensor_type = str(definition.payload.get("kind", definition.id))
            schema = _schema_for(sensor_type, dict(definition.payload))
            expected_numeric = "float32" if config.experiment.numerical_profile == "training" else "float64"
            for output in schema.fields:
                if requested and output.name not in requested:
                    continue
                label = output.name if len(vessel.sensors) == 1 else f"{definition.id}.{output.name}"
                dtype = expected_numeric if output.dtype.startswith("float") else output.dtype
                fields.append(ObservationField(label, definition.id, sensor_type,
                    definition.version, definition.source,
                    "ground_truth" if sensor_type == "ground_truth_state" else
                    "abstract" if sensor_type == "abstract_entities" else "physical",
                    output.name, output.shape, dtype, output.units, output.frame,
                    (rl.get("normalization", {}) or {}).get(label, "none")))
        if not fields:
            raise ConfigSchemaError(f"Agent {vessel.instance_id!r} has no supported declared sensor fields")
        contracts[vessel.instance_id] = ObservationContract(tuple(fields), allow_ground_truth=allow_truth)
    return contracts


def _experiment(source):
    if isinstance(source, ResolvedExperiment):
        return source, {}, None
    if isinstance(source, EpisodeEngine):
        return source, {}, None
    path = Path(source)
    bundle = ExperimentBundle.load(path)
    registry = Registry()
    for item in bundle.definitions:
        registry.register(**item)
    resolved = resolve(bundle.config, registry)
    return resolved, bundle.rl, registry


def _engine(source, profile: RLProfile | None = None):
    value, rl, registry = _experiment(source)
    if isinstance(value, EpisodeEngine):
        return value, rl
    contracts = (profile.observation_contracts if profile is not None and profile.observation_contracts is not None
                 else _contracts_for(value.config, registry, rl) if registry is not None else None)
    return build_engine(value, observation_contracts=contracts), rl


def _profile(engine, observation, action, central_state):
    try:
        from gymnasium.spaces import Box, Dict as DictSpace
    except ImportError as exc:
        raise ImportError("Install bcod-sim[test] or gymnasium to use the RL factories") from exc
    agents = sorted(name for name, vessel in engine.config_vessels.items()
                    if vessel.controller.mode != "scripted")
    contracts = engine.observation_contracts
    obs_spaces, obs_encoders = {}, {}
    metadata = {}
    for name in agents:
        contract = contracts.get(name)
        if contract is None:
            raise ConfigSchemaError(
                f"Agent {name!r} has no ObservationContract. Configure one in the experiment or supply RLProfile; "
                "the factory will not infer fields by silently exposing uncontracted sensor values.")
        fields = contract.fields
        if observation == "sensor_dict":
            spaces = {}
            for field in fields:
                if None in field.shape or field.dtype == "detections" or field.dtype == "record":
                    raise ConfigSchemaError(f"Field {field.name!r} is variable-length or structured; declare a custom RLProfile")
                dtype = np.bool_ if field.dtype == "bool" else np.float64 if field.dtype == "float64" else np.float32
                shape = field.shape
                spaces[field.name] = Box(False, True, shape=shape, dtype=dtype) if dtype == np.bool_ else Box(-np.inf, np.inf, shape=shape, dtype=dtype)
            obs_spaces[name] = DictSpace(spaces)
            def encode(raw, _fields=fields):
                result = {}
                for field in _fields:
                    value = raw[field.name]
                    result[field.name] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
                return result
            obs_encoders[name] = encode
        elif observation == "flat_sensors":
            if any(None in f.shape or f.dtype in {"detections", "record", "bool"} for f in fields):
                raise ConfigSchemaError("flat_sensors requires fixed-size numeric fields; provide a custom profile")
            sizes = [math.prod(f.shape) if f.shape else 1 for f in fields]
            order = tuple(f.name for f in fields)
            size = sum(sizes)
            obs_spaces[name] = Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
            def encode(raw, _order=order):
                result = np.concatenate([np.asarray(raw[key].detach().cpu() if hasattr(raw[key], "detach") else raw[key], dtype=np.float32).reshape(-1) for key in _order])
                return result
            obs_encoders[name] = encode
            metadata[name] = {"field_order": order, "units_frames": tuple((f.units, f.frame) for f in fields),
                              "normalization": tuple(f.normalization for f in fields), "shape": (size,),
                              "schema_hash": contract.content_hash}
        else:
            raise ConfigSchemaError("observation must be 'sensor_dict' or 'flat_sensors'")
    action_spaces, decoders = {}, {}
    action_cfg = action or {}
    for name in agents:
        vessel = engine.config_vessels[name]
        mode = action_cfg.get("mode", vessel.controller.mode)
        if mode == "high_level":
            speed = action_cfg.get("speed_range_mps")
            heading = action_cfg.get("heading", {})
            if action_cfg.get("normalized", True) is not True:
                raise ConfigSchemaError("Only normalized high_level policy actions are supported")
            if not speed or len(speed) != 2 or speed[0] >= speed[1]:
                raise ConfigSchemaError("high_level action requires explicit speed_range_mps [min,max]")
            if heading.get("mode") != "relative" or heading.get("max_offset_rad", 0) <= 0:
                raise ConfigSchemaError("high_level action requires heading.mode=relative and max_offset_rad")
            maximum = float(heading["max_offset_rad"])
            action_spaces[name] = Box(-1., 1., shape=(2,), dtype=np.float32)
            def decode(value, lo=float(speed[0]), hi=float(speed[1]), maxoff=maximum, engine=engine, agent=name):
                import torch
                q = engine.states[agent].q_body_to_ned
                w, x, y, z = (float(part) for part in q)
                heading_now = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                return HighLevelCommand(lo + (float(value[0]) + 1.) * .5 * (hi - lo),
                                        heading_now + float(value[1]) * maxoff)
            decoders[name] = decode
        elif mode == "direct_actuator":
            pipeline = engine.vessels[name].physical_pipeline
            if pipeline is not None:
                devices = pipeline.actuators.devices
                if not devices:
                    raise ConfigSchemaError(f"No declared actuators for {name}")
                lows = [float(device.config.thrust_bounds_n.minimum) for device in devices]
                highs = [float(device.config.thrust_bounds_n.maximum) for device in devices]
                ids = [device.id for device in devices]
                action_spaces[name] = Box(np.asarray(lows, np.float32), np.asarray(highs, np.float32), dtype=np.float32)
                decoders[name] = lambda value, ids=ids: PhysicalAction(tuple((key, tuple([float(v)])) for key, v in zip(ids, value)))
            else:
                actuators = engine.vessels[name].actuators
                if not actuators:
                    raise ConfigSchemaError(f"No declared actuators for {name}")
                ids = [x.config.instance_id for x in actuators]
                lows = [float(x.config.thrust_bounds_n.minimum) for x in actuators]
                highs = [float(x.config.thrust_bounds_n.maximum) for x in actuators]
                action_spaces[name] = Box(np.asarray(lows, np.float32), np.asarray(highs, np.float32), dtype=np.float32)
                from bcod_sim.actuators.thruster import ThrustCommand
                decoders[name] = lambda value, ids=ids: DirectAction(tuple((key, ThrustCommand(float(v))) for key, v in zip(ids, value)))
        elif mode == "desired_wrench":
            raise ConfigSchemaError(f"desired_wrench is not supported by runtime controller for {name}")
        else:
            raise ConfigSchemaError(f"Unsupported RL action mode: {mode}")
    central = CentralizedStateContract(tuple(central_state)) if central_state else None
    return RLProfile(obs_spaces, action_spaces, obs_encoders, decoders, central, metadata)


try:
    import gymnasium as _gymnasium
    _GymBase = _gymnasium.Env
except ImportError:
    _GymBase = object


class GymFromParallel(_GymBase):
    """Single-agent Gymnasium facade over PettingZooParallelEnv lifecycle."""
    metadata = {"render_modes": []}
    def __init__(self, parallel):
        super().__init__()
        self._env = parallel
        self.agent = parallel.possible_agents[0]
        self.observation_space = parallel.observation_space(self.agent)
        self.action_space = parallel.action_space(self.agent)
        self.np_random = None
        self.rl_profile = getattr(parallel, "rl_profile", None)
    @property
    def unwrapped(self): return self
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, infos = self._env.reset(seed=seed, options=options)
        return obs[self.agent], infos[self.agent]
    def step(self, action):
        obs, rewards, terms, truncs, infos = self._env.step({self.agent: action})
        return obs[self.agent], rewards[self.agent], terms[self.agent], truncs[self.agent], infos[self.agent]
    def close(self): self._env.close()


def make_parallel_env(source, *, profile: RLProfile | None = None, observation=None,
                      action: Mapping | None = None, centralized_state=None):
    engine, rl = _engine(source, profile)
    if not any(v.controller.mode != "scripted" for v in engine.config_vessels.values()):
        raise ConfigSchemaError("PettingZoo requires at least one RL-controlled agent")
    policy_action = action or rl.get("action", {})
    if profile is None and policy_action.get("mode") == "high_level":
        profile = _profile(engine, observation or rl.get("observation", "sensor_dict"),
                           policy_action, centralized_state or rl.get("centralized_state"))
    profile = profile or _profile(engine, observation or rl.get("observation", "sensor_dict"),
                                   policy_action,
                                   centralized_state or rl.get("centralized_state"))
    env = PettingZooParallelEnv(engine, observation_spaces=profile.observation_spaces,
        action_spaces=profile.action_spaces, centralized_state=profile.centralized_state,
        observation_encoders=profile.observation_encoders, action_decoders=profile.action_decoders)
    env.rl_profile = profile
    return env


def make_gym_env(source, **kwargs):
    profile = kwargs.pop("profile", None)
    engine, rl = _engine(source, profile)
    n = sum(v.controller.mode != "scripted" for v in engine.config_vessels.values())
    if n != 1:
        raise ConfigSchemaError(f"Gymnasium requires exactly one RL-controlled agent; found {n}")
    return GymFromParallel(make_parallel_env(engine, profile=profile, action=kwargs.pop("action", rl.get("action", {})),
                                             observation=kwargs.pop("observation", rl.get("observation", "sensor_dict")),
                                             centralized_state=kwargs.pop("centralized_state", rl.get("centralized_state")),
                                             **kwargs))


def make_env(source, *, api="auto", **kwargs):
    profile = kwargs.get("profile")
    engine, rl = _engine(source, profile)
    count = sum(v.controller.mode != "scripted" for v in engine.config_vessels.values())
    if count == 0:
        raise ConfigSchemaError("make_env requires at least one RL-controlled agent")
    if api not in {"auto", "gymnasium", "pettingzoo"}:
        raise ValueError("api must be 'auto', 'gymnasium', or 'pettingzoo'")
    selected = ("gymnasium" if count == 1 else "pettingzoo") if api == "auto" else api
    kwargs.setdefault("action", rl.get("action", {}))
    kwargs.setdefault("observation", rl.get("observation", "sensor_dict"))
    kwargs.setdefault("centralized_state", rl.get("centralized_state"))
    if selected == "gymnasium": return GymFromParallel(make_parallel_env(engine, **kwargs))
    return make_parallel_env(engine, **kwargs)
