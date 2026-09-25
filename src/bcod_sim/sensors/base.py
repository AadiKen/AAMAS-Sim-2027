"""Sensor identity, mount, timing, noise, provenance, and sample contracts."""

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal, Mapping, Protocol

import torch

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.frames.transforms import normalize_quaternion
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.world.world import ParametricWorld

SensorKind = Literal["physical", "abstract", "ground_truth"]


@dataclass(frozen=True)
class SensorConfig:
    instance_id: str
    sensor_type: str
    env_id: int
    owner_vessel_id: int
    mount_frd_m: tuple[float, float, float]
    mount_q_to_frd: tuple[float, float, float, float]
    rate_hz: float
    latency_steps: int
    noise_std: float
    seed: int
    version: str
    source: str
    power_w: float | None = None
    data_rate_bps: float | None = None
    warmup_s: float = 0.0
    dropout_probability: float = 0.0
    config_fingerprint: str | None = None
    max_age_s: float | None = None

    def __post_init__(self) -> None:
        if not self.instance_id or not self.sensor_type or not self.version or not self.source:
            raise PhysicalValidationError("Sensor identity, type, version, and source are required")
        if self.env_id < 0 or self.owner_vessel_id < 0 or self.latency_steps < 0 or self.seed < 0:
            raise PhysicalValidationError("Sensor owner, latency, and seed must be nonnegative")
        if len(self.mount_frd_m) != 3 or not all(math.isfinite(x) for x in self.mount_frd_m):
            raise PhysicalValidationError("Sensor mount position must be finite FRD vector")
        q = normalize_quaternion(self.mount_q_to_frd)
        if any(abs(a-b) > 1e-8 for a, b in zip(q, self.mount_q_to_frd)):
            raise PhysicalValidationError("Sensor mount quaternion must be normalized")
        if not math.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise PhysicalValidationError("Sensor rate must be positive and finite")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise PhysicalValidationError("Sensor noise must be finite and nonnegative")
        optional = (self.power_w, self.data_rate_bps, self.max_age_s)
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in optional):
            raise PhysicalValidationError("Sensor power and data rate must be finite and nonnegative")
        if not math.isfinite(self.warmup_s) or self.warmup_s < 0 or not 0 <= self.dropout_probability <= 1:
            raise PhysicalValidationError("Invalid sensor warmup or dropout model")

    @property
    def fingerprint(self) -> str:
        if self.config_fingerprint is not None:
            return self.config_fingerprint
        from bcod_sim.config.hashing import content_hash
        return content_hash({key: value for key, value in vars(self).items() if key != "config_fingerprint"})


@dataclass(frozen=True)
class SensorContext:
    state: VesselState
    world: ParametricWorld
    sim_time_s: float
    linear_acceleration_body_mps2: torch.Tensor | None = None  # time derivative of body velocity
    angular_acceleration_body_radps2: torch.Tensor | None = None

    @property
    def environment(self):
        from bcod_sim.world.environment import Environment
        return Environment(self.world, state=self.state,
                           linear_acceleration_body_mps2=self.linear_acceleration_body_mps2)


@dataclass(frozen=True)
class SensorOutputField:
    name: str
    dtype: str
    shape: tuple[int | None, ...]
    variable_length: bool = False
    encoding: str = "inline"
    units: str = "unitless"
    frame: str = "scalar"


@dataclass(frozen=True)
class SensorOutputSchema:
    fields: tuple[SensorOutputField, ...]


@dataclass(frozen=True)
class SensorPacket:
    sensor_id: str
    sensor_type: str
    sensor_version: str
    sensor_source: str
    kind: SensorKind
    env_id: int
    owner_vessel_id: int
    sample_step: int
    delivery_step: int
    values: Mapping[str, object]
    units: Mapping[str, str]
    frames: Mapping[str, str]
    config_fingerprint: str = ""
    sample_time_s: float | None = None
    delivery_time_s: float | None = None
    schema: SensorOutputSchema | None = None
    validity: str = "valid"
    provenance: Mapping[str, Any] | None = None

    def age_s(self, current_time_s: float) -> float:
        if self.sample_time_s is None or not math.isfinite(current_time_s):
            raise PhysicalValidationError("Packet age requires finite sample and current times")
        age = current_time_s - self.sample_time_s
        if age < -1e-12:
            raise PhysicalValidationError("Current time precedes packet sample time")
        return max(0.0, age)

    def freshness(self, current_time_s: float, max_age_s: float | None = None) -> Mapping[str, object]:
        age = self.age_s(current_time_s)
        stale = max_age_s is not None and age > max_age_s
        return {"sample_time_s": self.sample_time_s, "delivery_time_s": self.delivery_time_s,
                "age_s": age, "validity": "stale" if stale else self.validity, "stale": stale}


class Sensor(Protocol):
    config: SensorConfig
    kind: SensorKind
    requirements: frozenset[str]

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket: ...


def stable_noise(config: SensorConfig, sample_step: int, shape: tuple[int, ...], *, dtype: torch.dtype,
                 device: torch.device, stream: str = "default", std: float | None = None) -> torch.Tensor:
    magnitude = config.noise_std if std is None else std
    if magnitude == 0:
        return torch.zeros(shape, dtype=dtype, device=device)
    key = f"{config.seed}|{config.env_id}|{config.owner_vessel_id}|{config.instance_id}|{sample_step}|{stream}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=torch.float64) * magnitude).to(dtype=dtype, device=device)


def packet(config: SensorConfig, kind: SensorKind, sample_step: int, values: Mapping[str, object],
           units: Mapping[str, str], frames: Mapping[str, str], *, sample_time_s: float | None = None,
           schema: SensorOutputSchema | None = None, validity: str = "valid") -> SensorPacket:
    if schema is None:
        schema = infer_output_schema(values, units, frames)
    return SensorPacket(config.instance_id, config.sensor_type, config.version, config.source,
                        kind, config.env_id, config.owner_vessel_id,
                        sample_step, sample_step + config.latency_steps, values, units, frames,
                        config.fingerprint, sample_time_s, None, schema, validity,
                        {"source": config.source, "version": config.version})


def infer_output_schema(values: Mapping[str, object], units: Mapping[str, str],
                        frames: Mapping[str, str]) -> SensorOutputSchema:
    fields = []
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            dtype, shape, variable, encoding = str(value.dtype).removeprefix("torch."), tuple(value.shape), False, ("array" if value.numel() > 4096 else "inline")
        elif isinstance(value, (tuple, list)):
            dtype, shape, variable, encoding = "structured", (None,), True, "json"
        elif isinstance(value, Mapping):
            dtype, shape, variable, encoding = "record", (), False, "json"
        elif isinstance(value, (str, bool, int, float)) or value is None:
            dtype, shape, variable, encoding = type(value).__name__, (), False, "inline"
        else:
            dtype, shape, variable, encoding = "structured", (), False, "json"
        fields.append(SensorOutputField(name, dtype, shape, variable, encoding,
                                        units.get(name, "unitless"), frames.get(name, "scalar")))
    return SensorOutputSchema(tuple(fields))
