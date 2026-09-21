"""Sensor identity, mount, timing, noise, provenance, and sample contracts."""

from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Mapping, Protocol

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


@dataclass(frozen=True)
class SensorContext:
    state: VesselState
    world: ParametricWorld
    sim_time_s: float
    linear_acceleration_body_mps2: torch.Tensor | None = None  # time derivative of body velocity
    angular_acceleration_body_radps2: torch.Tensor | None = None


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


class Sensor(Protocol):
    config: SensorConfig
    kind: SensorKind

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket: ...


def stable_noise(config: SensorConfig, sample_step: int, shape: tuple[int, ...], *, dtype: torch.dtype,
                 device: torch.device, stream: str = "default") -> torch.Tensor:
    if config.noise_std == 0:
        return torch.zeros(shape, dtype=dtype, device=device)
    key = f"{config.seed}|{config.env_id}|{config.owner_vessel_id}|{config.instance_id}|{sample_step}|{stream}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=torch.float64) * config.noise_std).to(dtype=dtype, device=device)


def packet(config: SensorConfig, kind: SensorKind, sample_step: int, values: Mapping[str, object],
           units: Mapping[str, str], frames: Mapping[str, str]) -> SensorPacket:
    return SensorPacket(config.instance_id, config.sensor_type, config.version, config.source,
                        kind, config.env_id, config.owner_vessel_id,
                        sample_step, sample_step + config.latency_steps, values, units, frames)
