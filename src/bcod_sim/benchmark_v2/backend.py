"""Measured-reading versus scoring-truth backend contract."""
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from .scenarios import Scenario


@dataclass(frozen=True)
class VesselReading:
    x_m: float
    y_m: float
    heading_rad: float
    surge_mps: float
    yaw_rps: float
    nearby_agents: tuple[tuple[float, float], ...] = ()
    nearby_obstacles: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class Truth:
    x_m: float
    y_m: float
    heading_rad: float


@dataclass(frozen=True)
class BackendFrame:
    readings: Mapping[str, VesselReading]
    truth: Mapping[str, Truth]
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class Backend(Protocol):
    def reset(self, scenario: Scenario, seed: int) -> BackendFrame: ...
    def step(self, physical_commands: Mapping[str, tuple[float, float]], dt_s: float) -> BackendFrame: ...
    def close(self) -> None: ...
