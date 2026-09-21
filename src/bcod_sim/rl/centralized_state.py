"""Explicit centralized training view, separate from agent observations."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import torch

from bcod_sim.core.engine import EpisodeEngine
from bcod_sim.core.errors import PhysicalValidationError


StateField = Literal["position_ned", "q_body_to_ned", "nu_body", "rl_active", "physical_active"]


@dataclass(frozen=True)
class CentralizedStateContract:
    fields: tuple[StateField, ...]

    def __post_init__(self) -> None:
        if not self.fields or len(self.fields) != len(set(self.fields)):
            raise PhysicalValidationError("Centralized state fields must be nonempty and unique")

    def assemble(self, engine: EpisodeEngine) -> Mapping[str, torch.Tensor]:
        if engine.scenario is None:
            raise PhysicalValidationError("Centralized state requires an active episode")
        names = tuple(sorted(engine.vessels, key=lambda name: engine.vessels[name].vessel_id))
        device = engine.states[names[0]].nu_body.device
        result = {}
        for field in self.fields:
            if field in ("rl_active", "physical_active"):
                result[field] = torch.tensor([getattr(engine.statuses[name], field) for name in names],
                                             dtype=torch.bool, device=device)
            else:
                result[field] = torch.stack([getattr(engine.states[name], field).clone() for name in names])
        result["vessel_id"] = torch.tensor([engine.vessels[name].vessel_id for name in names],
                                           dtype=torch.int64, device=device)
        return MappingProxyType(result)
