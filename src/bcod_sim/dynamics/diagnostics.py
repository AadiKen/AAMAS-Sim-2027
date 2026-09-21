"""Inspectable body-FRD wrench ledger."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from bcod_sim.core.errors import PhysicalValidationError


EXTERNAL_TERMS = ("propulsion", "current", "wind", "wave", "wake", "contact", "manual")
INTERNAL_TERMS = ("rigid_coriolis", "added_mass_coriolis", "linear_damping", "nonlinear_damping", "restoring", "crossflow")


@dataclass(frozen=True)
class WrenchLedger:
    terms: Mapping[str, torch.Tensor]
    total: torch.Tensor
    component_diagnostics: Mapping[str, object] = MappingProxyType({})

    def assert_balanced(self, *, atol: float = 1e-9) -> None:
        summed = torch.stack(tuple(self.terms.values())).sum(dim=0)
        if not torch.allclose(summed, self.total, rtol=1e-9, atol=atol):
            raise PhysicalValidationError("Wrench contributions do not sum to applied total")


@dataclass(frozen=True)
class QuaternionNormalizationDiagnostic:
    kind: str
    norm_error: float
    threshold: float


def make_ledger(terms: Mapping[str, torch.Tensor], component_diagnostics: Mapping[str, object] | None = None) -> WrenchLedger:
    expected = set(EXTERNAL_TERMS + INTERNAL_TERMS)
    if set(terms) != expected:
        raise PhysicalValidationError("Wrench ledger has missing or unknown terms")
    detached = {name: value.clone() for name, value in terms.items()}
    total = torch.stack(tuple(detached.values())).sum(dim=0)
    ledger = WrenchLedger(MappingProxyType(detached), total, MappingProxyType(dict(component_diagnostics or {})))
    ledger.assert_balanced()
    return ledger
