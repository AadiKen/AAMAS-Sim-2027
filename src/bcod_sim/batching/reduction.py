"""Stable, exact-order segmented reduction for deterministic execution."""

from dataclasses import dataclass

import torch

from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError


@dataclass(frozen=True)
class Contribution:
    env_id: int
    owner_vessel_id: int
    component_id: str
    value: torch.Tensor


def deterministic_segmented_sum(rows: tuple[Contribution, ...]) -> dict[tuple[int, int], torch.Tensor]:
    keys = [(r.env_id, r.owner_vessel_id, r.component_id) for r in rows]
    if len(keys) != len(set(keys)):
        raise DuplicateIdentityError("Deterministic contribution identity must be unique")
    ordered = sorted(rows, key=lambda r: (r.env_id, r.owner_vessel_id, r.component_id))
    result: dict[tuple[int, int], torch.Tensor] = {}
    shape = ordered[0].value.shape if ordered else None
    dtype = ordered[0].value.dtype if ordered else None
    device = ordered[0].value.device if ordered else None
    for row in ordered:
        if row.env_id < 0 or row.owner_vessel_id < 0 or not row.component_id:
            raise PhysicalValidationError("Invalid contribution identity")
        if row.value.shape != shape or row.value.dtype != dtype or row.value.device != device:
            raise PhysicalValidationError("Segmented values require one shape, dtype, and device")
        if not torch.isfinite(row.value).all().item():
            raise PhysicalValidationError("Segmented values must be finite")
        owner = (row.env_id, row.owner_vessel_id)
        result[owner] = row.value.clone() if owner not in result else result[owner] + row.value
    return result

