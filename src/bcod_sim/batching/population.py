"""Stable flattened environment and vessel populations without padding."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from bcod_sim.core.engine import EpisodeEngine
from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError
from bcod_sim.state.tables import VesselTable


@dataclass(frozen=True)
class EnvironmentRow:
    env_id: int
    experiment_id: str
    config_hash: str
    master_step: int


@dataclass(frozen=True)
class Population:
    environments: tuple[EnvironmentRow, ...]
    vessels: VesselTable
    vessel_names: tuple[str, ...]
    row_by_identity: Mapping[tuple[int, int], int]


def flatten_population(engines: Mapping[int, EpisodeEngine]) -> Population:
    if any(not isinstance(env_id, int) or env_id < 0 for env_id in engines):
        raise PhysicalValidationError("Environment IDs must be nonnegative integers")
    environment_rows, rows = [], []
    for env_id, engine in sorted(engines.items()):
        if engine.scenario is None:
            raise PhysicalValidationError("Every batched environment must be reset")
        environment_rows.append(EnvironmentRow(env_id, engine.resolved.config.experiment.id,
                                                engine.resolved.content_hash, engine.master_step))
        for name, vessel in sorted(engine.vessels.items(), key=lambda item: item[1].vessel_id):
            state, status = engine.states[name], engine.statuses[name]
            rows.append((env_id, vessel.vessel_id, name, state, status))
    identities = [(row[0], row[1]) for row in rows]
    if len(identities) != len(set(identities)):
        raise DuplicateIdentityError("Duplicate flattened vessel identity")
    if rows:
        dtype, device = rows[0][3].nu_body.dtype, rows[0][3].nu_body.device
        if any(row[3].nu_body.dtype != dtype or row[3].nu_body.device != device for row in rows):
            raise PhysicalValidationError("One population table requires a common tensor dtype and device")
        table = VesselTable(
            torch.tensor([r[0] for r in rows], dtype=torch.int64, device=device),
            torch.tensor([r[1] for r in rows], dtype=torch.int64, device=device),
            torch.stack([r[3].position_ned for r in rows]), torch.stack([r[3].q_body_to_ned for r in rows]),
            torch.stack([r[3].nu_body for r in rows]),
            torch.tensor([r[4].rl_active for r in rows], dtype=torch.bool, device=device),
            torch.tensor([r[4].physical_active for r in rows], dtype=torch.bool, device=device),
            tuple(None for _ in rows))
    else:
        table = VesselTable(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64),
            torch.empty((0, 3)), torch.empty((0, 4)), torch.empty((0, 6)),
            torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool), ())
    index = MappingProxyType({identity: i for i, identity in enumerate(identities)})
    return Population(tuple(environment_rows), table, tuple(r[2] for r in rows), index)

