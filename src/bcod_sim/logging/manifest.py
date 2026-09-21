"""Run manifest skeleton populated from resolved objects and observed runtime facts."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bcod_sim.config.resolver import ResolvedExperiment


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    started_utc: datetime
    ended_utc: datetime | None = None
    config_hash: str
    git_commit: str | None = None
    dirty_patch_hash: str | None = None
    dependency_lock_hash: str | None = None
    backend: str | None = None
    device: str | None = None
    dtype: str | None = None
    deterministic: bool
    master_dt_s: float
    dynamics_substeps: int
    policy_every_n_master_steps: int
    dynamics_mode: Literal["full6", "planar3"]
    definition_hashes: dict[str, str]
    asset_hashes: dict[str, str] = Field(default_factory=dict)
    external_data: tuple[dict[str, str], ...] = ()
    seed: int
    seed_streams: dict[str, int] = Field(default_factory=dict)
    policy_provenance: dict[str, str] | None = None
    trainer_provenance: dict[str, str] | None = None
    observation_contract_hash: str | None = None
    action_contract_hash: str | None = None
    termination_reason: str | None = None

    @model_validator(mode="after")
    def check_timestamps(self):
        if self.started_utc.tzinfo is None or (self.ended_utc and self.ended_utc.tzinfo is None):
            raise ValueError("Manifest timestamps must be timezone-aware")
        return self


def manifest_skeleton(resolved: ResolvedExperiment, *, run_id: str, started_utc: datetime | None = None) -> RunManifest:
    """Create pre-run record; execution fills observed backend and completion fields."""
    config = resolved.config
    return RunManifest(
        run_id=run_id,
        started_utc=started_utc or datetime.now(timezone.utc),
        config_hash=resolved.content_hash,
        deterministic=config.simulation.deterministic,
        master_dt_s=config.simulation.master_dt_s,
        dynamics_substeps=config.simulation.dynamics_substeps,
        policy_every_n_master_steps=config.simulation.policy_every_n_master_steps,
        dynamics_mode=config.simulation.dynamics_mode,
        definition_hashes={f"{d.kind}:{d.id}@{d.version}": d.content_hash for d in resolved.definitions},
        asset_hashes={f"{d.id}@{d.version}": d.content_hash for d in resolved.definitions if d.kind == "asset"},
        seed=config.experiment.seed,
    )
