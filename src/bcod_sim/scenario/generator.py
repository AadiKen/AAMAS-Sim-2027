"""Reproducible scenario composition from resolved config and template."""

from dataclasses import dataclass
import math
import random
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bcod_sim.config.hashing import content_hash
from bcod_sim.config.resolver import ResolvedExperiment
from bcod_sim.core.errors import ConfigSchemaError
from bcod_sim.scenario.distributions import Distribution, draw


SPAWN_FIELDS = ("north_m", "east_m", "down_m", "roll_rad", "pitch_rad", "yaw_rad",
                "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps")


class ScenarioTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    spawn_overrides: dict[str, dict[str, Distribution]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known_fields(self):
        for vessel_id, fields in self.spawn_overrides.items():
            if not vessel_id or set(fields) - set(SPAWN_FIELDS):
                raise ValueError("Scenario spawn override has unknown vessel or field")
        return self


@dataclass(frozen=True)
class SpawnState:
    instance_id: str
    position_ned_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]
    nu_body: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ResolvedScenario:
    config_hash: str
    template_hash: str
    episode_seed: int
    episode_index: int
    max_master_steps: int
    spawns: tuple[SpawnState, ...]
    task_reference: str
    content_hash: str

    def payload(self) -> dict:
        return {"config_hash": self.config_hash, "template_hash": self.template_hash,
                "episode_seed": self.episode_seed, "episode_index": self.episode_index,
                "max_master_steps": self.max_master_steps,
                "spawns": [vars(spawn) for spawn in self.spawns], "task_reference": self.task_reference}


def generate(resolved: ResolvedExperiment, template: ScenarioTemplate, *, seed: int,
             episode_index: int = 0) -> ResolvedScenario:
    if seed < 0 or episode_index < 0:
        raise ConfigSchemaError("Scenario seed and episode index must be nonnegative")
    known_ids = {vessel.instance_id for vessel in resolved.config.vessels}
    if set(template.spawn_overrides) - known_ids:
        raise ConfigSchemaError("Scenario template refers to unknown vessel")
    template_payload = template.model_dump(mode="json")
    template_hash = content_hash(template_payload)
    rng_seed = int(content_hash({"seed": seed, "episode_index": episode_index,
                                 "template_hash": template_hash})[:16], 16)
    rng = random.Random(rng_seed)
    spawns = []
    for vessel in resolved.config.vessels:
        fields = dict(zip(SPAWN_FIELDS, (*vessel.spawn.ned_m, *vessel.spawn.rpy_rad, 0, 0, 0, 0, 0, 0)))
        for name, distribution in sorted(template.spawn_overrides.get(vessel.instance_id, {}).items()):
            fields[name] = draw(distribution, rng)
        if not all(math.isfinite(value) for value in fields.values()):
            raise ConfigSchemaError("Scenario generation produced nonfinite spawn")
        spawns.append(SpawnState(vessel.instance_id,
                                 tuple(fields[name] for name in SPAWN_FIELDS[:3]),
                                 tuple(fields[name] for name in SPAWN_FIELDS[3:6]),
                                 tuple(fields[name] for name in SPAWN_FIELDS[6:])))
    scenario = ResolvedScenario(resolved.content_hash, template_hash, seed, episode_index,
                                resolved.config.simulation.max_master_steps, tuple(spawns),
                                resolved.config.task.type, "")
    return ResolvedScenario(scenario.config_hash, scenario.template_hash, scenario.episode_seed,
                            scenario.episode_index, scenario.max_master_steps, scenario.spawns,
                            scenario.task_reference, content_hash(scenario.payload()))
