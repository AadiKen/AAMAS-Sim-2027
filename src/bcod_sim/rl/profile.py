"""Composable RL-facing spaces and codecs for an EpisodeEngine."""
from dataclasses import dataclass
from typing import Callable, Mapping

from bcod_sim.rl.centralized_state import CentralizedStateContract
from bcod_sim.rl.observation import ObservationContract


@dataclass(frozen=True)
class RLProfile:
    observation_spaces: Mapping[str, object]
    action_spaces: Mapping[str, object]
    observation_encoders: Mapping[str, Callable]
    action_decoders: Mapping[str, Callable]
    centralized_state: CentralizedStateContract | None = None
    metadata: Mapping[str, object] | None = None
    action_config: Mapping[str, object] | None = None
    observation_contracts: Mapping[str, ObservationContract] | None = None
