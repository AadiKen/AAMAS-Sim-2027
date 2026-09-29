"""Deterministic, episode-local simulator message delivery."""

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

from bcod_sim.core.errors import PhysicalValidationError


@dataclass(frozen=True)
class CommunicationConfig:
    enabled: bool = False
    message_dim: int = 4
    latency_steps: int = 1
    max_range_m: float | None = None
    dropout_probability: float = 0.0

    def __post_init__(self):
        if self.message_dim < 1 or self.latency_steps < 1 or (self.max_range_m is not None and self.max_range_m <= 0) or not 0 <= self.dropout_probability <= 1:
            raise PhysicalValidationError("Invalid communication configuration")


@dataclass(frozen=True)
class CommunicatingAction:
    control: object
    message: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.message, tuple) or not all(math.isfinite(x) and -1 <= x <= 1 for x in self.message):
            raise PhysicalValidationError("Message must be an immutable finite vector in [-1,1]")


class MessageChannel:
    def __init__(self, config: CommunicationConfig, seed: int, agent_ids: tuple[str, ...]):
        self.config, self.seed = config, seed
        self.agent_ids = tuple(sorted(agent_ids))
        self.queued: dict[int, list[tuple[int, str, str, tuple[float, ...]]]] = {}

    def reset(self, seed: int):
        self.seed, self.queued = seed, {}

    def send(self, step: int, positions: Mapping[str, tuple[float, float]], messages: Mapping[str, tuple[float, ...]]):
        if not self.config.enabled:
            return
        for sender in self.agent_ids:
            if sender not in messages:
                continue
            message = tuple(float(x) for x in messages[sender])
            if len(message) != self.config.message_dim or not all(math.isfinite(x) and -1 <= x <= 1 for x in message):
                raise PhysicalValidationError("Communication message has invalid dimension or value")
            for receiver in self.agent_ids:
                if sender == receiver:
                    continue
                if self.config.max_range_m is not None:
                    dx = positions[sender][0] - positions[receiver][0]
                    dy = positions[sender][1] - positions[receiver][1]
                    if dx*dx + dy*dy > self.config.max_range_m**2:
                        continue
                digest = hashlib.sha256(f"{self.seed}:{sender}:{receiver}:{step}".encode()).digest()
                draw = int.from_bytes(digest[:8], "big") / 2**64
                if draw < self.config.dropout_probability:
                    continue
                delivery = step + self.config.latency_steps
                self.queued.setdefault(delivery, []).append((step, sender, receiver, message))

    def receive(self, step: int, receiver: str):
        rows = sorted(self.queued.get(step, []), key=lambda row: (row[0], row[1], row[2]))
        by_receiver = {(sender): message for _, sender, target, message in rows if target == receiver}
        self.queued[step] = [row for row in rows if row[2] != receiver]
        if not self.queued[step]:
            self.queued.pop(step)
        senders = tuple(sender for sender in self.agent_ids if sender != receiver)
        values = tuple(by_receiver.get(sender, (0.,) * self.config.message_dim) for sender in senders)
        present = tuple(sender in by_receiver for sender in senders)
        return {"values": values, "present": present, "senders": senders}
