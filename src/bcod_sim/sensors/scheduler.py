"""Master-step sensor rate and integer-step latency scheduler."""

import math
import hashlib
from dataclasses import replace
from typing import Mapping

from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError
from bcod_sim.sensors.base import Sensor, SensorContext, SensorPacket


class SensorScheduler:
    def __init__(self, sensors: tuple[Sensor, ...], *, master_dt_s: float, rate_tolerance: float = 1e-9) -> None:
        if not math.isfinite(master_dt_s) or master_dt_s <= 0:
            raise PhysicalValidationError("Sensor scheduler requires positive master_dt_s")
        self.master_dt_s = master_dt_s
        self.sensors: dict[tuple[int, int, str], Sensor] = {}
        self.intervals: dict[tuple[int, int, str], int] = {}
        self.pending: dict[int, list[SensorPacket]] = {}
        self.disabled_owners: set[tuple[int, int]] = set()
        self.last_step = -1
        for sensor in sensors:
            reset = getattr(sensor, "reset", None)
            if reset is not None:
                reset()
            config = sensor.config
            key = (config.env_id, config.owner_vessel_id, config.instance_id)
            if key in self.sensors:
                raise DuplicateIdentityError(f"Duplicate sensor instance: {key}")
            requested = 1 / (config.rate_hz * master_dt_s)
            interval = round(requested)
            if interval < 1 or abs(requested - interval) > rate_tolerance:
                raise PhysicalValidationError(f"Sensor {config.instance_id} rate cannot be represented on master clock")
            self.sensors[key] = sensor
            self.intervals[key] = interval

    def tick(self, step_index: int, contexts: Mapping[tuple[int, int], SensorContext]) -> tuple[SensorPacket, ...]:
        if step_index != self.last_step + 1:
            raise PhysicalValidationError("Sensor scheduler must tick exactly once per master step")
        self.last_step = step_index
        for key in sorted(self.sensors):
            if key[:2] in self.disabled_owners:
                continue
            if step_index % self.intervals[key] != 0:
                continue
            owner = key[:2]
            if owner not in contexts:
                raise PhysicalValidationError(f"Missing context for sensor owner {owner}")
            context = contexts[owner]
            if context.world.env_id != owner[0]:
                raise PhysicalValidationError("Sensor context world belongs to a different environment")
            expected_time = step_index * self.master_dt_s
            if not math.isclose(context.sim_time_s, expected_time, rel_tol=0, abs_tol=1e-9):
                raise PhysicalValidationError("Sensor context time disagrees with master clock")
            if expected_time < self.sensors[key].config.warmup_s:
                continue
            probability = self.sensors[key].config.dropout_probability
            if probability:
                token = f"{self.sensors[key].config.seed}|{key}|{step_index}|dropout".encode()
                draw = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
                if draw < probability:
                    continue
            sample = self.sensors[key].sample(context, sample_step=step_index)
            sample = replace(sample,
                sample_time_s=expected_time if sample.sample_time_s is None else sample.sample_time_s,
                delivery_time_s=(sample.delivery_step*self.master_dt_s))
            self.pending.setdefault(sample.delivery_step, []).append(sample)
        return tuple(sorted(self.pending.pop(step_index, []), key=lambda packet: (
            packet.env_id, packet.owner_vessel_id, packet.sensor_id)))

    def disable_owner(self, owner: tuple[int, int]) -> None:
        self.disabled_owners.add(owner)
        for step in tuple(self.pending):
            kept = [packet for packet in self.pending[step]
                    if (packet.env_id, packet.owner_vessel_id) != owner]
            if kept:
                self.pending[step] = kept
            else:
                del self.pending[step]
