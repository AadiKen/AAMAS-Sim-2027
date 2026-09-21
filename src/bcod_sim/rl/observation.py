"""Ordered, hash-stable observation contracts and strict assembly."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from bcod_sim.config.hashing import content_hash
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.sensors.base import SensorKind, SensorPacket


@dataclass(frozen=True)
class ObservationField:
    name: str
    sensor_id: str
    sensor_type: str
    sensor_version: str
    sensor_source: str
    sensor_kind: SensorKind
    value_key: str
    shape: tuple[int | None, ...]
    dtype: str
    units: str
    frame: str
    normalization: str = "none"

    def __post_init__(self) -> None:
        if not all((self.name, self.sensor_id, self.sensor_type, self.sensor_version, self.sensor_source,
                    self.value_key, self.dtype, self.units, self.frame,
                    self.normalization)) or any(x is not None and x < 0 for x in self.shape):
            raise PhysicalValidationError("Invalid observation field contract")


@dataclass(frozen=True)
class ObservationContract:
    fields: tuple[ObservationField, ...]
    allow_ground_truth: bool = False

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise PhysicalValidationError("Duplicate observation field")
        if any(field.sensor_kind == "ground_truth" for field in self.fields) and not self.allow_ground_truth:
            raise PhysicalValidationError("Ground-truth observations require explicit debug opt-in")

    @property
    def content_hash(self) -> str:
        return content_hash({"fields": [vars(field) for field in self.fields],
                             "allow_ground_truth": self.allow_ground_truth})

    def assemble(self, packets: Mapping[str, SensorPacket]) -> Mapping[str, object]:
        values = {}
        for field in self.fields:
            if field.sensor_id not in packets:
                raise PhysicalValidationError(f"Missing delivered sensor packet: {field.sensor_id}")
            packet = packets[field.sensor_id]
            if (packet.kind != field.sensor_kind or packet.sensor_type != field.sensor_type or
                packet.sensor_version != field.sensor_version or packet.sensor_source != field.sensor_source or
                field.value_key not in packet.values):
                raise PhysicalValidationError(f"Sensor kind or field mismatch: {field.name}")
            if packet.units.get(field.value_key) != field.units or packet.frames.get(field.value_key) != field.frame:
                raise PhysicalValidationError(f"Sensor units/frame mismatch: {field.name}")
            value = packet.values[field.value_key]
            if isinstance(value, torch.Tensor):
                shape = tuple(value.shape)
                dtype = str(value.dtype).removeprefix("torch.")
            elif isinstance(value, tuple):
                shape = (len(value),)
                dtype = "detections"
            else:
                raise PhysicalValidationError(f"Unsupported observation value: {field.name}")
            if len(shape) != len(field.shape) or any(expected is not None and expected != actual
                                                      for expected, actual in zip(field.shape, shape)) or dtype != field.dtype:
                raise PhysicalValidationError(f"Observation shape/dtype mismatch: {field.name}")
            values[field.name] = value
        return MappingProxyType(values)
