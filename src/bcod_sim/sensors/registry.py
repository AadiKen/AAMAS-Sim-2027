"""Closed sensor type registry; unknown plugins are fatal."""

from typing import Callable

from bcod_sim.core.errors import DuplicateIdentityError, UnknownReferenceError
from bcod_sim.sensors.abstract import AbstractEntitySensor
from bcod_sim.sensors.base import Sensor, SensorConfig
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.ground_truth import GroundTruthState
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.lidar import LiDAR
from bcod_sim.sensors.sonar import Sonar


class SensorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Sensor]] = {
            "gps": GPS, "imu": IMU, "lidar": LiDAR, "sonar": Sonar,
            "abstract_entities": AbstractEntitySensor,
            "ground_truth_state": GroundTruthState,
        }

    def register(self, sensor_type: str, factory: Callable[..., Sensor]) -> None:
        if sensor_type in self._factories:
            raise DuplicateIdentityError(f"Duplicate sensor type: {sensor_type}")
        self._factories[sensor_type] = factory

    def create(self, config: SensorConfig, **kwargs) -> Sensor:
        try:
            factory = self._factories[config.sensor_type]
        except KeyError as exc:
            raise UnknownReferenceError(f"Unknown sensor type: {config.sensor_type}") from exc
        return factory(config, **kwargs)
