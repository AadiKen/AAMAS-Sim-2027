"""Explicit debug-only canonical state channel."""

from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.sensors.base import SensorConfig, SensorContext, SensorPacket, packet


class GroundTruthState:
    kind = "ground_truth"
    requirements = frozenset({"vehicle.pose", "vehicle.velocity"})

    def __init__(self, config: SensorConfig) -> None:
        if config.sensor_type != "ground_truth_state" or config.noise_std != 0:
            raise PhysicalValidationError("Ground truth channel requires explicit type and zero noise")
        self.config = config

    def sample(self, context: SensorContext, *, sample_step: int) -> SensorPacket:
        state = context.state
        return packet(self.config, self.kind, sample_step,
                      {"position_ned_m": state.position_ned.clone(),
                       "q_body_to_ned": state.q_body_to_ned.clone(),
                       "nu_body": state.nu_body.clone()},
                      {"position_ned_m": "m", "q_body_to_ned": "unitless",
                       "nu_body": "m/s,rad/s"},
                      {"position_ned_m": "NED", "q_body_to_ned": "FRD->NED", "nu_body": "FRD"})
