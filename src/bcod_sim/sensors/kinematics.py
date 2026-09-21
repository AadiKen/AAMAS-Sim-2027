"""Full mount pose, lever-arm velocity, and rotating-frame acceleration."""

from dataclasses import dataclass

import torch

from bcod_sim.frames.tensor import q_multiply, rotate_body_to_world, rotate_world_to_body
from bcod_sim.sensors.base import SensorConfig, SensorContext


@dataclass(frozen=True)
class MountKinematics:
    position_ned_m: torch.Tensor
    q_mount_to_ned: torch.Tensor
    velocity_ned_mps: torch.Tensor
    acceleration_body_mps2: torch.Tensor | None


def mount_kinematics(config: SensorConfig, context: SensorContext) -> MountKinematics:
    state = context.state
    r = state.position_ned.new_tensor(config.mount_frd_m)
    q_mount = state.q_body_to_ned.new_tensor(config.mount_q_to_frd)
    q_world = q_multiply(state.q_body_to_ned, q_mount)
    omega = state.nu_body[3:]
    point_position = state.position_ned + rotate_body_to_world(r, state.q_body_to_ned)
    point_velocity_body = state.nu_body[:3] + torch.linalg.cross(omega, r)
    point_velocity = rotate_body_to_world(point_velocity_body, state.q_body_to_ned)
    point_acceleration = None
    if context.linear_acceleration_body_mps2 is not None and context.angular_acceleration_body_radps2 is not None:
        # ν_dot is a rotating-frame derivative; add ω×v before lever-arm terms.
        point_acceleration = (context.linear_acceleration_body_mps2 +
                              torch.linalg.cross(omega, state.nu_body[:3]) +
                              torch.linalg.cross(context.angular_acceleration_body_radps2, r) +
                              torch.linalg.cross(omega, torch.linalg.cross(omega, r)))
    return MountKinematics(point_position, q_world, point_velocity, point_acceleration)
