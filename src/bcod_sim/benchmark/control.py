"""Benchmark command controllers and full-attitude IMU integration.

No plant coefficients are modified here. Saturation is explicit and integral
feedback is frozen when allocation cannot deliver the requested wrench.
"""
import math
import torch

from bcod_sim.frames.tensor import q_multiply, q_normalize
from bcod_sim.dynamics.restoring import quaternion_to_rpy


def integrate_gyro(q, previous_omega, omega, dt):
    """Midpoint three-axis body-rate integration using an exponential quaternion."""
    rotation = (previous_omega + omega) * (0.5 * dt)
    angle = torch.linalg.vector_norm(rotation)
    scale = 0.5 * torch.sinc(angle / (2 * math.pi))
    dq = torch.cat((torch.cos(angle / 2).reshape(1), rotation * scale))
    return q_normalize(q_multiply(q, dq))


def common_heading_rate(q, omega):
    roll, pitch, yaw = (float(v) for v in quaternion_to_rpy(q))
    if abs(math.cos(pitch)) < 1e-6:
        raise ValueError("Heading/yaw rate undefined near vertical attitude")
    rate = -(math.sin(roll) * float(omega[1]) + math.cos(roll) * float(omega[2])) / math.cos(pitch)
    return math.pi / 2 - yaw, rate


class RatePI:
    def __init__(self, kp, ki):
        self.kp, self.ki = kp, ki
        self.integral = 0.0

    def request(self, error, dt, feedforward=0.0):
        self.pending = self.integral + self.ki * error * dt
        return feedforward + self.kp * error + self.pending

    def commit(self, saturated):
        if not saturated:
            self.integral = self.pending
