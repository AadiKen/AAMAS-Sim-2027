"""Deterministic NED/SI continuous vector fields."""

from typing import Protocol

import torch

from bcod_sim.config.models import LinearVector, UniformVector, SinusoidalVector
import math


class VectorField(Protocol):
    def sample(self, position_ned_m: torch.Tensor, sim_time_s: float, env_id: int) -> torch.Tensor: ...


class UniformField:
    def __init__(self, vector_ned_mps: tuple[float, float, float], env_id: int) -> None:
        self.vector_ned_mps = vector_ned_mps
        self.env_id = env_id

    def sample(self, position_ned_m: torch.Tensor, sim_time_s: float, env_id: int) -> torch.Tensor:
        if env_id != self.env_id:
            raise ValueError("Field queried with wrong environment ID")
        vector = position_ned_m.new_tensor(self.vector_ned_mps)
        return vector.expand(position_ned_m.shape[0], 3).clone()


class LinearField:
    def __init__(self, spec: LinearVector, env_id: int) -> None:
        self.spec, self.env_id = spec, env_id

    def sample(self, position_ned_m: torch.Tensor, sim_time_s: float, env_id: int) -> torch.Tensor:
        if env_id != self.env_id:
            raise ValueError("Field queried with wrong environment ID")
        origin = position_ned_m.new_tensor(self.spec.origin_ned_m)
        base = position_ned_m.new_tensor(self.spec.base_ned_mps)
        gradient = position_ned_m.new_tensor(self.spec.gradient_per_s)
        return base + (position_ned_m - origin) @ gradient.T

class SinusoidalField:
    def __init__(self,spec:SinusoidalVector,env_id:int)->None: self.spec,self.env_id=spec,env_id
    def sample(self,position_ned_m:torch.Tensor,sim_time_s:float,env_id:int)->torch.Tensor:
        if env_id!=self.env_id: raise ValueError("Field queried with wrong environment ID")
        factor=math.sin(2*math.pi*sim_time_s/self.spec.period_s+self.spec.phase_rad)
        vector=position_ned_m.new_tensor(self.spec.mean_ned_mps)+factor*position_ned_m.new_tensor(self.spec.amplitude_ned_mps)
        return vector.expand(position_ned_m.shape[0],3).clone()


def build_vector_field(spec: UniformVector | LinearVector | SinusoidalVector, env_id: int) -> VectorField:
    if isinstance(spec, UniformVector):
        return UniformField(spec.ned_mps, env_id)
    if isinstance(spec,LinearVector): return LinearField(spec, env_id)
    return SinusoidalField(spec,env_id)
