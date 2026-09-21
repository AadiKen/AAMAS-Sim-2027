"""Common pure/stateful component interfaces; all outputs are plant-origin body FRD."""
from dataclasses import dataclass,field
from typing import Mapping,Protocol
import torch
from bcod_sim.dynamics.restoring import WrenchResult
from bcod_sim.state.vessel_state import VesselState

@dataclass(frozen=True)
class PhysicsEnvironment:
    water_velocity_ned:torch.Tensor
    water_acceleration_ned:torch.Tensor|None=None
    water_density_kg_m3:float|None=None
    air_velocity_ned:torch.Tensor|None=None
    air_density_kg_m3:float|None=None
    surface_elevation_ned_m:torch.Tensor|None=None
    water_depth_m:torch.Tensor|None=None

@dataclass
class PhysicsRuntimeContext:
    sim_time_s:float
    dt_s:float
    mutable:dict[str,object]=field(default_factory=dict)

class PhysicsComponent(Protocol):
    model_name:str
    def reset(self,runtime_context:PhysicsRuntimeContext)->None: ...
    def compute(self,state:VesselState,environment:PhysicsEnvironment,runtime_context:PhysicsRuntimeContext)->WrenchResult: ...
