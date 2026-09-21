"""Configurable command/RPM thrust maps and stateful propulsor dynamics."""
from dataclasses import dataclass
import bisect,math
from typing import Protocol
from bcod_sim.actuators.base import ActuatorConfig,ActuatorResult,ActuatorState,declared_power,mounted_wrench
from bcod_sim.core.errors import CommandBoundsError,PhysicalValidationError

class ThrustMap(Protocol):
    model_name:str
    def thrust(self,command:float)->float: ...

@dataclass(frozen=True)
class LinearThrustMap:
    slope_n_per_unit:float; intercept_n:float=0.; model_name:str="linear"
    def thrust(self,command:float)->float:
        value=self.slope_n_per_unit*command+self.intercept_n
        if not math.isfinite(value): raise PhysicalValidationError("Nonfinite thrust map result")
        return value

@dataclass(frozen=True)
class PiecewiseLinearThrustMap:
    commands:tuple[float,...]; thrust_n:tuple[float,...]; extrapolation:str="error"; model_name:str="piecewise_linear"
    def __post_init__(self):
        if len(self.commands)<2 or len(self.commands)!=len(self.thrust_n) or any(not math.isfinite(x) for x in (*self.commands,*self.thrust_n)) or any(b<=a for a,b in zip(self.commands,self.commands[1:])) or self.extrapolation not in ("error","clamp"):
            raise PhysicalValidationError("Invalid piecewise thrust map")
    def thrust(self,command:float)->float:
        if not math.isfinite(command): raise CommandBoundsError("Nonfinite thrust command")
        if command<self.commands[0] or command>self.commands[-1]:
            if self.extrapolation=="error": raise CommandBoundsError("Thrust-map command outside table")
            command=min(max(command,self.commands[0]),self.commands[-1])
        i=min(max(bisect.bisect_right(self.commands,command)-1,0),len(self.commands)-2)
        f=(command-self.commands[i])/(self.commands[i+1]-self.commands[i])
        return self.thrust_n[i]+f*(self.thrust_n[i+1]-self.thrust_n[i])

@dataclass(frozen=True)
class PolynomialThrustMap:
    coefficients_ascending:tuple[float,...]; command_bounds:tuple[float,float]; extrapolation:str="error"; model_name:str="polynomial"
    def thrust(self,command:float)->float:
        lo,hi=self.command_bounds
        if not lo<=command<=hi:
            if self.extrapolation=="error": raise CommandBoundsError("Polynomial thrust command outside validity")
            command=min(max(command,lo),hi)
        return sum(c*command**i for i,c in enumerate(self.coefficients_ascending))

@dataclass(frozen=True)
class RpmCommand: rpm:float

class RPMThruster:
    """RPM state advances once when the engine advances the actuator bank."""
    def __init__(self,config:ActuatorConfig,thrust_map:ThrustMap,*,rpm_bounds:tuple[float,float],rpm_time_constant_s:float|None=None,rpm_rate_limit_per_s:float|None=None,rpm_deadband:float=0.):
        self.config=config; self.thrust_map=thrust_map; self.rpm_bounds=rpm_bounds; self.rpm_time_constant_s=rpm_time_constant_s; self.rpm_rate_limit_per_s=rpm_rate_limit_per_s; self.rpm_deadband=rpm_deadband
        if rpm_bounds[0]>=rpm_bounds[1] or rpm_deadband<0: raise PhysicalValidationError("Invalid RPM actuator bounds")
    def step(self,command:RpmCommand,state:ActuatorState,dt_s:float,local_water_velocity_frd_mps:tuple[float,float,float]|None=None)->ActuatorResult:
        if not isinstance(command,RpmCommand) or not math.isfinite(command.rpm) or dt_s<=0: raise PhysicalValidationError("Invalid RPM actuator step")
        target=min(max(command.rpm,*self.rpm_bounds[:1]),self.rpm_bounds[1])
        if abs(target)<=self.rpm_deadband: target=0.
        rpm=target
        if self.rpm_time_constant_s is not None: rpm=state.rpm+(target-state.rpm)*(1-math.exp(-dt_s/self.rpm_time_constant_s))
        if self.rpm_rate_limit_per_s is not None:
            delta=self.rpm_rate_limit_per_s*dt_s; rpm=min(max(rpm,state.rpm-delta),state.rpm+delta)
        thrust=self.thrust_map.thrust(rpm)
        lo,hi=self.config.thrust_bounds_n.minimum,self.config.thrust_bounds_n.maximum
        if not lo<=thrust<=hi: raise CommandBoundsError("Mapped thrust exceeds configured physical bounds")
        next_state=ActuatorState(thrust,state.steering_rad,rpm,state.rudder_rad)
        return ActuatorResult(next_state,mounted_wrench(self.config,(thrust,0,0)),declared_power(self.config,thrust),())
