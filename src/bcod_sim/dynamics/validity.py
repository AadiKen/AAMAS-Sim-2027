"""Model validity envelopes and deterministic excursion tracking."""
from dataclasses import dataclass,field
from typing import Mapping
import math
from bcod_sim.core.errors import OperatingEnvelopeError,PhysicalValidationError

@dataclass(frozen=True)
class ValidityBound:
    minimum:float; maximum:float; provenance:str
    def __post_init__(self):
        if not(math.isfinite(self.minimum) and math.isfinite(self.maximum) and self.minimum<self.maximum and self.provenance): raise PhysicalValidationError("Invalid validity bound")

@dataclass(frozen=True)
class ModelValidityEnvelope:
    model_name:str
    bounds:Mapping[str,ValidityBound]
    strict:bool=False

@dataclass
class ValidityMonitor:
    envelope:ModelValidityEnvelope
    maxima:dict[str,float]=field(default_factory=dict)
    first_exceedance:dict[str,float]=field(default_factory=dict)
    validated:bool=True
    def observe(self,values:Mapping[str,float],time_s:float)->None:
        for name,bound in self.envelope.bounds.items():
            if name not in values: raise PhysicalValidationError(f"Missing required validity variable: {name}")
            value=float(values[name]); self.maxima[name]=max(self.maxima.get(name,0.),abs(value))
            if not bound.minimum<=value<=bound.maximum:
                self.validated=False; self.first_exceedance.setdefault(name,time_s)
                if self.envelope.strict: raise OperatingEnvelopeError(f"{self.envelope.model_name}.{name} exceeded validity")
