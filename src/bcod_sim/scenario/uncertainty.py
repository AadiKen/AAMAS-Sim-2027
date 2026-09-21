"""Stable-substream deterministic resolution of parameter uncertainty."""
from dataclasses import dataclass
import hashlib
import numpy as np
from bcod_sim.core.errors import PhysicalValidationError

@dataclass(frozen=True)
class Uncertainty:
    distribution:str
    parameters:dict[str,float]

def resolve_uncertain(nominal:float,uncertainty:Uncertainty|None,*,master_seed:int,parameter_path:str)->tuple[float,dict[str,object]]:
    if uncertainty is None: return nominal,{"nominal":nominal,"sampled":nominal,"distribution":None}
    seed=int.from_bytes(hashlib.sha256(f"{master_seed}:{parameter_path}".encode()).digest()[:8],"little"); rng=np.random.default_rng(seed)
    p=uncertainty.parameters
    if uncertainty.distribution=="normal": value=float(rng.normal(nominal,p["sigma"]))
    elif uncertainty.distribution=="uniform": value=float(rng.uniform(p["minimum"],p["maximum"]))
    elif uncertainty.distribution=="triangular": value=float(rng.triangular(p["minimum"],nominal,p["maximum"]))
    else: raise PhysicalValidationError("Unsupported uncertainty distribution")
    return value,{"nominal":nominal,"sampled":value,"distribution":uncertainty.distribution,"parameters":p,"master_seed":master_seed,"substream_seed":seed,"parameter_path":parameter_path}
