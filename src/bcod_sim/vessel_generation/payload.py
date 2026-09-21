"""Immutable payload installation into resolved mass properties."""
from dataclasses import dataclass
import hashlib,json
import torch
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.dynamics.matrices import MassProperties,skew

@dataclass(frozen=True)
class Payload:
    mass_kg:float
    position_frd_m:tuple[float,float,float]
    inertia_cg_kg_m2:tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
    aerodynamic_area_delta_m2:tuple[float,float]|None=None

def install_payload(base:MassProperties,payload:Payload)->tuple[MassProperties,dict[str,object]]:
    if payload.mass_kg<=0: raise PhysicalValidationError("Payload mass must be positive")
    dtype,device=base.cg_frd_m.dtype,base.cg_frd_m.device; rp=torch.tensor(payload.position_frd_m,dtype=dtype,device=device); ip=torch.tensor(payload.inertia_cg_kg_m2,dtype=dtype,device=device)
    total=base.mass_kg+payload.mass_kg; cg=(base.mass_kg*base.cg_frd_m+payload.mass_kg*rp)/total
    rb=base.cg_frd_m-cg; rr=rp-cg
    inertia=base.inertia_cg_kg_m2-base.mass_kg*skew(rb)@skew(rb)+ip-payload.mass_kg*skew(rr)@skew(rr)
    resolved=MassProperties(total,cg,inertia,base.added_mass_kg,max_condition_number=base.max_condition_number,
                            added_mass_reference_point_frd_m=base.added_mass_reference_point_frd_m,
                            added_mass_provenance=base.added_mass_provenance,added_mass_units=base.added_mass_units)
    data={"payload":{"mass_kg":payload.mass_kg,"position_frd_m":payload.position_frd_m,"inertia_cg_kg_m2":payload.inertia_cg_kg_m2},"total_mass_kg":total,"cg_frd_m":cg.tolist(),"inertia_cg_kg_m2":inertia.tolist()}
    data["fingerprint"]=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(); return resolved,data
