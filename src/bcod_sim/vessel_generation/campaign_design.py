"""CFD excitation design and requested-coefficient identifiability checks."""
from dataclasses import dataclass
import numpy as np
from bcod_sim.core.errors import PhysicalValidationError

SUPPORTED={
 "X_u":("surge","u"),"X_uu":("surge","abs_u_u"),"Y_v":("sway","v"),"Y_vv":("sway","abs_v_v"),
 "N_r":("yaw","r"),"N_rr":("yaw","abs_r_r"),"Y_r":("combined_sway_yaw","r"),"N_v":("combined_sway_yaw","v"),
 "Z_w":("heave","w"),"K_p":("roll","p"),"M_q":("pitch","q")}

@dataclass(frozen=True)
class CFDCampaignDesign:
    target_parameters:tuple[str,...]
    cases:tuple[dict[str,float],...]
    design_rank:int
    condition_number:float
    identifiable:bool

def design_cfd_campaign(target_parameters:list[str]|tuple[str,...],vessel:object=None,constraints:dict[str,float]|None=None)->CFDCampaignDesign:
    unknown=[p for p in target_parameters if p not in SUPPORTED]
    if unknown: raise PhysicalValidationError(f"Unsupported CFD targets: {unknown}")
    limit=(constraints or {}).get("velocity_limit",1.0); rate=(constraints or {}).get("rate_limit",.3)
    cases=[]
    for motion in sorted({SUPPORTED[p][0] for p in target_parameters}):
        if motion=="surge": cases += [{"u":s*limit} for s in (-1,-.5,.5,1)]
        elif motion=="sway": cases += [{"v":s*limit} for s in (-1,-.5,.5,1)]
        elif motion=="yaw": cases += [{"r":s*rate} for s in (-1,-.5,.5,1)]
        elif motion=="combined_sway_yaw": cases += [{"v":sv*limit,"r":sr*rate} for sv,sr in ((-1,-1),(-1,1),(1,-1),(1,1),(.5,-1))]
        else: cases += [{motion[0]:s*rate} for s in (-1,1)]
    axes="XYZKMN"
    def feature(case,name,output_axis):
        if name[0]!=output_axis: return 0.
        key=SUPPORTED[name][1]; u=case.get(key[-1],0.)
        if key.startswith("abs_"): return abs(u)*u
        return u
    matrix=np.array([[feature(case,p,axis) for p in target_parameters] for case in cases for axis in axes],dtype=float)
    rank=int(np.linalg.matrix_rank(matrix)); condition=float(np.linalg.cond(matrix)) if rank==len(target_parameters) else float("inf")
    result=CFDCampaignDesign(tuple(target_parameters),tuple(cases),rank,condition,rank==len(target_parameters))
    if not result.identifiable: raise PhysicalValidationError(f"Requested CFD parameter set is not identifiable (rank {rank}/{len(target_parameters)})")
    return result
