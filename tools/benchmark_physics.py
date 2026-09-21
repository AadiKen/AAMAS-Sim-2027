#!/usr/bin/env python3
"""Deterministic scalar and multi-environment physics microbenchmarks."""
import json,time,tracemalloc
from pathlib import Path
import torch
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics,LinearHydrostatics
from bcod_sim.dynamics.crossflow import StripTheoryCrossflow
from bcod_sim.state.vessel_state import VesselState

D=torch.float64; z6=torch.zeros(6,dtype=D); state=VesselState(torch.zeros(3,dtype=D),torch.tensor((1.,0,0,0),dtype=D),torch.zeros(6,dtype=D)); external={k:z6.clone() for k in EXTERNAL_TERMS}
mass=MassProperties(55,torch.tensor((.2,0,-.2),dtype=D),torch.diag(torch.tensor((10.,14.,14.),dtype=D)),torch.diag(torch.tensor((3.,82.,55.,2.,11.,23.),dtype=D)))
damp=Damping(torch.tensor((78.,140.,100.,20.,30.,40.),dtype=D),torch.tensor((1.,1.,1.,1.,1.,10.),dtype=D)); env=OperatingEnvelope(torch.ones(6,dtype=D)*20,max_substep_s=.1)
base=Plant6(mass,damp,Hydrostatics(55*9.81,torch.tensor((.2,0,-.4),dtype=D),9.81),env)
G=torch.diag(torch.tensor((0.,0,7500.,1000.,2800.,0.),dtype=D)); upgraded=Plant6(mass,damp,LinearHydrostatics(G,torch.zeros(3,dtype=D),torch.zeros(3,dtype=D),.5,.5),env,crossflow=StripTheoryCrossflow.constant_section(2,.25,.14,20,dtype=D))
def bench(plant,count,repeats):
 current=[state]*count
 for _ in range(5): plant.step(state,external,.01)
 tracemalloc.start(); start=time.perf_counter()
 for _ in range(repeats): current=[plant.step(s,external,.01).state for s in current]
 elapsed=time.perf_counter()-start; _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 steps=count*repeats
 return {'microseconds_per_env_step':elapsed*1e6/steps,'env_steps_per_second':steps/elapsed,'peak_memory_bytes':peak}
result={'scalar_base':bench(base,1,300),'scalar_upgraded':bench(upgraded,1,300),'100_env_upgraded':bench(upgraded,100,3),'1000_env_upgraded':bench(upgraded,1000,1),'execution':'scalar CPU deterministic loop','gpu_vectorized':False}
path=Path('artifacts/physics-foundation/performance.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
