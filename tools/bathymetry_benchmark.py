#!/usr/bin/env python3
"""Small repeatable bathymetry overhead benchmark."""
import argparse,json,time,tracemalloc
from pathlib import Path
import torch
from bcod_sim.config.models import RegularWaves
from bcod_sim.world.waves import WaveField

def measure(field,positions,depth,iterations):
    tracemalloc.start(); start=time.perf_counter()
    for index in range(iterations): field.sample_kinematics(positions,index*.02,local_depth_m=depth)
    elapsed=time.perf_counter()-start; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return {"queries_per_second":iterations/elapsed,"peak_memory_bytes":peak}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="artifacts/bathymetry-physics/performance.json");p.add_argument("--iterations",type=int,default=1000);a=p.parse_args()
    positions=torch.zeros((16,3),dtype=torch.float64);depth=torch.full((16,),10.,dtype=torch.float64)
    deep=WaveField(RegularWaves(kind="regular",height_m=.4,period_s=3,direction_rad=.4))
    finite=WaveField(RegularWaves(kind="regular",height_m=.4,period_s=3,direction_rad=.4,depth_model="finite_depth"))
    result={"scenario":"16 simultaneous query points, regular wave","iterations":a.iterations,"before_deep_water":measure(deep,positions,depth,a.iterations),"after_finite_depth":measure(finite,positions,depth,a.iterations)}
    result["throughput_ratio_after_over_before"]=result["after_finite_depth"]["queries_per_second"]/result["before_deep_water"]["queries_per_second"]
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+"\n");print(path)
if __name__=="__main__":main()
