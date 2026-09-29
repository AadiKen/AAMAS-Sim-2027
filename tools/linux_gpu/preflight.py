#!/usr/bin/env python3
import argparse, importlib.util, json, os, platform, shutil, subprocess, sys, traceback
from pathlib import Path
from hardware_manifest import manifest

def check(name,fn):
    try:return {'ok':True,'detail':fn()}
    except Exception as e:return {'ok':False,'error':f'{type(e).__name__}: {e}'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='paper_results/linux_gpu/preflight.json');p.add_argument('--require-cuda',action='store_true');a=p.parse_args()
    h=manifest();d={'hardware':h,'checks':{},'simulator_physics_device':'CPU','policy_training_device':'CUDA' if h.get('cuda_available') else 'CPU','gpu_rendered_external_simulator':'HoloOcean if installed and engine launches'}
    def core():
        from bcod_sim.benchmark_v3.deadline_harness.marl import ResidualCoordinatorEnv,make_four_vessel_scenario
        import numpy as np
        e=ResidualCoordinatorEnv(scenario=make_four_vessel_scenario(np.random.default_rng(11),"easy"));obs,_=e.reset(seed=11);actions={n:np.array([.5,0.],np.float32) for n in e.possible_agents};e.step(actions);e.close();return 'four-vessel reset and step passed'
    def backward(device):
        import torch
        from bcod_sim.benchmark_v3.deadline_harness.algorithms import MAPPO, TD3Actor
        m=MAPPO().to(device);x=torch.zeros((4,m.obs_mean.numel()),device=device);m.mean_action(x).sum().backward()
        t=TD3Actor().to(device);t(torch.zeros((4,t.obs_mean.numel()),device=device)).sum().backward()
        return 'MAPPO and TD3 backward passed on '+device
    d['checks']['core']=check('core',core)
    d['checks']['cpu_backward']=check('cpu_backward',lambda:backward('cpu'))
    d['checks']['cuda_backward']=check('cuda_backward',lambda:backward('cuda')) if h.get('cuda_available') else {'ok':False,'error':'CUDA unavailable'}
    for n in ('benchmarl','torchrl','tensordict','pettingzoo','gymnasium','pyquaticus','holoocean'):
        d['checks'][n]={'ok':importlib.util.find_spec(n) is not None}
    d['checks']['octave']={'ok':shutil.which('octave') is not None}
    d['disk_free_bytes']=shutil.disk_usage('.').free
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(d,indent=2,default=str)+'\n')
    for k,v in d['checks'].items():print(f'{k}: {"PASS" if v["ok"] else "UNAVAILABLE/FAIL"} {v.get("error", "")}')
    print('physics=CPU policy='+d['policy_training_device']+' JSON='+str(out))
    return int(not d['checks']['core']['ok'] or not d['checks']['cpu_backward']['ok'] or (a.require_cuda and not d['checks']['cuda_backward']['ok']))
if __name__=='__main__':sys.exit(main())
