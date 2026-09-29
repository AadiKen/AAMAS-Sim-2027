#!/usr/bin/env python3
import argparse, importlib.metadata as md, json, os, platform, shutil, socket, subprocess, sys
from pathlib import Path

def command(args):
    try: return subprocess.run(args,text=True,capture_output=True,timeout=8).stdout.strip()
    except Exception: return ''
def manifest():
    root=Path(__file__).resolve().parents[2]
    d={'os':platform.platform(),'kernel':platform.release(),'architecture':platform.machine(),'hostname':socket.gethostname(),'python':sys.version,'cpu_model':platform.processor(),'logical_cpus':os.cpu_count(),'slurm':{k:v for k,v in os.environ.items() if k.startswith('SLURM_')},'thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES') if k in os.environ},'git_commit':command(['git','-C',str(root),'rev-parse','HEAD']),'git_dirty':bool(command(['git','-C',str(root),'status','--porcelain'])),'dependencies':{}}
    try:
        import psutil
        d.update(cpu_physical_cores=psutil.cpu_count(logical=False),ram_bytes=psutil.virtual_memory().total)
    except ImportError: pass
    for n in ('torch','numpy','benchmarl','torchrl','tensordict','gymnasium','pettingzoo','pyquaticus','holoocean'):
        try:d['dependencies'][n]=md.version(n)
        except md.PackageNotFoundError:d['dependencies'][n]=None
    try:
        import torch
        d.update(torch_version=torch.__version__,torch_cuda_version=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu_count=torch.cuda.device_count(),gpus=[{'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())])
    except Exception as e:d['torch_error']=str(e)
    d['nvidia_smi']=command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']) if shutil.which('nvidia-smi') else None
    return d
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();d=manifest();s=json.dumps(d,indent=2,default=str)+'\n';print(s)
    if a.output:Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(s)
