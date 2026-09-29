#!/usr/bin/env python3
"""Fixed four-vessel CPU physics scaling; append each completed repetition."""
import argparse,csv,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,psutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/validation'))
sys.path.insert(0,str(ROOT/'tools'))
from test_stage5a_marl import command
from stage5c_scaling_validation import make_case,names_for
from bcod_sim.rl.vector_env import VectorEnvironment
from hardware_manifest import manifest
FIELDS=('count','repetition','status','aggregate_env_steps_per_s','vessel_steps_per_s','per_env_steps_per_s','p50_ms','p95_ms','p99_ms','rss_bytes','peak_rss_bytes','wall_s','simulated_s','real_time_factor','cpu_percent_one_core','load_average','available_ram_bytes','gpu_utilization','gpu_memory_used','slurm_job_id')
def gpu():
    import subprocess
    try:return subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3).stdout.strip()
    except Exception:return ''
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='paper_results/linux_gpu/scaling');p.add_argument('--repetitions',type=int,default=5);p.add_argument('--steps',type=int,default=8);p.add_argument('--warmup',type=int,default=2);p.add_argument('--counts',type=int,nargs='+',default=[1,8,32,128,512,2048,8192]);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);csvpath=out/'scaling.csv'
    if csvpath.exists() and not a.overwrite:raise SystemExit('Existing scaling.csv; use --overwrite or new output-dir')
    (out/'hardware.json').write_text(json.dumps(manifest(),indent=2,default=str)+'\n')
    (out/'manifest.json').write_text(json.dumps({'simulator_physics_device':'CPU','policy_training_device':'none','vessels_per_environment':4,'dt_s':0.1,'workload':'stage5c light','slurm_allocation':os.environ.get('SLURM_JOB_ID'),'counts':a.counts,'repetitions':a.repetitions,'steps':a.steps,'warmup':a.warmup},indent=2)+'\n')
    proc=psutil.Process(); contaminated=False
    with csvpath.open('w',newline='') as f:
      writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();f.flush()
      for count in a.counts:
       for rep in range(a.repetitions):
        vector=None
        try:
         load=os.getloadavg()[0];avail=psutil.virtual_memory().available;g=gpu()
         if load>max(2,os.cpu_count() or 1)*1.5 and not os.environ.get('SLURM_JOB_ID'):contaminated=True
         if avail<max(1_000_000_000,count*1_000_000):raise MemoryError('Insufficient available memory before allocation')
         engines={i:make_case(4,'light').engine for i in range(count)};vector=VectorEnvironment(engines)
         actions={i:{name:command(0) for name in names_for(4)} for i in engines}
         vector.reset(seeds={i:53+rep for i in engines})
         for _ in range(a.warmup):vector.step(actions)
         vector.reset(seeds={i:53+rep for i in engines})
         rss=proc.memory_info().rss;cpu=proc.cpu_times();c0=cpu.user+cpu.system;lat=[];start=time.perf_counter()
         for _ in range(a.steps):
          t=time.perf_counter();vector.step(actions);lat.append(time.perf_counter()-t)
         wall=time.perf_counter()-start;cpu=proc.cpu_times();peak=max(rss,proc.memory_info().rss)
         row=dict(count=count,repetition=rep,status='CONTAMINATED_EXPLORATORY' if contaminated else 'CLEAN_CANDIDATE',aggregate_env_steps_per_s=count*a.steps/wall,vessel_steps_per_s=4*count*a.steps/wall,per_env_steps_per_s=a.steps/wall,p50_ms=1000*np.percentile(lat,50),p95_ms=1000*np.percentile(lat,95),p99_ms=1000*np.percentile(lat,99),rss_bytes=proc.memory_info().rss,peak_rss_bytes=peak,wall_s=wall,simulated_s=a.steps*.1,real_time_factor=a.steps*.1/wall,cpu_percent_one_core=100*(cpu.user+cpu.system-c0)/wall,load_average=load,available_ram_bytes=avail,gpu_utilization=g,gpu_memory_used=g,slurm_job_id=os.environ.get('SLURM_JOB_ID',''))
         writer.writerow(row);f.flush();os.fsync(f.fileno());print(count,rep,row['status'],flush=True)
        except (MemoryError,RuntimeError) as e:
         (out/'oom.log').write_text(f'count={count} rep={rep}: {e}\n');print('Partial result, stopping larger N:',e,flush=True);return 0
        finally:del vector
    return 0
if __name__=='__main__':sys.exit(main())
