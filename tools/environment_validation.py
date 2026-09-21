#!/usr/bin/env python3
"""Separated deterministic environment-response campaign on production Plant6."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np,torch
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.dynamics.environmental import KinematicWaveLoads,RelativeWindLoads,WindCoefficientPoint
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.config.models import CalmWaves,RegularWaves,IrregularWaves
from bcod_sim.world.waves import WaveField
from bcod_sim.world.world import WorldSample
from mss_6dof_validation import build_plant,bcod_parameters,resolved_parameters,quaternion_to_rpy

ROOT=Path(__file__).resolve().parents[1]
CASES={"still_water":{},"current_only":{"current":(0.5,0.2,0)},"wind_only":{"wind":(5,2,0)},
 "regular_waves_only":{"wave":"regular"},"irregular_waves_only":{"wave":"irregular"},
 "current_wind":{"current":(0.5,0.2,0),"wind":(5,2,0)},"wind_waves":{"wind":(5,2,0),"wave":"regular"},
 "current_wind_waves":{"current":(0.5,0.2,0),"wind":(5,2,0),"wave":"irregular"}}

def sample(current,wind,wave,position,time):
    surface,orbital,acceleration=wave.sample_kinematics(position[None,:],time); z=position.new_zeros((1,3)); scalar=position.new_zeros(1)
    return WorldSample(position.new_tensor([current]),position.new_tensor([wind]),z,surface,orbital,acceleration,position.new_tensor([1025.]),position.new_tensor([1.225]),position.new_tensor([10000.]),scalar,scalar,None,None)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='artifacts/environment-validation/latest'); args=parser.parse_args()
    out=(ROOT/args.output).resolve(); out.mkdir(parents=True,exist_ok=True); dt=.02; duration=30.; times=np.arange(0,duration+dt/2,dt)
    params=bcod_parameters(resolved_parameters()); points=(WindCoefficientPoint(-math.pi,-.8,0,0,0),WindCoefficientPoint(-math.pi/2,0,-1,-.1,-.2),WindCoefficientPoint(0,.8,0,0,0),WindCoefficientPoint(math.pi/2,0,1,.1,.2),WindCoefficientPoint(math.pi,-.8,0,0,0))
    wind_model=RelativeWindLoads(1.5,3.,.8,points); wave_load=KinematicWaveLoads((20,30,40),(8,12,15),(8,8,15)); report={}
    for name,config in CASES.items():
        plant=build_plant(params); state=VesselState(torch.zeros(3,dtype=torch.float64),torch.tensor((1.,0,0,0),dtype=torch.float64),torch.zeros(6,dtype=torch.float64)); rows=[]
        wave_spec=CalmWaves(kind='calm') if 'wave' not in config else (RegularWaves(kind='regular',height_m=.4,period_s=3.,direction_rad=.4,phase_rad=0) if config['wave']=='regular' else IrregularWaves(kind='irregular',spectrum='jonswap',significant_height_m=.4,peak_period_s=3.,direction_rad=.4,component_count=32,seed=17))
        field=WaveField(wave_spec); current=config.get('current',(0,0,0)); wind=config.get('wind',(0,0,0)); wave_wrench=[]; wind_wrench=[]
        for i,time in enumerate(times):
            env=sample(current,wind,field,state.position_ned,time); external={key:torch.zeros(6,dtype=torch.float64) for key in EXTERNAL_TERMS}
            if 'wind' in config: external['wind']=wind_model.evaluate(state,env).tau_body
            if 'wave' in config: external['wave']=wave_load.evaluate(state,env).tau_body
            water=env.current_ned_mps[0]+env.wave_orbital_ned_mps[0]; rpy=quaternion_to_rpy(state.q_body_to_ned.numpy()[None,:])[0]
            rows.append(np.r_[time,state.position_ned.numpy(),rpy,state.nu_body.numpy(),external['current'].numpy(),external['wind'].numpy(),external['wave'].numpy()]); wind_wrench.append(float(torch.linalg.vector_norm(external['wind']))); wave_wrench.append(float(torch.linalg.vector_norm(external['wave'])))
            if i+1<len(times): state=plant.step(state,external,dt,water_velocity_ned=water).state
        values=np.asarray(rows); case_dir=out/name; case_dir.mkdir(parents=True,exist_ok=True)
        with (case_dir/'trajectory.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time','x','y','z','roll','pitch','yaw','u','v','w','p','q','r']+[f'{term}_{axis}' for term in ('current','wind','wave') for axis in ('X','Y','Z','K','M','N')]);w.writerows(values)
        report[name]={"position_drift_m":float(np.linalg.norm(values[-1,1:4]-values[0,1:4])),"heading_drift_rad":float(values[-1,6]-values[0,6]),"heave_rms_m":float(np.sqrt(np.mean(values[:,3]**2))),"roll_rms_rad":float(np.sqrt(np.mean(values[:,4]**2))),"pitch_rms_rad":float(np.sqrt(np.mean(values[:,5]**2))),"speed_rms_mps":float(np.sqrt(np.mean(np.sum(values[:,7:10]**2,axis=1)))),"wind_wrench_rms":float(np.sqrt(np.mean(np.square(wind_wrench)))),"wave_wrench_rms":float(np.sqrt(np.mean(np.square(wave_wrench))),),"control_effort":0.0,"energy_proxy":0.0}
    (out/'report.json').write_text(json.dumps({"campaign":"environment_response","dt_s":dt,"duration_s":duration,"cases":report,"deterministic":True},indent=2)+'\n'); print(out)
if __name__=='__main__': main()
