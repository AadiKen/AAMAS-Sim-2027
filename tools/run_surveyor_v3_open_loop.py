"""Uncalibrated Plant6 open-loop sanity cases for the frozen Surveyor package."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.vessel_generation.coefficient_package import _runtime_plant, load_coefficient_package

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/surveyor_cad_validation/surrogate_cad/v3_fallback_repair'
PACKAGE = BASE/'coefficients/nominal/coefficient_package.yaml'
DT = .02
STEPS = 150


def tensor(values) -> torch.Tensor:
    return torch.as_tensor(values,dtype=torch.float64)


def state(equilibrium, position_delta=(0.,0.,0.), velocity=(0.,0.,0.,0.,0.,0.),
          roll_delta=0., pitch_delta=0.) -> VesselState:
    heave,roll,pitch=equilibrium
    roll+=roll_delta;pitch+=pitch_delta
    q=(math.cos(roll/2)*math.cos(pitch/2), math.sin(roll/2)*math.cos(pitch/2),
       math.cos(roll/2)*math.sin(pitch/2), -math.sin(roll/2)*math.sin(pitch/2))
    p=(position_delta[0],position_delta[1],heave+position_delta[2])
    return VesselState(tensor(p),tensor(q),tensor(velocity))


def force(surge=0.,yaw=0.) -> dict:
    result={name:torch.zeros(6,dtype=torch.float64) for name in EXTERNAL_TERMS}
    result['propulsion'][0]=surge
    result['propulsion'][5]=yaw
    return result


def run() -> None:
    package=load_coefficient_package(PACKAGE)
    plant=_runtime_plant(package)
    equilibrium=package['runtime_payload']['equilibrium_heave_roll_pitch']
    angle=.02
    scenarios={
        'zero_input_equilibrium':(state(equilibrium),force()),
        'straight_surge':(state(equilibrium),force(surge=8.)),
        'reverse_surge':(state(equilibrium),force(surge=-5.)),
        'pure_yaw_moment':(state(equilibrium),force(yaw=.8)),
        'coast_down':(state(equilibrium,velocity=(.7,0,0,0,0,0)),force()),
        'small_yaw_rate':(state(equilibrium,velocity=(0,0,0,0,0,.1)),force()),
        'combined_surge_yaw':(state(equilibrium,velocity=(.4,.05,0,0,0,.05)),force(surge=4.,yaw=.3)),
        'heave_return':(state(equilibrium,position_delta=(0,0,.01)),force()),
        'roll_return':(state(equilibrium,roll_delta=angle),force()),
        'pitch_return':(state(equilibrium,pitch_delta=angle),force()),
    }
    rows={}; summaries={}
    for name,(initial,external) in scenarios.items():
        current=initial
        time=[]; position=[];velocity=[];orientation=[];energy=[]
        initial_acceleration=plant.acceleration(initial,external).numpy()
        initial_restoring=plant.diagnostics(initial,external).terms['restoring'].numpy()
        for i in range(STEPS+1):
            time.append(i*DT);position.append(current.position_ned.numpy().tolist())
            velocity.append(current.nu_body.numpy().tolist())
            orientation.append(current.q_body_to_ned.numpy().tolist())
            energy.append(float(.5*current.nu_body@plant.total_mass@current.nu_body))
            if i<STEPS:current=plant.step(current,external,DT).state
        arr=np.asarray(velocity)
        item={'time_s':time,'position_ned_m':position,'nu_body':velocity,
              'orientation_quaternion':orientation,'kinetic_energy_j':energy}
        rows[name]=item
        summaries[name]={'initial_acceleration_frd':initial_acceleration.tolist(),
                         'initial_restoring_frd':initial_restoring.tolist(),
                         'final_position_ned_m':position[-1],
                         'final_nu_body':velocity[-1],
                         'initial_energy_j':energy[0],'final_energy_j':energy[-1],
                         'maximum_abs_speed_or_rate':float(np.abs(arr).max()),
                         'finite':bool(np.isfinite(arr).all())}
        print(name,summaries[name]['final_nu_body'],flush=True)
    checks={
        'finite_all':all(x['finite'] for x in summaries.values()),
        'zero_equilibrium':bool(np.max(np.abs(rows['zero_input_equilibrium']['nu_body']))<1e-6),
        'surge_signs':bool(summaries['straight_surge']['final_nu_body'][0]>0 and summaries['reverse_surge']['final_nu_body'][0]<0),
        'yaw_sign':bool(summaries['pure_yaw_moment']['final_nu_body'][5]>0),
        'coast_down_energy':bool(summaries['coast_down']['final_energy_j']<summaries['coast_down']['initial_energy_j']),
        'yaw_decay':bool(abs(summaries['small_yaw_rate']['final_nu_body'][5])<.1),
        'hydrostatic_restoring_signs':bool(summaries['heave_return']['initial_restoring_frd'][2]<0 and
                                       summaries['roll_return']['initial_restoring_frd'][3]<0 and
                                       summaries['pitch_return']['initial_restoring_frd'][4]<0),
        'bounded_response':bool(max(x['maximum_abs_speed_or_rate'] for x in summaries.values())<3.),
    }
    report={'schema':'surveyor-v3-open-loop-sanity-1','mass_kg':52.3,'dt_s':DT,
            'duration_s':STEPS*DT,'actuator_geometry_defined':False,
            'yaw_test_input':'external yaw moment diagnostic; no physical thruster model inferred',
            'checks':checks,'passed':all(checks.values()),'scenarios':summaries}
    (BASE/'dynamics_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (BASE/'dynamics_trajectories.json').write_text(json.dumps(rows)+'\n')
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for name in ('zero_input_equilibrium','straight_surge','reverse_surge','coast_down','combined_surge_yaw'):
        axes[0].plot(rows[name]['time_s'],np.asarray(rows[name]['nu_body'])[:,0],label=name)
    for name in ('pure_yaw_moment','small_yaw_rate','combined_surge_yaw'):
        axes[1].plot(rows[name]['time_s'],np.asarray(rows[name]['nu_body'])[:,5],label=name)
    for name in ('heave_return','roll_return','pitch_return'):
        component={'heave_return':2,'roll_return':3,'pitch_return':4}[name]
        axes[2].plot(rows[name]['time_s'],np.asarray(rows[name]['nu_body'])[:,component],label=name)
    for ax,label in zip(axes,('Surge u (m/s)','Yaw rate r (rad/s)','Return velocity/rate')):
        ax.set_ylabel(label);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(BASE/'figures/open_loop_sanity.png',dpi=170);plt.close(fig)
    if not report['passed']:raise RuntimeError(f'Open-loop sanity failed: {checks}')


if __name__=='__main__':
    run()
