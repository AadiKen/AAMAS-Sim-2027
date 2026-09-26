"""Native, simulator-independent control/observation audit; never trains a policy.

python -m bcod_sim.benchmark.qualify --sim bcod --output audit-bcod.json
A failed gate is emitted as data and exit status 1, never disguised as a pass.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from .core import Benchmark, BenchmarkConfig, NAMES, Obstacle, Scenario, VesselPose, observation
from .runner import make_adapter


def clear_scenario(heading=0.):
    return Scenario("nominal", 123, tuple(VesselPose(-50., y, heading) for y in (-150., -50., 50., 150.)),
                    tuple((50., y) for y in (-150., -50., 50., 150.)), ())


def response(adapter, config, surge, yaw, duration):
    scenario = clear_scenario()
    readings, truth = adapter.reset(scenario, scenario.seed)
    initial = truth[NAMES[0]].heading_rad
    prior = initial
    heading_change = 0.
    samples = []
    for index in range(round(duration / config.dt_s)):
        readings, truth = adapter.step({n: (surge, yaw) for n in NAMES})
        r = readings[NAMES[0]]
        h = truth[NAMES[0]].heading_rad
        delta = math.atan2(math.sin(h - prior), math.cos(h - prior))
        heading_change += delta
        prior = h
        diagnostics = getattr(adapter, "diagnostics", {})
        samples.append({"time_s": (index + 1) * config.dt_s, "surge_mps": r.surge_mps,
                        "yaw_rps": r.yaw_rps, "truth_yaw_rps": delta / config.dt_s,
                        "heading_error_rad": math.atan2(math.sin(r.heading_rad-h), math.cos(r.heading_rad-h)),
                        "saturated": bool(diagnostics.get("saturated", {}).get(NAMES[0], False))})
    target_speed, target_yaw = max(0., surge)*config.max_surge_mps, yaw*config.max_yaw_rps
    final = samples[-round(5 / config.dt_s):]
    metrics = {}
    for field, target, tolerance in (("surge_mps", target_speed, .2), ("yaw_rps", target_yaw, .05)):
        values = [s[field] for s in samples]
        mean = float(np.mean([s[field] for s in final]))
        metrics[field] = {"target": target, "final_mean": mean, "peak_abs": max(map(abs, values)),
                          "rise_time_90_s": next((s["time_s"] for s in samples if target and
                                                   s[field] * math.copysign(1,target) >= .9*abs(target)), None),
                          "overshoot": max(0., max(v*math.copysign(1,target) for v in values)-abs(target)),
                          "sign": 0 if abs(mean)<1e-6 else int(math.copysign(1,mean)),
                          "monotone": all(b >= a-1e-5 for a,b in zip(values, values[1:])),
                          "tolerance": tolerance, "pass": abs(mean-target) <= tolerance}
    return {"action": [surge, yaw], "metrics": metrics, "heading_change_rad": heading_change,
            "final_saturation_fraction": float(np.mean([s["saturated"] for s in final])),
            "stable": all(math.isfinite(s[f]) for s in samples for f in ("surge_mps", "yaw_rps", "heading_error_rad")),
            "saturation_steps": sum(s["saturated"] for s in samples), "samples": samples}


def control_checks(rows):
    sweeps = [[r for r in rows[:15] if r['action'][0] == u] for u in (.5,.8,1.)]
    rates = [[r['metrics']['yaw_rps']['final_mean'] for r in sweep] for sweep in sweeps]
    return {
        'yaw_symmetry_error': [abs(v[i]+v[4-i]) for v in rates for i in (0,1)],
        'yaw_sign_pass': all(abs(v[2]) <= .01 and all(v[i]*(-1 if i<2 else 1)>0 for i in (0,1,3,4)) for v in rates),
        'yaw_monotonic_pass': all(all(b>a for a,b in zip(v,v[1:])) for v in rates),
        'normal_saturation_pass': all(r['final_saturation_fraction'] == 0 for r in sweeps[1] if abs(r['action'][1]) == 1),
    }


def coordinates(adapter, config):
    results = []
    for heading in [i*math.pi/4 for i in range(8)]:
        starts = (VesselPose(0,0,heading), VesselPose(0,10,0), VesselPose(-20,0,0), VesselPose(0,-40,0))
        obstacles = (Obstacle(10,0,1), Obstacle(0,29.9,1), Obstacle(30,0,1), Obstacle(0,-30.1,1))
        scenario = Scenario('nominal', 123, starts, ((0,20),)*4, obstacles)
        readings, truth = adapter.reset(scenario,123)
        observed = observation(readings[NAMES[0]], scenario.goals[0], config)
        # Independent known ranges and bearings in common East/North axes.
        expected = [0,0,math.sin(heading),math.cos(heading),0,0,0,20/(2*config.half_width_m)]
        for distance, angle in ((10, math.pi/2),(20, math.pi), (0,0),
                                (10,0),(29.9,math.pi/2),(30,0)):
            expected.extend([distance/30, math.sin(angle-heading),math.cos(angle-heading)] if distance else [0,0,0])
        expected.extend([0.] * (47-len(expected)))
        errors = np.abs(observed-np.asarray(expected))
        initial_truth = truth[NAMES[0]]
        for _ in range(5):
            _, truth = adapter.step({n:(.8,.5) for n in NAMES})
        delta = math.atan2(math.sin(truth[NAMES[0]].heading_rad-heading),math.cos(truth[NAMES[0]].heading_rad-heading))
        results.append({"heading_rad": heading,"observed":observed.tolist(),"expected":expected,
                        "absolute_errors":errors.tolist(),"max_error":float(errors.max()),
                        "truth":asdict(initial_truth),"positive_yaw_delta":delta,
                        "pass": bool(errors.max()<1e-5 and delta>0 and abs(initial_truth.x_m)<1e-6 and abs(initial_truth.y_m)<1e-6)})
    return results


def timing(adapter, config):
    scenario = clear_scenario()
    adapter.reset(scenario,scenario.seed)
    result = []
    for step in range(1,601):
        adapter.step({n:(0.,0.) for n in NAMES})
        if step in (1,10,100,600):
            actual = getattr(adapter,'diagnostics',{}).get('sim_time_s')
            result.append({'steps':step,'expected_s':step*.2,'reported_s':actual,
                           'pass': actual is not None and abs(actual-step*.2)<1e-8})
    return result


def navigation_scenario(kind):
    rows = (-30.,-10.,10.,30.)
    starts = tuple(VesselPose(-15.,y,0.) for y in rows)
    goals = tuple((15.,y) for y in rows)
    waypoints = {n:[goals[i]] for i,n in enumerate(NAMES)}
    obstacles = tuple(Obstacle(x,y,1.) for x,y in ((-40,-40),(-40,0),(-40,40),(40,-40),(40,0),(40,40)))
    if kind in ('left','right'):
        angle = -math.pi/2 if kind=='left' else math.pi/2
        starts = tuple(replace(p,heading_rad=angle) for p in starts)
    if kind in ('s_turn','obstacle'):
        waypoints = {n:[(-6.,y+5.),(6.,y-5.),goals[i]] for i,(n,y) in enumerate(zip(NAMES,rows))}
        if kind=='obstacle':
            obstacles = tuple(Obstacle(0.,y,1.) for y in rows)+obstacles[:2]
            waypoints = {n:[(-6.,y+5.),(6.,y+5.),goals[i]] for i,(n,y) in enumerate(zip(NAMES,rows))}
    if kind=='crossing':
        angles=(0,math.pi/2,math.pi,3*math.pi/2)
        starts=tuple(VesselPose(-25*math.cos(a)+6*math.sin(a),-25*math.sin(a)-6*math.cos(a),a) for a in angles)
        goals=tuple((25*math.cos(a)+6*math.sin(a),25*math.sin(a)-6*math.cos(a)) for a in angles)
        waypoints={n:[goals[i]] for i,n in enumerate(NAMES)}
    return Scenario('nominal',123,starts,goals,obstacles),waypoints


def navigation(adapter, config, kind):
    scenario, waypoints = navigation_scenario(kind)
    env = Benchmark(adapter,config)
    obs = env.reset(scenario)
    indices = dict.fromkeys(NAMES,0)
    reached = dict.fromkeys(NAMES,False)
    for step in range(config.max_steps):
        actions={}
        for i,n in enumerate(NAMES):
            o=obs[n]; x,y=o[:2]*config.half_width_m
            target=waypoints[n][indices[n]]
            if math.dist((x,y),target)<2 and indices[n]<len(waypoints[n])-1:
                indices[n]+=1;target=waypoints[n][indices[n]]
            angle=math.atan2(target[1]-y,target[0]-x)-math.atan2(o[2],o[3])
            angle=math.atan2(math.sin(angle),math.cos(angle))
            # Same modest feasible rate envelope and waypoint controller for all.
            yaw=max(-1.,min(1.,angle/config.max_yaw_rps))
            speed=(.6 if kind=='crossing' else .4)*max(.2,math.cos(angle))
            if reached[n] or (kind=='crossing' and step*config.dt_s < i*22):speed,yaw=0.,0.
            actions[n]=(speed,yaw)
        obs,_,done,truncated,info=env.step(actions)
        reached=info['per_agent_success']
        if done or truncated:return {"scenario":kind, **info}


def collisions(adapter,config):
    results=[]
    for kind in ('obstacle','vessel'):
        starts=(VesselPose(-8.,0.,0.),VesselPose(0.,0.,math.pi),VesselPose(-30.,-30.,0.),VesselPose(-30.,30.,0.))
        if kind=='obstacle':starts=(starts[0],VesselPose(20.,20.,0.),*starts[2:])
        far=tuple(Obstacle(100.,100.+10*i,1.) for i in range(6))
        obstacles=(Obstacle(0.,0.,1.),)+far[:5] if kind=='obstacle' else far
        scenario=Scenario('nominal',123,starts,((20.,0.),)*4,obstacles)
        env=Benchmark(adapter,config)
        env.reset(scenario)
        first_native=first_common=None;trace=[]
        for step in range(120):
            actions={n:(.5,0.) if n==NAMES[0] else (0.,0.) for n in NAMES}
            if first_common is None:
                obs,_,_,_,info=env.step(actions)
                truth=env.truth
                speed=float(obs[NAMES[0]][4])*config.max_surge_mps
                common=info['collision_count']>0
            else:
                readings,truth=adapter.step(actions)
                speed=readings[NAMES[0]].surge_mps
                common=True
            p=truth[NAMES[0]]
            diag=getattr(adapter,'diagnostics',{})
            native=bool(diag.get('native_contacts')) or bool(diag.get('native_agent_contacts'))
            row={'time_s':(step+1)*config.dt_s,'x':p.x_m,'y':p.y_m,'surge':speed}
            trace.append(row)
            if native and first_native is None:first_native=row
            if common and first_common is None:first_common=row
            if first_common and first_native:break
        results.append({'kind':kind,'first_native_contact':first_native,'first_common_contact':first_common,'trace':trace})
    return results


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--sim',required=True,choices=('bcod','bcod-reduced','pyquaticus','holoocean'))
    parser.add_argument('--pyquaticus-python')
    parser.add_argument('--output',required=True)
    parser.add_argument('--sections',default='control,coordinates,timing,navigation,collisions')
    parser.add_argument('--duration',type=float,default=30.)
    args=parser.parse_args()
    if args.duration < 5:
        parser.error("duration must include at least the 5 second final window")
    config=BenchmarkConfig(half_width_m=500.,max_steps=650)
    result={'simulator':args.sim,'config':asdict(config),'tolerances':{'surge_mps':.2,'yaw_rps':.05,'symmetry_rps':.03},
            'native_host_required':args.sim=='holoocean'}
    try:
        adapter=make_adapter(args.sim,config,pyquaticus_python=args.pyquaticus_python)
    except (RuntimeError,ImportError) as error:
        result.update(qualification_pass=False, status='NEEDS_NATIVE_HOST', error=str(error))
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        raise SystemExit(2)
    try:
        sections=args.sections.split(',')
        if 'control' in sections:
            pairs=[(u,y) for u in (.5,.8,1.) for y in (-1.,-.5,0.,.5,1.)]+[(u,0.) for u in (0.,.25,.5,.75,1.)]+[(.4,1.)]
            result['control']=[response(adapter,config,u,y,args.duration) for u,y in pairs]
            result.update(control_checks(result['control']))
        if 'coordinates' in sections:result['coordinates']=coordinates(adapter,config)
        if 'timing' in sections:result['timing']=timing(adapter,config)
        if 'navigation' in sections:result['navigation']=[navigation(adapter,config,k) for k in ('straight','left','right','s_turn','obstacle','crossing')]
        if 'collisions' in sections:result['collisions']=collisions(adapter,config)
    finally:
        adapter.close()
    result['control_pass']=all(m['pass'] for r in result.get('control',[]) for m in r['metrics'].values()) and result['yaw_sign_pass'] and result['yaw_monotonic_pass'] and result['normal_saturation_pass'] and all(r['stable'] for r in result['control']) if 'control' in result else None
    result['qualification_pass'] = (result['control_pass'] is True and
        all(v <= .03 for v in result.get('yaw_symmetry_error', [float('inf')])) and
        all(row['pass'] for row in result.get('coordinates', [{'pass':False}])) and
        all(row['pass'] for row in result.get('timing', [{'pass':False}])) and
        all(row['fleet_success'] for row in result.get('navigation', [{'fleet_success':False}])))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('simulator','control_pass','yaw_symmetry_error')}))
    if not result['qualification_pass']:raise SystemExit(1)


if __name__=='__main__':main()
