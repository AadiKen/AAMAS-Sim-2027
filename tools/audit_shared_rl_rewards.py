import json
from pathlib import Path
from bcod_sim.benchmark.core import *
from bcod_sim.benchmark.qualify_scoring import PrescribedTrajectory,path
from bcod_sim.benchmark.rl_diagnostics import EpisodeDiagnostics
rows=[]
for name,route,speed in [('aggressive_collision',[(72,0)],2),('slow_safe_detour',[(0,12),(60,12),(72,0)],.8),('partial_timeout',[(30,0)],1),('stationary',[],1),('direct_success',[(72,0)],1),('avoidance_away_first',[(-10,12),(60,12),(72,0)],1)]:
    offsets=(-150,-50,50,150)
    obstacles=[Obstacle(65,-150,1)]+[Obstacle(-100,y,1) for y in offsets]+[Obstacle(-100,250,1)]
    if name=='direct_success':obstacles[0]=Obstacle(65,-170,1)
    s=Scenario('nominal',123,tuple(VesselPose(0,y,0) for y in offsets),tuple((72,y) for y in offsets),tuple(obstacles))
    env=Benchmark(PrescribedTrajectory(path(route,speed)),BenchmarkConfig(half_width_m=500))
    env.reset(s);ep=EpisodeDiagnostics()
    for _ in range(600):
        _,reward,done,truncated,info=env.step({n:(0,0) for n in NAMES});ep.add(reward,info,env)
        if done or truncated:break
    sensitivity={str(g):sum(sum(rewards)*g**i for i,rewards in enumerate(ep.rewards)) for g in (.99,.995,.997,.999,1.)}
    rows.append({'gamma_sensitivity_fleet':sensitivity,'name':name,'speed_mps':speed,'route':route,**ep.result(info)})
result={'gamma':.99,'discount_by_seconds':{str(t):.99**(5*t) for t in (1,10,20,40,60,120)},'trajectories':rows}
Path('artifacts/shared-rl-audit/reward-ordering.json').write_text(json.dumps(result,indent=2))
for r in rows:print(r['name'],r['steps'],r['fleet_return'],r['discounted_fleet_return'],r['per_agent_discounted_return'])
