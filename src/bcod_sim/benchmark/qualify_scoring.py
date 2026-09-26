"""Deterministic reward/goal audit using prescribed trajectories, not simulator performance."""
import json
import math
from pathlib import Path

from .core import Benchmark,BenchmarkConfig,NAMES,Obstacle,Scenario,VesselPose,Truth,VesselReading


class PrescribedTrajectory:
    def __init__(self, points):self.points=iter(points)
    def reset(self,scenario,seed):
        self.scenario=scenario
        self.offsets=[p.y_m for p in scenario.starts]
        return self._frame((0.,0.))
    def _frame(self,point):
        truth={n:Truth(point[0],point[1]+y,0.) for n,y in zip(NAMES,self.offsets)}
        readings={n:VesselReading(p.x_m,p.y_m,0.,0.,0.) for n,p in truth.items()}
        return readings,truth
    def step(self,actions):return self._frame(next(self.points))
    def close(self):pass


def path(waypoints,speed=1.):
    current=(0.,0.)
    for target in waypoints:
        while math.dist(current,target)>1e-10:
            distance=math.dist(current,target);fraction=min(1.,speed*.2/distance)
            current=tuple(a+(b-a)*fraction for a,b in zip(current,target))
            yield current
    while True:yield current


def reward_audit():
    rows=[]
    cases=[('stationary',[],1.,20.,None),('straight_success',[(20.,0.)],1.,20.,None),
           ('progress_then_collision',[(20.,0.)],1.,20.,15.),
           ('partial_progress_timeout',[(10.,0.)],1.,20.,None),
           ('safe_detour_success',[(6.,5.),(16.,5.),(20.,0.)],1.,20.,10.),
           ('reckless_collision',[(20.,0.)],2.,20.,15.),
           ('long_progress_then_collision',[(72.,0.)],2.,72.,65.)]
    for name,route,speed,goal,obstacle_x in cases:
        obstacles=[Obstacle(-100.,y,1.) for y in (-150.,-50.,50.,150.,250.,350.)]
        if obstacle_x is not None:obstacles[0]=Obstacle(obstacle_x,-150.,1.)
        scenario=Scenario('nominal',123,tuple(VesselPose(0.,y,0.) for y in (-150.,-50.,50.,150.)),
                          tuple((goal,y) for y in (-150.,-50.,50.,150.)),tuple(obstacles))
        env=Benchmark(PrescribedTrajectory(path(route,speed)),BenchmarkConfig(half_width_m=500.))
        env.reset(scenario);returns=dict.fromkeys(NAMES,0.);discounted=dict.fromkeys(NAMES,0.)
        for step in range(600):
            _,reward,done,truncated,info=env.step({n:(0.,0.) for n in NAMES})
            for n in NAMES:returns[n]+=reward[n];discounted[n]+=.99**step*reward[n]
            if done or truncated:break
        rows.append({'trajectory':name,'returns':returns,'discounted_returns_gamma_099':discounted,**info})
    return rows


if __name__=='__main__':
    destination=Path('artifacts/benchmark-audit/reward.json');destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(reward_audit(),indent=2)+'\n')
