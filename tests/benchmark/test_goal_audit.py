"""Geometry-only audit of common scoring, independent of native dynamics."""
from dataclasses import replace
import pytest
from bcod_sim.benchmark.core import Benchmark,BenchmarkConfig,NAMES,Scenario,VesselPose,Obstacle,Truth,VesselReading

class Replay:
    def reset(self,scenario,seed):
        self.pose={n:Truth(p.x_m,p.y_m,p.heading_rad) for n,p in zip(NAMES,scenario.starts)}
        return self.read(),dict(self.pose)
    def read(self):return {n:VesselReading(p.x_m,p.y_m,p.heading_rad,0,0) for n,p in self.pose.items()}
    def step(self,actions):return self.read(),dict(self.pose)
    def close(self):pass


def scenario():
    return Scenario('nominal',1,tuple(VesselPose(0,y,0) for y in (-30,-10,10,30)),
                    tuple((10,y) for y in (-30,-10,10,30)),
                    tuple(Obstacle(x,y,1) for x,y in ((-40,-40),(-40,0),(-40,40),(40,-40),(40,0),(40,40))))


def test_goal_entry_latches_and_other_vessels_continue_then_collision():
    adapter=Replay();env=Benchmark(adapter)
    env.reset(scenario());zero={n:(0,0) for n in NAMES}
    adapter.pose[NAMES[0]]=Truth(8.,-30.,0.) # exact 2m boundary
    _,r,done,_,info=env.step(zero)
    assert not done and info['per_agent_success'][NAMES[0]]
    assert r[NAMES[0]]==pytest.approx(10-env.config.gamma*2+20-.01)
    adapter.pose[NAMES[0]]=Truth(5.,-30.,0.)
    _,r,done,_,info=env.step(zero)
    assert info['per_agent_success'][NAMES[0]] and r[NAMES[0]]==pytest.approx(2-env.config.gamma*5-.01)
    adapter.pose[NAMES[1]]=Truth(5.,-30.,0.)
    _,r,done,_,info=env.step(zero)
    assert done and not info['fleet_success'] and info['collision_count']==2
    assert info['per_agent_success'][NAMES[0]]


def test_swept_goal_crossing_and_nonsimultaneous_fleet_completion():
    adapter=Replay();env=Benchmark(adapter);env.reset(scenario());zero={n:(0,0) for n in NAMES}
    for i,n in enumerate(NAMES):
        adapter.pose[n]=Truth(14.,scenario().starts[i].y_m,0.)
        _,_,done,_,info=env.step(zero)
        assert info['per_agent_success'][n]
        assert done==(i==3)
    assert info['fleet_success']


def test_just_outside_goal_does_not_count():
    adapter=Replay();env=Benchmark(adapter);env.reset(scenario())
    adapter.pose[NAMES[0]]=Truth(7.999,-30.,0.)
    _,_,_,_,info=env.step({n:(0,0) for n in NAMES})
    assert not info['per_agent_success'][NAMES[0]]
