"""Stage 4 independent-oracle sensor validation. Production defects are not repaired here."""
import json, math
from pathlib import Path
import numpy as np
import pytest
import torch

from bcod_sim.config.models import World
from bcod_sim.core.errors import ExternalDataCoverageError, InvalidMediumError
from bcod_sim.frames.transforms import rpy_to_quaternion
from bcod_sim.logging.recorder import _encode_sensor_values
from bcod_sim.sensors.base import SensorConfig, SensorContext, stable_noise
from bcod_sim.sensors.configs import GPSErrorModel, IMUErrorModel
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.kinematics import mount_kinematics
from bcod_sim.sensors.lidar import LiDAR
from bcod_sim.sensors.scheduler import SensorScheduler
from bcod_sim.sensors.sonar import Sonar
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.world.bathymetry import RasterBathymetry
from bcod_sim.world.environment import Environment, Medium
from bcod_sim.world.world import ParametricWorld

D=torch.float64
def t(x): return torch.tensor(x,dtype=D)
def cfg(kind="gps",**kw):
    base=dict(instance_id=kind,sensor_type=kind,env_id=0,owner_vessel_id=0,mount_frd_m=(0,0,0),mount_q_to_frd=(1,0,0,0),rate_hz=10,latency_steps=0,noise_std=0,seed=19,version="1",source="stage4")
    base.update(kw);return SensorConfig(**base)
def world(*,entities=(),bath=None,current=None,wind=None,waves=None,boundary=None):
    raw={"source":{"kind":"parametric"},"environment":{"current":current or {"kind":"uniform","ned_mps":[0,0,0]},"wind":wind or {"kind":"uniform","ned_mps":[0,0,0]},"waves":waves or {"kind":"calm"},"visibility_m":1000},"static_entities":entities}
    if bath:raw["bathymetry"]=bath
    if boundary:raw["boundary"]=boundary
    return ParametricWorld(World.model_validate(raw),0)
def ctx(w,position=(0,0,0),q=(1,0,0,0),nu=(0,0,0,0,0,0),time=0,la=(0,0,0),aa=(0,0,0)):
    return SensorContext(VesselState(t(position),t(q),t(nu)),w,time,t(la),t(aa))

@pytest.mark.parametrize("mount",[(1,0,0),(0,2,0),(0,0,3),(1,2,3)])
def test_SENS_001_mount_translation(mount):
    k=mount_kinematics(cfg(mount_frd_m=mount),ctx(world(),position=(4,5,6)))
    assert k.position_ned_m.tolist()==pytest.approx([4+mount[0],5+mount[1],6+mount[2]],abs=1e-12)

def test_SENS_002_003_mount_rotation_and_vessel_pose_composition():
    q=rpy_to_quaternion(.3,-.2,math.pi/2); c=ctx(world(),position=(4,-2,1),q=q)
    k=mount_kinematics(cfg(mount_frd_m=(2,0,0),mount_q_to_frd=(math.sqrt(.5),0,0,math.sqrt(.5))),c)
    # Independent quaternion matrix oracle for body x under roll/pitch/yaw.
    cr,sr,cp,sp,cy,sy=math.cos(.3),math.sin(.3),math.cos(-.2),math.sin(-.2),0,1
    body_x=np.array([cy*cp,sy*cp,-sp]); expected=np.array([4,-2,1])+2*body_x
    assert k.position_ned_m.numpy()==pytest.approx(expected,abs=1e-12)
    local_x=t((1,0,0)); from bcod_sim.frames.tensor import rotate_body_to_world
    beam=rotate_body_to_world(local_x,k.q_mount_to_ned)
    assert abs(float(torch.linalg.vector_norm(beam))-1)<1e-12

def test_SENS_004_005_006_007_008_fov_range_nearest_occlusion_no_return():
    sphere=lambda name,r,a=0:{"id":name,"position_ned_m":[r*math.cos(a),r*math.sin(a),0],"shape":{"kind":"sphere","radius_m":.1}}
    sensor=LiDAR(cfg("lidar"),min_range_m=1,max_range_m=10,fov_rad=math.pi/2,ray_count=3)
    p=sensor.sample(ctx(world(entities=[sphere("near",3),sphere("far",6),sphere("edge",5,math.pi/4),sphere("outside",5,math.pi/4+1e-4)])),sample_step=0)
    assert p.values["hit"].tolist()==[False,True,True]
    assert p.values["range_m"][1].item()==pytest.approx(2.9,abs=1e-10)
    empty=sensor.sample(ctx(world()),sample_step=0)
    assert not empty.values["hit"].any() and torch.equal(empty.values["range_m"],torch.full((3,),10.,dtype=D))
    for center,expected in ((1.1,1.0),(10.1,10.0),(.9,10.0),(10.2,10.0)):
        one=LiDAR(cfg("lidar"),min_range_m=1,max_range_m=10,fov_rad=0,ray_count=1)
        out=one.sample(ctx(world(entities=[sphere("x",center)])),sample_step=0)
        assert out.values["range_m"].item()==pytest.approx(expected,abs=1e-9)

@pytest.mark.parametrize("bath,expected",[
 ({"kind":"flat","bottom_ned_z_m":10,"vertical_datum":"MSL"},9),
 ({"kind":"plane","origin_ned_m":[0,0,0],"bottom_at_origin_ned_z_m":10,"north_slope":.2,"east_slope":0,"vertical_datum":"MSL"},9)])
def test_SENS_010_011_bathymetric_ranges(bath,expected):
    s=Sonar(cfg("sonar"),min_range_m=0,max_range_m=30,fov_rad=0,beam_count=1)
    assert s.sample(ctx(world(bath=bath),position=(0,0,1)),sample_step=0).values["range_m"].item()==pytest.approx(expected,abs=1e-9)

def test_SENS_012_015_raster_and_authoritative_consistency():
    w=world(bath={"kind":"flat","bottom_ned_z_m":10,"vertical_datum":"MSL"})
    w.bathymetry=RasterBathymetry(t(((8,10),(12,14))),(-1,-1),(2,2),"MSL")
    e=Environment(w); bottom=e.sample("bathymetry.bottom_z",t((0,0,1)),0).value.item()
    assert bottom==pytest.approx(11)
    s=Sonar(cfg("sonar"),min_range_m=0,max_range_m=30,fov_rad=0,beam_count=1)
    assert s.sample(ctx(w,position=(0,0,1)),sample_step=0).values["range_m"].item()==pytest.approx(10,abs=1e-8)
    assert e.sample("bathymetry.normal",t((0,0,1)),0).value.shape==(1,3)

def test_SENS_014_bathymetry_only_sonar_ignores_obstacle_by_contract():
    obstacle={"id":"submerged","position_ned_m":[0,0,5],"shape":{"kind":"sphere","radius_m":1}}
    w=world(entities=[obstacle],bath={"kind":"flat","bottom_ned_z_m":10,"vertical_datum":"MSL"})
    s=Sonar(cfg("sonar"),min_range_m=0,max_range_m=30,fov_rad=0,beam_count=1)
    assert s.sample(ctx(w,position=(0,0,1)),sample_step=0).values["range_m"].item()==pytest.approx(9)

def test_SENS_020_gps_truth_mount_velocity_and_time():
    origin=(.5,-1.,12.);g=GPS(cfg("gps",mount_frd_m=(0,2,0)),origin_wgs84_rad_m=origin)
    p=g.sample(ctx(world(),nu=(1,0,0,0,0,3),time=2.5),sample_step=25)
    from bcod_sim.frames.geodesy import geodetic_to_ned
    assert geodetic_to_ned(*p.values["wgs84_lat_lon_alt"].tolist(),origin)==pytest.approx((0,2,0),abs=1e-6)
    assert p.values["velocity_ned_mps"].tolist()==pytest.approx((-5,0,0))

def test_SENS_021_gps_error_statistics_predeclared_n5000():
    model=GPSErrorModel(2,.5,(1,-1,.25),(0,0,0));g=GPS(cfg("gps"),origin_wgs84_rad_m=(0,0,0),error_model=model);c=ctx(world())
    velocities=np.array([g.sample(c,sample_step=i).values["velocity_ned_mps"].numpy() for i in range(5000)])
    assert np.max(abs(velocities.mean(0)))<.025 and np.max(abs(velocities.std(0,ddof=1)-.5))<.025

def test_SENS_022_imu_truth_gravity_rotation_and_lever_arm():
    imu=IMU(cfg("imu",mount_frd_m=(0,1,0)));p=imu.sample(ctx(world(),nu=(2,0,0,0,0,3),la=(1,0,0)),sample_step=0)
    assert p.values["specific_force_mount_mps2"].tolist()==pytest.approx((1,-3,-9.80665),abs=1e-10)
    assert p.values["angular_rate_mount_radps"].tolist()==pytest.approx((0,0,3),abs=1e-12)

def test_SENS_023_imu_random_walk_adjacent_correlation():
    imu=IMU(cfg("imu",rate_hz=100),error_model=IMUErrorModel(accel_random_walk_mps2=.1));c=ctx(world())
    values=[]
    for n in range(5000):values.append(float(imu.sample(c,sample_step=n).values["specific_force_mount_mps2"][0]))
    values=np.array(values)
    # A cumulative walk is strongly correlated and increments have sigma_rw*sqrt(dt)=0.01.
    assert np.corrcoef(values[:-1],values[1:])[0,1]>.99
    assert np.std(np.diff(values),ddof=1)==pytest.approx(.01,rel=.04)

def test_SENS_024_025_026_environment_truth_position_and_sample_time():
    current={"kind":"linear","origin_ned_m":[0,0,0],"base_ned_mps":[1,2,3],"gradient_per_s":[[.1,0,0],[0,-.05,0],[0,0,.2]]}
    w=world(current=current,wind={"kind":"sinusoidal","mean_ned_mps":[1,0,0],"amplitude_ned_mps":[2,0,0],"period_s":4})
    e=Environment(w);p=t((10,20,5));assert e.sample("water.current",p,7).value[0].tolist()==pytest.approx((2,1,4))
    assert e.sample("atmosphere.wind",t((0,0,-1)),1).value[0,0].item()==pytest.approx(3)
    k=mount_kinematics(cfg(mount_frd_m=(10,0,0)),ctx(w,position=(1,0,0)))
    assert e.sample("water.current",k.position_ned_m,0).value[0,0].item()==pytest.approx(2.1)

class Dummy:
    kind="physical";requirements=frozenset()
    def __init__(self,c):self.config=c
    def sample(self,c,*,sample_step):
        from bcod_sim.sensors.base import packet
        return packet(self.config,self.kind,sample_step,{"x":sample_step},{"x":"1"},{"x":"scalar"})

def test_SENS_030_031_032_034_035_rates_latency_warmup_async_dt_independence():
    def stream(dt):
        sensors=(Dummy(cfg("a",instance_id="a",rate_hz=10,latency_steps=2,warmup_s=.2)),Dummy(cfg("b",instance_id="b",rate_hz=5)))
        scheduler=SensorScheduler(sensors,master_dt_s=dt);out=[]
        for n in range(round(1/dt)+1):out.extend(scheduler.tick(n,{(0,0):ctx(world(),time=n*dt)}))
        return [(p.sensor_id,p.sample_time_s,p.delivery_time_s) for p in out]
    a=stream(.01);b=stream(.02)
    assert [(x,round(s,9)) for x,s,_ in a]==[(x,round(s,9)) for x,s,_ in b]
    assert all(d-s==(pytest.approx(.02) if x=="a" else pytest.approx(0)) for x,s,d in a)
    assert min(s for x,s,d in a if x=="a")==pytest.approx(.2)

def test_SENS_033_dropout_predeclared_n10000_tolerance_002():
    sensor=Dummy(cfg("d",dropout_probability=.3,rate_hz=100));scheduler=SensorScheduler((sensor,),master_dt_s=.01);kept=0
    w=world()
    for n in range(10000):kept+=len(scheduler.tick(n,{(0,0):ctx(w,time=n*.01)}))
    assert abs((1-kept/10000)-.3)<=.02

def test_SENS_040_041_042_replay_seed_and_geometry_independence():
    c=ctx(world(entities=[{"id":"x","position_ned_m":[5,0,0],"shape":{"kind":"sphere","radius_m":1}}]))
    def sample(seed):return LiDAR(cfg("lidar",seed=seed,noise_std=.1),min_range_m=0,max_range_m=10,fov_rad=0,ray_count=1).sample(c,sample_step=4)
    a,b,z=sample(1),sample(1),sample(2)
    assert torch.equal(a.values["range_m"],b.values["range_m"]) and not torch.equal(a.values["range_m"],z.values["range_m"])
    assert a.values["hit"].tolist()==z.values["hit"].tolist()==[True]

def test_SENS_060_061_062_063_064_persistence_encoding(tmp_path):
    values={"scalar":1.25,"record":{"name":"a","v":2},"detections":[{"id":"a"},{"id":"b"}],"image":torch.arange(5000,dtype=torch.float32).reshape(50,100)}
    encoded=_encode_sensor_values(tmp_path,"sensor",3,values);path=tmp_path/encoded["image"]["array_artifact"]
    restored=np.load(path,allow_pickle=False)
    assert encoded["scalar"]==1.25 and encoded["record"]=={"name":"a","v":2} and encoded["detections"][1]["id"]=="b"
    assert restored.dtype==np.float32 and restored.shape==(50,100) and np.array_equal(restored,values["image"].numpy())

def test_SENS_070_071_medium_and_coverage_fail_closed():
    w=world(bath={"kind":"flat","bottom_ned_z_m":10,"vertical_datum":"MSL"},boundary={"id":"b","min_ned_m":[-5,-5,-5],"max_ned_m":[5,5,20]});e=Environment(w)
    assert e.sample("water.current",t((0,0,1)),0).valid
    for p in ((0,0,-1),(0,0,11)):
        with pytest.raises(InvalidMediumError):e.sample("water.current",t(p),0)
    # Public domain API classifies coverage before provider sampling and raises a typed domain-query error.
    with pytest.raises(InvalidMediumError,match="OUT_OF_COVERAGE"):e.sample("weather.visibility",t((6,0,0)),0)
