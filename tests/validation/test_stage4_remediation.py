import json, math
from pathlib import Path
import numpy as np
import pytest
import torch

from bcod_sim.sensors.base import packet
from bcod_sim.sensors.configs import IMUErrorModel
from bcod_sim.sensors.imu import IMU
from bcod_sim.sensors.scheduler import SensorScheduler
from bcod_sim.sensors.gps import GPS
from bcod_sim.sensors.lidar import LiDAR
from bcod_sim.sensors.sonar import Sonar
from bcod_sim.sensors.kinematics import mount_kinematics
from bcod_sim.frames.transforms import rpy_to_quaternion
from bcod_sim.frames.geodesy import geodetic_to_ned
from bcod_sim.logging.recorder import _encode_sensor_values
from bcod_sim.logging.recorder import RunRecorder
from bcod_sim.logging.recorder import _json_safe
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.core.lifecycle import DirectAction
from bcod_sim.collision.shapes import Sphere
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from tests.validation.test_stage4_sensor_validation import cfg, ctx, world


def drift(imu, count=1000):
    context=ctx(world()); result=[]
    for step in range(count):
        result.append(imu.sample(context,sample_step=step).values["specific_force_mount_mps2"].numpy())
    return np.asarray(result)-np.array((0,0,-9.80665))


def test_SENS_023A_zero_sigma_has_no_drift():
    assert np.array_equal(drift(IMU(cfg("imu"),error_model=IMUErrorModel()),20),np.zeros((20,3)))


def test_SENS_023B_C_same_seed_replays_and_different_seed_changes():
    model=IMUErrorModel(accel_random_walk_mps2=.2,gyro_random_walk_radps=.05)
    a=drift(IMU(cfg("imu",seed=7),error_model=model),100)
    b=drift(IMU(cfg("imu",seed=7),error_model=model),100)
    c=drift(IMU(cfg("imu",seed=8),error_model=model),100)
    assert np.array_equal(a,b) and not np.array_equal(a,c)


def test_SENS_023D_E_increment_statistics_and_cumulative_correlation():
    sigma=.3;rate=100;values=drift(IMU(cfg("imu",rate_hz=rate),error_model=IMUErrorModel(accel_random_walk_mps2=sigma)),5000)[:,0]
    increments=np.diff(np.concatenate(([0.],values)))
    assert increments.mean()==pytest.approx(0,abs=.001)
    assert increments.std(ddof=1)==pytest.approx(sigma/math.sqrt(rate),rel=.04)
    assert np.corrcoef(values[:-1],values[1:])[0,1]>.99


def test_SENS_023F_scheduler_reset_restores_sequence():
    imu=IMU(cfg("imu",rate_hz=10),error_model=IMUErrorModel(accel_random_walk_mps2=.2));context=ctx(world())
    first=SensorScheduler((imu,),master_dt_s=.1).tick(0,{(0,0):context})[0].values["specific_force_mount_mps2"].clone()
    second=SensorScheduler((imu,),master_dt_s=.1).tick(0,{(0,0):context})[0].values["specific_force_mount_mps2"]
    assert torch.equal(first,second)


def test_SENS_023G_sampling_rate_scales_increment_by_sqrt_dt():
    model=IMUErrorModel(accel_random_walk_mps2=.2)
    fast=np.diff(drift(IMU(cfg("imu",rate_hz=100),error_model=model),4000)[:,0]).std(ddof=1)
    slow=np.diff(drift(IMU(cfg("imu",rate_hz=25),error_model=model),4000)[:,0]).std(ddof=1)
    assert slow/fast==pytest.approx(2,rel=.06)


def test_SENS_023H_elapsed_sample_time_and_snapshot_restore():
    model=IMUErrorModel(accel_random_walk_mps2=.2)
    sensor=IMU(cfg("imu",rate_hz=10),error_model=model)
    first=sensor.sample(ctx(world(),time=0),sample_step=0)
    saved=sensor.snapshot()
    later=sensor.sample(ctx(world(),time=.3),sample_step=3).values["specific_force_mount_mps2"].clone()
    sensor.restore(saved)
    replay=sensor.sample(ctx(world(),time=.3),sample_step=3).values["specific_force_mount_mps2"]
    assert torch.equal(later,replay)
    from bcod_sim.sensors.base import stable_noise
    expected_increment=stable_noise(sensor.config,3,(3,),dtype=torch.float64,device=torch.device("cpu"),
        stream="accel_random_walk_increment",std=.2*math.sqrt(.3))
    assert torch.allclose(later-first.values["specific_force_mount_mps2"],expected_increment,atol=1e-15)


def test_SENS_036A_F_freshness_retention_age_latency_expiry():
    class Dummy:
        kind="physical";requirements=frozenset()
        def __init__(self,config):self.config=config
        def sample(self,context,*,sample_step):return packet(self.config,self.kind,sample_step,{"value":sample_step},{"value":"1"},{"value":"scalar"})
    sensor=Dummy(cfg("fresh",rate_hz=2,latency_steps=2,max_age_s=.35));scheduler=SensorScheduler((sensor,),master_dt_s=.1);context_world=world();latest=None;observed=[]
    for step in range(9):
        delivered=scheduler.tick(step,{(0,0):ctx(context_world,time=step*.1)})
        if delivered:latest=delivered[-1]
        if latest:observed.append((step,latest.values["value"],latest.freshness(step*.1,sensor.config.max_age_s)))
    # Sample 0 arrives at .2, is retained, ages from original sample time, expires after .35,
    # and sample 5 arrives at .7 resetting age to .2 (its latency).
    assert observed[0][0:2]==(2,0) and observed[0][2]["age_s"]==pytest.approx(.2)
    assert next(row for row in observed if row[0]==4)[2]["stale"] is True
    assert next(row for row in observed if row[0]==7)[1]==5
    assert next(row for row in observed if row[0]==7)[2]["age_s"]==pytest.approx(.2)
    assert latest.freshness(100,None)["stale"] is False


def test_SENS_081_binary_and_structured_replay_roundtrip(tmp_path):
    values={"scalar":1.25,"structured":{"items":[{"id":"a","valid":True}]},
            "array":torch.arange(6000,dtype=torch.float32).reshape(60,100)}
    a=_encode_sensor_values(tmp_path/"a","multi",4,values)
    b=_encode_sensor_values(tmp_path/"b","multi",4,values)
    assert json.dumps({k:v for k,v in a.items() if k!="array"},sort_keys=True)==json.dumps({k:v for k,v in b.items() if k!="array"},sort_keys=True)
    aa=np.load(tmp_path/"a"/a["array"]["array_artifact"],allow_pickle=False)
    bb=np.load(tmp_path/"b"/b["array"]["array_artifact"],allow_pickle=False)
    assert aa.dtype==bb.dtype==np.float32 and aa.shape==bb.shape==(60,100) and np.array_equal(aa,bb)


class CurrentMagnitude:
    kind="physical";requirements=frozenset({"water.current"})
    def __init__(self,config):self.config=config
    def sample(self,context,*,sample_step):
        mount=mount_kinematics(self.config,context)
        value=context.environment.sample("water.current",mount.position_ned_m,context.sim_time_s).value[0]
        return packet(self.config,self.kind,sample_step,{"current_ned_mps":value},
                      {"current_ned_mps":"m/s"},{"current_ned_mps":"NED"})


def _ray_sphere(origin,direction,center,radius):
    delta=np.asarray(origin)-np.asarray(center);direction=np.asarray(direction)
    b=float(delta@direction);c=float(delta@delta-radius*radius);disc=b*b-c
    if disc<0:return None
    a=-b-math.sqrt(disc);z=-b+math.sqrt(disc)
    return a if a>=0 else z if z>=0 else None


def test_SENS_080_moving_five_sensor_independent_oracles(record_property):
    obstacle={"id":"target","position_ned_m":[5,0,1],"shape":{"kind":"sphere","radius_m":1}}
    current={"kind":"linear","origin_ned_m":[0,0,0],"base_ned_mps":[1,0,0],
             "gradient_per_s":[[.1,0,0],[0,.2,0],[0,0,0]]}
    bath={"kind":"plane","origin_ned_m":[0,0,0],"bottom_at_origin_ned_z_m":10,
          "north_slope":.1,"east_slope":.05,"vertical_datum":"MSL"}
    w=world(entities=[obstacle],bath=bath,current=current)
    origin=(.5,-1.,12.)
    sensors=(
        LiDAR(cfg("lidar",instance_id="lidar",rate_hz=10),min_range_m=0,max_range_m=20,fov_rad=0,ray_count=1),
        Sonar(cfg("sonar",instance_id="sonar",rate_hz=10),min_range_m=0,max_range_m=30,fov_rad=0,beam_count=1),
        GPS(cfg("gps",instance_id="gps",rate_hz=10,mount_frd_m=(.5,0,0)),origin_wgs84_rad_m=origin),
        IMU(cfg("imu",instance_id="imu",rate_hz=10,mount_frd_m=(0,1,0))),
        CurrentMagnitude(cfg("current",instance_id="current",rate_hz=10,mount_frd_m=(.25,0,0))),
    )
    scheduler=SensorScheduler(sensors,master_dt_s=.1);seen={name:[] for name in ("lidar","sonar","gps","imu","current")}
    maximum_error={name:0.0 for name in seen}
    for step in range(6):
        time=step*.1;yaw=.05*step;q=rpy_to_quaternion(0,0,yaw);position=(.2*step,0,1)
        context=ctx(w,position=position,q=q,nu=(2,0,0,0,0,.5),time=time,la=(0,1,0),aa=(0,0,0))
        packets=scheduler.tick(step,{(0,0):context});assert {p.sensor_id for p in packets}==set(seen)
        for p in packets:seen[p.sensor_id].append(p)
        by={p.sensor_id:p for p in packets}
        # Independent rigid transforms in yaw-only trajectory.
        c,s=math.cos(yaw),math.sin(yaw);forward=np.array((c,s,0.));body_y=np.array((-s,c,0.))
        lidar_origin=np.array(position);expected_hit=_ray_sphere(lidar_origin,forward,(5,0,1),1)
        actual_hit=bool(by["lidar"].values["hit"][0])
        assert actual_hit==(expected_hit is not None and expected_hit<=20)
        if actual_hit:
            measured=by["lidar"].values["range_m"][0].item()
            maximum_error["lidar"]=max(maximum_error["lidar"],abs(measured-expected_hit))
            assert measured==pytest.approx(expected_hit,abs=1e-9)
        expected_bottom=10+.1*position[0]+.05*position[1]
        sonar_error=abs(by["sonar"].values["range_m"][0].item()-(expected_bottom-position[2]))
        maximum_error["sonar"]=max(maximum_error["sonar"],sonar_error)
        assert sonar_error<=1e-8
        gps_mount=np.array(position)+.5*forward
        recovered_gps=np.array(geodetic_to_ned(*by["gps"].values["wgs84_lat_lon_alt"].tolist(),origin))
        maximum_error["gps"]=max(maximum_error["gps"],float(np.max(abs(recovered_gps-gps_mount))))
        assert recovered_gps==pytest.approx(gps_mount,abs=1e-6)
        # mount velocity = body v + omega x r = (2,.25,0), rotated to NED.
        expected_velocity=2*forward+.25*body_y
        assert by["gps"].values["velocity_ned_mps"].numpy()==pytest.approx(expected_velocity,abs=1e-10)
        # νdot + ω×v + ω×(ω×r) = (0,1,0)+(0,1,0)+(0,-.25,0).
        actual_force=by["imu"].values["specific_force_mount_mps2"].numpy()
        maximum_error["imu"]=max(maximum_error["imu"],float(np.max(abs(actual_force-np.array((0,1.75,-9.80665))))))
        assert actual_force==pytest.approx((0,1.75,-9.80665),abs=1e-10)
        assert by["imu"].values["angular_rate_mount_radps"].tolist()==pytest.approx((0,0,.5),abs=1e-12)
        current_mount=np.array(position)+.25*forward
        expected_current=(1+.1*current_mount[0],.2*current_mount[1],0)
        actual_current=by["current"].values["current_ned_mps"].numpy()
        maximum_error["current"]=max(maximum_error["current"],float(np.max(abs(actual_current-np.array(expected_current)))))
        assert actual_current==pytest.approx(expected_current,abs=1e-12)
        assert all(p.sample_time_s==pytest.approx(time) for p in packets)
    for name,error in maximum_error.items():record_property(f"{name}_max_abs_error",error)


class LargeStructured:
    kind="physical";requirements=frozenset()
    def __init__(self,config):self.config=config
    def sample(self,context,*,sample_step):
        array=torch.arange(5000,dtype=context.state.position_ned.dtype).reshape(50,100)+sample_step
        return packet(self.config,self.kind,sample_step,{"record":{"step":sample_step,"ok":True},"array":array},
                      {"record":"structured","array":"unitless"},{"record":"scalar","array":"sensor"})


def replay_engine(sensor_seed):
    registry=Registry();registry.register("vessel","vessel","1",{},"stage4")
    registry.register("task","waypoint","1",{"kind":"waypoint","agent_id":"agent","target_ned_m":[100,0,0],"radius_m":.1},"stage4")
    raw={"schema_version":1,"experiment":{"id":"stage4-replay","seed":3},
         "simulation":{"dynamics_mode":"full6","master_dt_s":.1,"dynamics_substeps":1,"policy_every_n_master_steps":1,"max_master_steps":20},
         "world":{"source":{"kind":"parametric"},"environment":{"current":{"kind":"uniform","ned_mps":[0,0,0]},"wind":{"kind":"uniform","ned_mps":[0,0,0]},"waves":{"kind":"calm"},"visibility_m":1000},"bathymetry":{"kind":"flat","bottom_ned_z_m":20,"vertical_datum":"MSL"},"static_entities":[{"id":"target","position_ned_m":[5,0,0],"shape":{"kind":"sphere","radius_m":1}}]},
         "vessels":[{"instance_id":"agent","definition":"vessel@1","controller":{"mode":"direct_actuator"},"spawn":{"ned_m":[0,0,0],"rpy_rad":[0,0,0]}}],
         "task":{"type":"waypoint@1","reward":{"individual_weight":1,"team_weight":0},"disabled_agent_behavior":"deactivate_keep_physical"},
         "logging":{"metrics":[],"states":False,"sensor_payloads":True,"queue_capacity":128,"backpressure":"block"}}
    resolved=resolve(raw,registry);v=lambda x:torch.tensor(x,dtype=torch.float64)
    plant=Plant6(MassProperties(10,v((0,0,0)),torch.diag(v((4,5,6))),torch.zeros((6,6),dtype=torch.float64)),
        Damping(v((0,)*6),v((0,)*6)),Hydrostatics(10*9.80665,v((0,0,0))),OperatingEnvelope(v((1e4,)*6),max_substep_s=.2))
    sensors=(
        LiDAR(cfg("lidar",instance_id="lidar",owner_vessel_id=1,rate_hz=10),min_range_m=0,max_range_m=20,fov_rad=0,ray_count=1),
        Sonar(cfg("sonar",instance_id="sonar",owner_vessel_id=1,rate_hz=5,latency_steps=1),min_range_m=0,max_range_m=30,fov_rad=0,beam_count=1),
        GPS(cfg("gps",instance_id="gps",owner_vessel_id=1,rate_hz=2,latency_steps=2,noise_std=.1,seed=sensor_seed,warmup_s=.5,dropout_probability=.25),origin_wgs84_rad_m=(0,0,0)),
        LargeStructured(cfg("large",instance_id="large",owner_vessel_id=1,rate_hz=5,latency_steps=1)),)
    engine=EpisodeEngine(resolved,(EpisodeVessel("agent",1,plant,Sphere(.2),(),sensors,ExplicitZeroLoads()),))
    return engine


def run_replay(engine,parent,run_id):
    initial=engine.reset();recorder=RunRecorder(parent,run_id=run_id,resolved=engine.resolved,scenario=engine.scenario,
        initial_frame=initial,project_root=Path(__file__).parents[2]);recorder.record_frame(initial);packets=list(initial.delivered_packets);frame=initial
    while not frame.terminated:
        frame=engine.step({"agent":DirectAction(())});packets.extend(frame.delivered_packets);recorder.record_frame(frame)
    root=recorder.close(final_frame=frame);return packets,root


def canonical_packets(packets):
    rows=[]
    for p in packets:
        values={}
        for key,value in p.values.items():
            if isinstance(value,torch.Tensor):values[key]={"dtype":str(value.dtype),"shape":tuple(value.shape),"bytes":value.cpu().numpy().tobytes()}
            else:values[key]=value
        rows.append((p.sensor_id,p.sample_time_s,p.delivery_time_s,p.validity,p.config_fingerprint,p.schema,values))
    return rows


def persisted(root):
    rows=[json.loads(line) for line in (root/"sensors"/"packets.jsonl").read_text().splitlines()]
    for row in rows:
        for value in row["measurement"].values():
            if isinstance(value,dict) and value.get("encoding")=="npy":
                array=np.load(root/"sensors"/value["array_artifact"],allow_pickle=False)
                value["loaded_dtype"],value["loaded_shape"],value["loaded_sha"] = str(array.dtype),list(array.shape),hash(array.tobytes())
    return rows


def test_SENS_081_engine_persistence_same_seed_and_changed_seed(tmp_path):
    packets_a,root_a=run_replay(replay_engine(31),tmp_path,"a")
    packets_b,root_b=run_replay(replay_engine(31),tmp_path,"b")
    packets_c,root_c=run_replay(replay_engine(32),tmp_path,"c")
    assert canonical_packets(packets_a)==canonical_packets(packets_b)
    saved_a,saved_b=persisted(root_a),persisted(root_b)
    assert saved_a==saved_b
    assert len(saved_a)==len(packets_a)
    for row, packet_value in zip(saved_a,packets_a):
        assert row["sensor_id"]==packet_value.sensor_id
        assert row["sample_time_s"]==packet_value.sample_time_s
        assert row["delivery_time_s"]==packet_value.delivery_time_s
        assert row["validity"]==packet_value.validity
        assert row["sensor_config_fingerprint"]==packet_value.config_fingerprint
        assert row["schema"]==_json_safe(packet_value.schema)
        assert row["provenance"]==_json_safe(packet_value.provenance)
        for name, original in packet_value.values.items():
            encoded=row["measurement"][name]
            if isinstance(original,torch.Tensor) and original.numel()>4096:
                reloaded=np.load(root_a/"sensors"/encoded["array_artifact"],allow_pickle=False)
                assert reloaded.dtype==original.numpy().dtype
                assert reloaded.shape==tuple(original.shape)
                assert np.array_equal(reloaded,original.numpy())
            else:
                assert encoded==_json_safe(original)
    assert all(row["sensor_config_fingerprint"] and row["schema"] and row["provenance"] for row in saved_a)
    arrays=[value for row in saved_a for value in row["measurement"].values() if isinstance(value,dict) and value.get("encoding")=="npy"]
    assert arrays and all(value["loaded_dtype"]=="float64" and value["loaded_shape"]==[50,100] for value in arrays)
    # Only the GPS seed changed: deterministic geometry schedules stay identical, stochastic values/dropout differ.
    deterministic=lambda rows:[(p.sensor_id,p.sample_time_s,p.delivery_time_s,p.values) for p in rows if p.sensor_id in {"lidar","sonar","large"}]
    assert canonical_packets([p for p in packets_a if p.sensor_id!="gps"])==canonical_packets([p for p in packets_c if p.sensor_id!="gps"])
    assert canonical_packets([p for p in packets_a if p.sensor_id=="gps"])!=canonical_packets([p for p in packets_c if p.sensor_id=="gps"])
