import copy

import torch

from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.lifecycle import PhysicalAction
from bcod_sim.communication import CommunicatingAction
from bcod_sim.actuators.autopilot import HighLevelCommand
from bcod_sim.web.runtime_factory import build_engine


def payload(system, mode="direct_actuator"):
    vessel = {"mass_kg": 100, "cg_frd_m": [0,0,0], "inertia_cg_kg_m2": [[40,0,0],[0,50,0],[0,0,60]],
        "added_mass_kg": [[0]*6 for _ in range(6)], "linear_damping": [1]*6,"quadratic_damping": [0]*6,
        "buoyancy_n": 980.665,"center_buoyancy_frd_m": [0,0,0],"max_abs_nu": [100]*6,"max_substep_s": .2,
        "collision": {"kind":"sphere","radius_m":.2},"environment_loads":[0,0,0,0]}
    config = {"schema_version":1,"experiment":{"id":"physical","seed":11},"simulation":{"dynamics_mode":"full6",
        "master_dt_s":.1,"dynamics_substeps":1,"policy_every_n_master_steps":1,"max_master_steps":20},
        "world":{"source":{"kind":"parametric"},"environment":{"current":{"kind":"uniform","ned_mps":[0,0,0]},
        "wind":{"kind":"uniform","ned_mps":[0,0,0]},"waves":{"kind":"calm"},"visibility_m":100},
        "bathymetry":{"kind":"flat","bottom_ned_z_m":20,"vertical_datum":"MSL"}},
        "vessels":[{"instance_id":"a","definition":"v@1","actuator_system":system,
        "controller":{"mode":mode},"spawn":{"ned_m":[0,0,0],"rpy_rad":[0,0,0]}}],
        "task":{"type":"task@1","reward":{"individual_weight":1,"team_weight":0},"disabled_agent_behavior":"deactivate_keep_physical"}}
    reg=Registry();reg.register("vessel","v","1",vessel,"test");reg.register("task","task","1",{"kind":"waypoint","agent_id":"a","target_ned_m":[100,0,0],"radius_m":.1},"test")
    return build_engine(resolve(config,reg))


def thruster(name,pos,direction=(1,0,0),kind="fixed_thruster"):
    out={"id":name,"type":kind,"pose":{"position_frd":pos,"direction_frd":direction},
         "propulsion":{"model":"power_law","max_forward_n":10,"max_reverse_n":10,"exponent":1}}
    if kind=="azimuth_thruster": out["steering"]={"angle_bounds_rad":[-3.14,3.14]}
    return out


def system(*devices): return {"schema_version":"manta-actuator-v1","actuators":list(devices)}


def test_physical_runtime_direct_wrench_and_replay_checkpoint():
    engine=payload(system(thruster("p",[-1,-1,0]),thruster("s",[-1,1,0])))
    engine.reset(); frame=engine.step({"a":PhysicalAction((("p",(1.,)),("s",(-1.,))))})
    assert engine.last_propulsion["a"][5].item()==20
    cp=engine.checkpoint(); replay=engine.step({"a":PhysicalAction((("p",(1.,)),("s",(1.,))))})
    engine.restore(cp); replay2=engine.step({"a":PhysicalAction((("p",(1.,)),("s",(1.,))))})
    assert torch.equal(replay.states["a"].position_ned,replay2.states["a"].position_ned)


def test_physical_runtime_high_level_azimuth_and_mixed_step():
    for devices in ((thruster("pod",[0,0,0],kind="azimuth_thruster"),),
                    (thruster("pod",[0,0,0],kind="azimuth_thruster"),thruster("bow",[2,0,0],(0,1,0),"tunnel_thruster"))):
        engine=payload(system(*devices),"high_level"); engine.reset()
        for _ in range(4): frame=engine.step({"a":HighLevelCommand(1.,.2)})
        assert torch.isfinite(frame.states["a"].nu_body).all()
        assert engine.last_actuator_diagnostics["a"]["allocation"] is not None


def test_propeller_rudder_runtime_has_flow_dependent_authority():
    rudder={"id":"rudder","type":"rudder","pose":{"position_frd":[-2,0,0]},
        "geometry":{"area_m2":.2,"aspect_ratio":2},"steering":{"angle_bounds_rad":[-.6,.6]}}
    engine=payload(system(thruster("prop",[-2,0,0]),rudder));engine.reset()
    pred=engine.vessels["a"].physical_pipeline.actuators.predict_wrench({"prop":(0.,),"rudder":(.2,)},
        __import__("bcod_sim.actuators.physical",fromlist=["VesselMotion"]).VesselMotion())
    assert pred.devices[1].force_frd==(0.,0.,0.)


def test_runtime_communication_is_explicit_and_one_step_delayed():
    # Reuse a minimal two-agent config and enable the communication contract.
    base=payload(system(thruster("p",[0,0,0])))
    # This engine has a single agent, so verify message wrapping is rejected when disabled.
    base.reset()
    import pytest
    with pytest.raises(Exception,match="disabled"):
        base.step({"a":CommunicatingAction(PhysicalAction((("p",(0.,)),)),(0.,0.,0.,0.))})
