import math

import numpy as np
import pytest

from bcod_sim.actuators.physical import (ActuatorPipeline, ActuatorSet, ControlAllocator,
                                          SpeedHeadingController, VesselMotion)


def thruster(name, position, direction=(1, 0, 0), kind="fixed_thruster"):
    return {"id": name, "type": kind,
            "pose": {"position_frd": position, "direction_frd": direction},
            "propulsion": {"model": "power_law", "max_forward_n": 10,
                           "max_reverse_n": 10, "exponent": 1}}


def test_twin_geometry_and_fixed_allocation():
    system = ActuatorSet([thruster("p", (-1, -1, 0)), thruster("s", (-1, 1, 0))])
    motion = VesselMotion()
    assert system.predict_wrench({"p": (1,), "s": (1,)}, motion).wrench_frd == (20, 0, 0, 0, 0, 0)
    assert system.predict_wrench({"p": (1,), "s": (-1,)}, motion).wrench_frd[5] == 20
    result = ControlAllocator(system).allocate((10, 0, 0, 0, 0, 0), motion, .1)
    assert result.success and result.solver_status.startswith("fixed:")
    assert np.linalg.norm(result.residual) < 1e-5
    impossible = ControlAllocator(system).allocate((0, 100, 0, 0, 0, 0), motion, .1)
    assert impossible.success and impossible.residual[1] == 100


def test_tunnel_and_azimuth_orientation_and_wrapping():
    tunnel = ActuatorSet([thruster("bow", (2, 0, 0), (0, 1, 0), "tunnel_thruster")])
    assert tunnel.predict_wrench({"bow": (1,)}, VesselMotion()).wrench_frd == (0, 10, 0, 0, 0, 20)
    pod = thruster("pod", (0, 0, 0), kind="azimuth_thruster")
    pod["steering"] = {"continuous": True}
    pod["dynamics"] = {"steering_rate_radps": math.radians(10)}
    system = ActuatorSet([pod])
    for angle, expected in ((0, (10, 0)), (math.pi/2, (0, 10)), (math.pi, (-10, 0)), (-math.pi/2, (0, -10))):
        force = system.predict_wrench({"pod": (1, angle)}, VesselMotion()).devices[0].force_frd
        assert force[:2] == pytest.approx(expected, abs=1e-12)
    system.by_id["pod"].state.angle_rad = math.radians(179)
    next_load = system.predict_wrench({"pod": (1, math.radians(-179))}, VesselMotion(), dt=.1, commit=True).devices[0]
    assert next_load.angle_rad == pytest.approx(math.radians(180), abs=1e-8)
    assert next_load.rate_limited


def test_rudder_flow_slipstream_and_zero_authority():
    rudder = {"id": "rudder", "type": "rudder", "pose": {"position_frd": (-2, 0, 0)},
              "geometry": {"area_m2": .2, "aspect_ratio": 2, "slipstream_disk_area_m2": .5},
              "steering": {"angle_bounds_rad": (-.6, .6)}, "inflow_sources": ["prop"]}
    system = ActuatorSet([thruster("prop", (-2, 0, 0)), rudder])
    quiet = system.predict_wrench({"prop": (0,), "rudder": (.2,)}, VesselMotion()).devices[1]
    moving = system.predict_wrench({"prop": (0,), "rudder": (.2,)}, VesselMotion((1, 0, 0))).devices[1]
    powered = system.predict_wrench({"prop": (1,), "rudder": (.2,)}, VesselMotion((1, 0, 0))).devices[1]
    assert quiet.force_frd == (0, 0, 0)
    assert moving.force_frd[1] > 0
    assert powered.force_frd[1] > moving.force_frd[1]
    alone = ActuatorSet([{**rudder, "inflow_sources": []}])
    authority = alone.authority_report(VesselMotion())
    assert authority["0.0"]["max_yaw_nm"] == pytest.approx(0)
    assert authority["2.0"]["max_yaw_nm"] > 0


def test_pipeline_modes_deterministic_and_antiwindup():
    specs = [thruster("p", (-1, -1, 0)), thruster("s", (-1, 1, 0))]
    def run():
        system = ActuatorSet(specs)
        controller = SpeedHeadingController(100, 40, 10, 5)
        pipeline = ActuatorPipeline(system, controller)
        motion = VesselMotion()
        direct = pipeline.step("DIRECT_ACTUATOR", {"p": (1,), "s": (1,)}, motion, .02)
        wrench = pipeline.step("DESIRED_WRENCH", (0, 0, 0, 0, 0, 10), motion, .02)
        high = pipeline.step("DESIRED_SPEED_HEADING", (1, .2), motion, .1)
        return direct.wrench_frd, wrench.wrench_frd, high.wrench_frd
    assert run() == run()


def test_mixed_nonlinear_allocation_and_rate_bounds():
    port = thruster("port", (-1, -1, 0), kind="azimuth_thruster")
    starboard = thruster("starboard", (-1, 1, 0), kind="azimuth_thruster")
    for pod in (port, starboard):
        pod["steering"] = {"angle_bounds_rad": (-math.pi, math.pi)}
    bow = thruster("bow", (2, 0, 0), (0, 1, 0), "tunnel_thruster")
    system = ActuatorSet([port, starboard, bow])
    result = ControlAllocator(system).allocate((5, 4, 0, 0, 0, 3), VesselMotion(), .1)
    assert result.success and result.solver_status.startswith("nonlinear:")
    assert np.linalg.norm(np.asarray(result.residual)[[0, 1, 5]]) < .01
    assert all(np.all(np.isfinite(command)) for command in result.commands.values())
