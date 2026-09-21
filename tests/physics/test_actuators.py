import math

import pytest
import torch

from bcod_sim.actuators.allocation import ActuatorBank, WrenchContribution, allocate_fixed_thrusters, reduce_wrenches
from bcod_sim.actuators.autopilot import HeadingSpeedAutopilot, HighLevelCommand
from bcod_sim.actuators.azipod import Azipod, AzipodCommand
from bcod_sim.actuators.base import ActuatorConfig, ActuatorState, Bounds
from bcod_sim.actuators.generic import ScalarCommand, ScalarWrenchActuator, VectorCommand, VectorWrenchActuator
from bcod_sim.actuators.rudder import PropellerRudder, PropellerRudderCommand
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.core.errors import CommandBoundsError, DuplicateIdentityError, PhysicalValidationError
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.state.vessel_state import VesselState


def config(name="port", *, env=0, owner=0, y=0.0, policy="error", q=(1, 0, 0, 0), **changes):
    values = dict(instance_id=name, env_id=env, owner_vessel_id=owner, mount_frd_m=(0, y, 0),
                  mount_q_to_frd=q, thrust_bounds_n=Bounds(-100, 100), command_bounds_policy=policy)
    values.update(changes)
    return ActuatorConfig(**values)


def mirrored_pair(*, env=0, owner=0):
    return (FixedThruster(config("port", env=env, owner=owner, y=-1)),
            FixedThruster(config("starboard", env=env, owner=owner, y=1)))


def test_equal_mirrored_thrust_zero_yaw_and_differential_sign():
    port, starboard = mirrored_pair()
    a = port.step(ThrustCommand(20), ActuatorState(), 0.1).wrench_frd
    b = starboard.step(ThrustCommand(20), ActuatorState(), 0.1).wrench_frd
    assert a[0] + b[0] == 40
    assert a[5] + b[5] == pytest.approx(0)
    assert a[5] > 0 and b[5] < 0
    assert starboard.step(ThrustCommand(30), ActuatorState(), 0.1).wrench_frd[5] == -30


def test_mount_orientation_and_azipod_rudder_signs():
    yaw90 = (math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    side = FixedThruster(config(q=yaw90)).step(ThrustCommand(10), ActuatorState(), 0.1)
    assert side.wrench_frd[:3] == pytest.approx((0, 10, 0), abs=1e-12)
    pod = Azipod(config(y=1), Bounds(-math.pi/2, math.pi/2))
    pod_result = pod.step(AzipodCommand(10, math.pi/2), ActuatorState(), 0.1)
    assert pod_result.wrench_frd[:3] == pytest.approx((0, 10, 0), abs=1e-12)
    rudder = PropellerRudder(config(y=0), Bounds(-0.5, 0.5), side_force_gain=0.5)
    rudder_result = rudder.step(PropellerRudderCommand(10, 0.5), ActuatorState(), 0.1)
    assert rudder_result.wrench_frd[1] == pytest.approx(5 * math.sin(0.5))


def test_command_bounds_error_and_explicit_clamp_event():
    thruster = FixedThruster(config())
    with pytest.raises(CommandBoundsError):
        thruster.step(ThrustCommand(101), ActuatorState(), 0.1)
    with pytest.raises(CommandBoundsError):
        thruster.step(ThrustCommand(float("nan")), ActuatorState(), 0.1)
    clamping = FixedThruster(config(policy="clamp_with_event"))
    result = clamping.step(ThrustCommand(120), ActuatorState(), 0.1)
    assert result.state.thrust_n == 100
    assert len(result.events) == 1
    assert (result.events[0].requested, result.events[0].applied) == (120, 100)


def test_actuator_dynamics_deadband_rate_and_power():
    thruster = FixedThruster(config(thrust_rate_limit_nps=10, deadband_n=2,
                                    power_coefficient_w_per_n=3))
    first = thruster.step(ThrustCommand(50), ActuatorState(), 0.1)
    assert first.state.thrust_n == pytest.approx(1)
    assert first.power_w == pytest.approx(3)
    second = thruster.step(ThrustCommand(1), first.state, 0.1)
    assert second.state.thrust_n == pytest.approx(0)
    pod = Azipod(config(), Bounds(-1, 1), azimuth_rate_limit_radps=2)
    result = pod.step(AzipodCommand(10, 0.5), ActuatorState(), 0.1)
    assert result.state.steering_rad == pytest.approx(0.2)


def test_bank_owner_isolation_and_order_independence():
    a = FixedThruster(config("a", env=0, owner=0))
    b = FixedThruster(config("a", env=1, owner=0))
    c = FixedThruster(config("b", env=0, owner=0))
    keys = ((0, 0, "a"), (1, 0, "a"), (0, 0, "b"))
    commands = dict(zip(keys, (ThrustCommand(10), ThrustCommand(20), ThrustCommand(30))))
    states = {key: ActuatorState() for key in keys}
    forward = ActuatorBank((a, b, c)).step(commands, states, 0.1)
    reverse = ActuatorBank((c, b, a)).step(commands, states, 0.1)
    assert forward.wrenches_by_owner == reverse.wrenches_by_owner
    assert forward.wrenches_by_owner[(0, 0)][0] == 40
    assert forward.wrenches_by_owner[(1, 0)][0] == 20
    assert len(ActuatorBank((a, b, c)).schema_groups[FixedThruster]) == 3
    with pytest.raises(PhysicalValidationError):
        ActuatorBank((a, b)).step({keys[0]: ThrustCommand(10)}, {keys[0]: ActuatorState()}, 0.1)
    with pytest.raises(DuplicateIdentityError):
        ActuatorBank((a, a))


def test_reduction_sorted_and_duplicate_rejected():
    items = (WrenchContribution(0, 0, "z", (1e16, 0, 0, 0, 0, 0)),
             WrenchContribution(0, 0, "a", (1, 0, 0, 0, 0, 0)),
             WrenchContribution(0, 0, "m", (-1e16, 0, 0, 0, 0, 0)))
    assert reduce_wrenches(items) == reduce_wrenches(tuple(reversed(items)))
    assert reduce_wrenches(items)[(0, 0)][0] == 1
    with pytest.raises(DuplicateIdentityError):
        reduce_wrenches((items[0], items[0]))


def test_high_level_heading_speed_allocation_and_unreachable_request():
    pair = mirrored_pair()
    autopilot = HeadingSpeedAutopilot(20, 10)
    commands = autopilot.commands(HighLevelCommand(2, 0.2), current_speed_mps=1,
                                  current_heading_rad=0, thrusters=pair)
    bank = ActuatorBank(pair)
    keyed = {(0, 0, name): command for name, command in commands.items()}
    result = bank.step(keyed, {key: ActuatorState() for key in keyed}, 0.1)
    assert result.wrenches_by_owner[(0, 0)][0] == pytest.approx(20)
    assert result.wrenches_by_owner[(0, 0)][5] == pytest.approx(2)
    too_fast = autopilot.commands(HighLevelCommand(20, 0), current_speed_mps=0,
                                  current_heading_rad=0, thrusters=pair)
    with pytest.raises(CommandBoundsError):
        bank.step({(0, 0, name): command for name, command in too_fast.items()},
                  {key: ActuatorState() for key in keyed}, 0.1)
    with pytest.raises(PhysicalValidationError):
        allocate_fixed_thrusters((pair[0],), surge_n=10, yaw_nm=0)


def test_generic_scalar_and_vector_contract():
    scalar = ScalarWrenchActuator(config(), Bounds(-2, 2), (1, 0, 0, 0, 0, 0))
    assert scalar.step(ScalarCommand(2), ActuatorState(), 0.1).wrench_frd[0] == 2
    vector = VectorWrenchActuator(config(), (Bounds(-2, 2), Bounds(-2, 2)),
                                  ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)))
    assert vector.step(VectorCommand((1, -1)), ActuatorState(), 0.1).wrench_frd[:2] == (1, -1)


def test_mirrored_commands_produce_mirrored_trajectories():
    pair = mirrored_pair()
    dtype = torch.float64
    z6 = torch.zeros(6, dtype=dtype)
    model = Plant6(MassProperties(10, torch.zeros(3, dtype=dtype), torch.diag(torch.tensor([4., 5., 6.], dtype=dtype)),
                                  torch.zeros((6, 6), dtype=dtype)),
                   Damping(z6, z6), Hydrostatics(10 * 9.80665, torch.zeros(3, dtype=dtype)),
                   OperatingEnvelope(torch.ones(6, dtype=dtype) * 1e6))
    states = [VesselState(torch.zeros(3, dtype=dtype), torch.tensor([1., 0, 0, 0], dtype=dtype), z6.clone()) for _ in range(2)]
    for _ in range(10):
        for i, thrusts in enumerate(((30, 10), (10, 30))):
            contribution = [pair[j].step(ThrustCommand(thrusts[j]), ActuatorState(), 0.01).wrench_frd for j in range(2)]
            total = tuple(math.fsum(w[k] for w in contribution) for k in range(6))
            external = {name: z6.clone() for name in EXTERNAL_TERMS}
            external["propulsion"] = torch.tensor(total, dtype=dtype)
            states[i] = model.step(states[i], external, 0.01).state
    assert states[0].position_ned[0].item() == pytest.approx(states[1].position_ned[0].item(), abs=1e-12)
    assert states[0].position_ned[1].item() == pytest.approx(-states[1].position_ned[1].item(), abs=1e-12)
    assert states[0].nu_body[5].item() == pytest.approx(-states[1].nu_body[5].item(), abs=1e-12)
