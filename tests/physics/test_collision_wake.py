import math

import pytest
import torch

from bcod_sim.collision.broadphase import CollisionBody, broadphase_pairs
from bcod_sim.collision.narrowphase import detect_contacts
from bcod_sim.collision.shapes import Box, Capsule, Compound, CompoundChild, ConvexHull, Sphere, TriangleMesh
from bcod_sim.collision.solver import ContactMaterial, resolve_contacts
from bcod_sim.collision.world_adapter import world_collision_bodies
from bcod_sim.config.models import World
from bcod_sim.core.errors import CollisionSolverError, DuplicateIdentityError, PhysicalValidationError
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import PlanarEquilibrium, Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.world.wake import GaussianWakeEmitter, WakeField, WakeParameters
from bcod_sim.world.world import ParametricWorld


DTYPE = torch.float64


def vector(values):
    return torch.tensor(values, dtype=DTYPE)


def model(mode="full6", mass=10):
    zero = torch.zeros(6, dtype=DTYPE)
    return Plant6(MassProperties(mass, vector((0, 0, 0)), torch.diag(vector((4, 5, 6))),
                                 torch.zeros((6, 6), dtype=DTYPE)),
                  Damping(zero, zero), Hydrostatics(mass * 9.80665, vector((0, 0, 0))),
                  OperatingEnvelope(vector((1e6,)*6)), mode=mode,
                  planar_equilibrium=PlanarEquilibrium(0, 0, 0) if mode == "planar3" else None)


def vessel(name, x, u, *, shape=None, mode="full6", y=0, v=0, z=0, w=0, env=0):
    state = VesselState(vector((x, y, z)), vector((1, 0, 0, 0)), vector((u, v, w, 0, 0, 0)))
    return CollisionBody(env, name, shape or Sphere(1), state, model(mode))


def static(name, x, shape, *, y=0, z=0, env=0):
    return CollisionBody(env, name, shape, static_position_ned_m=(x, y, z))


def kinetic(body, state):
    return 0.5 * (state.nu_body @ body.plant.total_mass @ state.nu_body).item()


def test_elastic_equal_mass_conserves_energy_and_momentum():
    a, b = vessel("a", 0, 1), vessel("b", 1.8, -1)
    before = kinetic(a, a.state) + kinetic(b, b.state)
    result = resolve_contacts((b, a), material=ContactMaterial(restitution=1), dt_s=0.01)
    assert len(result.contacts) == 1
    assert result.states[(0, "a")].nu_body[0].item() == pytest.approx(-1)
    assert result.states[(0, "b")].nu_body[0].item() == pytest.approx(1)
    assert kinetic(a, result.states[(0, "a")]) + kinetic(b, result.states[(0, "b")]) == pytest.approx(before)
    assert result.states[(0, "a")].nu_body[0].item() + result.states[(0, "b")].nu_body[0].item() == 0
    assert result.events[0].normal_impulse_ns == pytest.approx(20)


def test_inelastic_wall_and_friction_dissipate_energy():
    a = vessel("a", 0.8, 1, v=1)
    wall = static("wall", 2, Box((1, 5, 5)))
    before = kinetic(a, a.state)
    result = resolve_contacts((wall, a), material=ContactMaterial(restitution=0, friction=0.5), dt_s=0.1)
    assert result.states[(0, "a")].nu_body[0].item() == pytest.approx(0)
    assert kinetic(a, result.states[(0, "a")]) <= before + 1e-10
    assert result.events[0].friction_impulse_ns > 0
    assert result.states[(0, "a")].position_ned[0].item() == pytest.approx(0)
    assert result.events[0].body_a_equivalent_wrench_frd.shape == (6,)
    assert abs(result.states[(0, "a")].nu_body[5].item()) > 0


def test_scripted_world_entity_velocity_enters_contact():
    spec = World.model_validate({"source": {"kind": "parametric"},
        "environment": {"current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                        "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                        "waves": {"kind": "calm"}, "visibility_m": 1000},
        "scripted_entities": [{"id": "traffic", "position_ned_m": [2, 0, 0],
                               "shape": {"kind": "sphere", "radius_m": 1},
                               "velocity_ned_mps": [-1, 0, 0]}]})
    world = ParametricWorld(spec, 0)
    bodies = world_collision_bodies(world, sim_time_s=0.2)
    assert bodies[0].kinematic_velocity_ned_mps == (-1, 0, 0)
    a = vessel("a", 0, 0)
    result = resolve_contacts((a, *bodies), material=ContactMaterial(), dt_s=0.1)
    assert result.states[(0, "a")].nu_body[0].item() < 0


def test_planar_contact_projects_out_of_plane_response():
    a = vessel("a", 0.8, 1, mode="planar3")
    wall = static("wall", 2, Box((1, 5, 5)))
    result = resolve_contacts((a, wall), material=ContactMaterial(), dt_s=0.1)
    assert result.states[(0, "a")].nu_body.tolist() == pytest.approx([0, 0, 0, 0, 0, 0])
    assert result.states[(0, "a")].position_ned[2].item() == 0
    below = static("below", 0, Sphere(1), z=1.8)
    vertical = resolve_contacts((a, below), material=ContactMaterial(), dt_s=0.1)
    assert vertical.states[(0, "a")].nu_body[2].item() == 0
    assert vertical.states[(0, "a")].position_ned[2].item() == 0


def test_contact_ordering_and_environment_isolation():
    a, b, c = vessel("a", 0, 1), vessel("b", 1.8, -1), vessel("c", 0, 1, env=1)
    one = resolve_contacts((a, b, c), material=ContactMaterial(restitution=0.2), dt_s=0.01)
    two = resolve_contacts((c, b, a), material=ContactMaterial(restitution=0.2), dt_s=0.01)
    assert [contact.id for contact in one.contacts] == [contact.id for contact in two.contacts]
    for name in ("a", "b", "c"):
        env_id = 1 if name == "c" else 0
        assert torch.equal(one.states[(env_id, name)].nu_body, two.states[(env_id, name)].nu_body)
    assert torch.equal(one.states[(1, "c")].nu_body, c.state.nu_body)
    with pytest.raises(DuplicateIdentityError):
        broadphase_pairs((a, a))
    repeated_name = vessel("a", 0, 3, env=1)
    isolated = resolve_contacts((a, b, repeated_name), material=ContactMaterial(), dt_s=0.01)
    assert (0, "a") in isolated.states and (1, "a") in isolated.states
    assert isolated.states[(1, "a")].nu_body[0].item() == 3


def test_simultaneous_contacts_are_order_independent_and_dissipative():
    a, b, c = vessel("a", 0, 1), vessel("b", 1.8, 0), vessel("c", 3.6, -1)
    before = sum(kinetic(body, body.state) for body in (a, b, c))
    first = resolve_contacts((a, b, c), material=ContactMaterial(restitution=0.25), dt_s=0.01)
    second = resolve_contacts((c, a, b), material=ContactMaterial(restitution=0.25), dt_s=0.01)
    assert len(first.contacts) == 2
    assert [contact.id for contact in first.contacts] == [contact.id for contact in second.contacts]
    for name in ("a", "b", "c"):
        assert torch.equal(first.states[(0, name)].nu_body, second.states[(0, name)].nu_body)
    after = sum(kinetic(body, first.states[(0, body.id)]) for body in (a, b, c))
    assert after <= before + 1e-9


def test_rotated_vessel_impact_uses_body_frame_inertia():
    a = vessel("a", 0.8, 0)
    yaw90 = math.sqrt(0.5)
    rotated = VesselState(a.state.position_ned, vector((yaw90, 0, 0, yaw90)),
                          vector((0, -1, 0, 0, 0, 0)))
    a = CollisionBody(0, "a", Sphere(1), rotated, a.plant)
    wall = static("wall", 2, Box((1, 5, 5)))
    result = resolve_contacts((a, wall), material=ContactMaterial(restitution=0), dt_s=0.1)
    assert result.states[(0, "a")].nu_body[1].item() == pytest.approx(0, abs=1e-12)


def test_supported_shape_pairs_and_explicit_unsupported_pair():
    sphere = vessel("a", 0, 0, shape=Sphere(1))
    capsule = static("capsule", 1.3, Capsule(0.7, 0.5))
    assert len(detect_contacts((sphere, capsule))) == 1
    moving_capsule = vessel("moving_capsule", 0, 0, shape=Capsule(0.7, 0.5))
    assert len(detect_contacts((moving_capsule, capsule))) == 1
    box = vessel("box", 0, 0, shape=Box((1, 1, 1)))
    other_box = static("other", 1.5, Box((1, 1, 1)))
    assert len(detect_contacts((box, other_box))) == 1
    mesh = TriangleMesh(((0.8, -2, -2), (0.8, 2, -2), (0.8, 0, 2)), ((0, 1, 2),))
    assert len(detect_contacts((sphere, static("mesh", 0, mesh)))) == 1
    tetra = ConvexHull(((0.8, -1, -1), (0.8, 1, -1), (0.8, 0, 1), (1.8, 0, 0)),
                       ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)))
    assert len(detect_contacts((sphere, static("hull", 0, tetra)))) == 1
    compound = Compound((CompoundChild(Sphere(0.6), (1.5, 0, 0)),))
    assert len(detect_contacts((sphere, static("compound", 0, compound)))) == 1
    with pytest.raises(CollisionSolverError):
        detect_contacts((box, capsule))


def test_bad_contact_inputs_fail_closed():
    with pytest.raises(CollisionSolverError):
        ContactMaterial(restitution=1.1)
    with pytest.raises(CollisionSolverError):
        resolve_contacts((vessel("a", 0, 0),), material=ContactMaterial(), dt_s=0)
    with pytest.raises(PhysicalValidationError):
        Sphere(-1)


def wake_state(x, surge):
    return VesselState(vector((x, 0, 0)), vector((1, 0, 0, 0)), vector((surge, 0, 0, 0, 0, 0)))


def test_wake_is_double_buffered_and_order_independent():
    params = WakeParameters(0.2, 2, 10, "analytic-baseline", "1")
    a = GaussianWakeEmitter(0, 1, params).emit(wake_state(0, 2))
    b = GaussianWakeEmitter(0, 2, params).emit(wake_state(1, 1))
    query = vector(((-2, 0, 0),)).reshape((1, 3))
    first, reverse = WakeField(), WakeField()
    first.begin_step()
    reverse.begin_step()
    first.emit(a)
    first.emit(b)
    reverse.emit(b)
    reverse.emit(a)
    assert first.sample(query, env_id=0)[0, 0].item() == 0
    assert reverse.sample(query, env_id=0)[0, 0].item() == 0
    first.swap()
    reverse.swap()
    assert torch.equal(first.sample(query, env_id=0), reverse.sample(query, env_id=0))
    assert first.sample(query, env_id=0)[0, 0].item() < 0
    assert first.sample(query, env_id=0, receiver_vessel_id=1)[0, 0].item() > first.sample(query, env_id=0)[0, 0].item()
    assert first.sample(query, env_id=1)[0, 0].item() == 0
    previous = first.sample(query, env_id=0).clone()
    first.begin_step()
    first.emit(a)
    assert torch.equal(first.sample(query, env_id=0), previous)
    first.swap()
    assert first.generation == 2


def test_wake_duplicate_emitter_rejected():
    params = WakeParameters(0.2, 2, 10, "analytic-baseline", "1")
    emission = GaussianWakeEmitter(0, 1, params).emit(wake_state(0, 2))
    field = WakeField()
    field.begin_step()
    field.emit(emission)
    with pytest.raises(DuplicateIdentityError):
        field.emit(emission)


def test_world_samples_only_swapped_wake_buffer():
    spec = World.model_validate({"source": {"kind": "parametric"},
        "environment": {"current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                        "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]},
                        "waves": {"kind": "calm"}, "visibility_m": 1000}})
    world = ParametricWorld(spec, 0)
    emission = GaussianWakeEmitter(0, 1, WakeParameters(0.2, 2, 10, "baseline", "1")).emit(wake_state(0, 2))
    query = vector(((-2, 0, 0),)).reshape((1, 3))
    world.wake.begin_step()
    world.wake.emit(emission)
    assert world.sample(query, sim_time_s=0, env_id=0, receiver_vessel_id=2).wake_ned_mps[0, 0].item() == 0
    world.wake.swap()
    assert world.sample(query, sim_time_s=0.1, env_id=0, receiver_vessel_id=2).wake_ned_mps[0, 0].item() < 0
