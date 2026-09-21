"""Sequential deterministic impulse solver at the post-integration contact stage."""

from dataclasses import dataclass
import math

import torch

from bcod_sim.collision.broadphase import CollisionBody
from bcod_sim.collision.narrowphase import Contact, detect_contacts
from bcod_sim.core.errors import CollisionSolverError
from bcod_sim.frames.tensor import rotate_body_to_world, rotate_world_to_body
from bcod_sim.state.vessel_state import VesselState


@dataclass(frozen=True)
class ContactMaterial:
    restitution: float = 0.0
    friction: float = 0.0
    position_correction_fraction: float = 1.0

    def __post_init__(self) -> None:
        if (not all(math.isfinite(x) for x in (self.restitution, self.friction, self.position_correction_fraction)) or
            not (0 <= self.restitution <= 1) or self.friction < 0 or
            not (0 <= self.position_correction_fraction <= 1)):
            raise CollisionSolverError("Invalid contact material")


@dataclass(frozen=True)
class ContactEvent:
    contact_id: str
    point_ned_m: torch.Tensor
    normal_a_to_b_ned: torch.Tensor
    normal_impulse_ns: float
    friction_impulse_ns: float
    body_a_impulse_frd: torch.Tensor
    body_b_impulse_frd: torch.Tensor
    body_a_equivalent_wrench_frd: torch.Tensor
    body_b_equivalent_wrench_frd: torch.Tensor
    penetration_m: float
    normal_relative_velocity_mps: float
    tangential_relative_velocity_mps: float
    contact_force_on_a_ned_n: torch.Tensor


@dataclass(frozen=True)
class ContactResult:
    states: dict[tuple[int, str], VesselState]
    contacts: tuple[Contact, ...]
    events: tuple[ContactEvent, ...]


def _contact_jacobian(body: CollisionBody, state: VesselState, point: torch.Tensor,
                      direction_world: torch.Tensor) -> torch.Tensor:
    direction_body = rotate_world_to_body(direction_world, state.q_body_to_ned)
    r_body = rotate_world_to_body(point - state.position_ned, state.q_body_to_ned)
    return torch.cat((direction_body, torch.linalg.cross(r_body, direction_body)))


def _inverse_mass_action(body: CollisionBody, jacobian: torch.Tensor) -> torch.Tensor:
    plant = body.plant
    assert plant is not None
    if plant.mode == "full6":
        return torch.linalg.solve(plant.total_mass, jacobian)
    active = plant.active
    projected = torch.linalg.solve(plant.active_mass, jacobian.index_select(0, active))
    result = torch.zeros_like(jacobian)
    return result.index_copy(0, active, projected)


def _point_velocity(body: CollisionBody, state: VesselState | None, point: torch.Tensor) -> torch.Tensor:
    if state is None:
        return point.new_tensor(body.kinematic_velocity_ned_mps)
    r_body = rotate_world_to_body(point - state.position_ned, state.q_body_to_ned)
    velocity_body = state.nu_body[:3] + torch.linalg.cross(state.nu_body[3:], r_body)
    return rotate_body_to_world(velocity_body, state.q_body_to_ned)


def _effective_mass(body: CollisionBody, state: VesselState | None, point: torch.Tensor,
                    direction: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
    if state is None:
        return None, None, 0.0
    jacobian = _contact_jacobian(body, state, point, direction)
    inverse_action = _inverse_mass_action(body, jacobian)
    return jacobian, inverse_action, torch.dot(jacobian, inverse_action).item()


def _apply_impulse(body: CollisionBody, state: VesselState | None, inverse_action: torch.Tensor | None,
                   scalar_impulse: float) -> VesselState | None:
    if state is None:
        return None
    assert inverse_action is not None
    nu = state.nu_body + scalar_impulse * inverse_action
    if body.plant.mode == "planar3":
        nu = nu.index_fill(0, torch.tensor((2, 3, 4), device=nu.device), 0)
    if not torch.isfinite(nu).all().item():
        raise CollisionSolverError("Contact impulse produced nonfinite velocity")
    return VesselState(state.position_ned, state.q_body_to_ned, nu)


def _wrench_impulse(body: CollisionBody, state: VesselState | None, point: torch.Tensor,
                    impulse_world: torch.Tensor) -> torch.Tensor:
    if state is None:
        return torch.zeros(6, dtype=point.dtype, device=point.device)
    force = rotate_world_to_body(impulse_world, state.q_body_to_ned)
    r = rotate_world_to_body(point - state.position_ned, state.q_body_to_ned)
    wrench = torch.cat((force, torch.linalg.cross(r, force)))
    if body.plant.mode == "planar3":
        wrench = wrench.index_fill(0, torch.tensor((2, 3, 4), device=wrench.device), 0)
    return wrench


def resolve_contacts(bodies: tuple[CollisionBody, ...], *, material: ContactMaterial,
                     dt_s: float) -> ContactResult:
    """Call once after a continuous substep; impulses precede planar state enforcement."""
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise CollisionSolverError("Contact stage requires positive finite dt")
    modes: dict[int, str] = {}
    tensor_schemas: dict[int, tuple[torch.dtype, torch.device]] = {}
    for body in bodies:
        if body.dynamic:
            prior = modes.setdefault(body.env_id, body.plant.mode)
            if prior != body.plant.mode:
                raise CollisionSolverError("One environment cannot mix dynamics modes")
            schema = (body.state.nu_body.dtype, body.state.nu_body.device)
            if tensor_schemas.setdefault(body.env_id, schema) != schema:
                raise CollisionSolverError("One environment cannot mix collision tensor schemas")
    contacts = detect_contacts(bodies)
    by_id = {(body.env_id, body.id): body for body in bodies}
    states = {(body.env_id, body.id): body.state.clone() for body in bodies if body.state is not None}
    events = []
    for contact in contacts:
        a, b = by_id[(contact.env_id, contact.body_a)], by_id[(contact.env_id, contact.body_b)]
        pair_material = b.contact_material or a.contact_material or material
        state_a, state_b = states.get((a.env_id, a.id)), states.get((b.env_id, b.id))
        point = contact.point_ned_m
        normal = contact.normal_a_to_b_ned
        if point.dtype != (state_a or state_b).position_ned.dtype:
            raise CollisionSolverError("Contact geometry and state dtype mismatch")
        relative = _point_velocity(b, state_b, point) - _point_velocity(a, state_a, point)
        normal_speed = torch.dot(relative, normal).item()
        ja, ia, ka = _effective_mass(a, state_a, point, normal)
        jb, ib, kb = _effective_mass(b, state_b, point, normal)
        k = ka + kb
        normal_impulse = 0.0
        if normal_speed < 0 and k > 1e-14:
            normal_impulse = -(1 + pair_material.restitution) * normal_speed / k
            state_a = _apply_impulse(a, state_a, ia, -normal_impulse)
            state_b = _apply_impulse(b, state_b, ib, normal_impulse)
        tangent_impulse = 0.0
        tangent = torch.zeros_like(normal)
        if pair_material.friction > 0 and normal_impulse > 0:
            relative_after = _point_velocity(b, state_b, point) - _point_velocity(a, state_a, point)
            tangential = relative_after - torch.dot(relative_after, normal) * normal
            speed = torch.linalg.vector_norm(tangential).item()
            if speed > 1e-12:
                tangent = tangential / speed
                _, ita, kta = _effective_mass(a, state_a, point, tangent)
                _, itb, ktb = _effective_mass(b, state_b, point, tangent)
                if kta + ktb > 1e-14:
                    tangent_impulse = min(speed / (kta + ktb), pair_material.friction * normal_impulse)
                    state_a = _apply_impulse(a, state_a, ita, tangent_impulse)
                    state_b = _apply_impulse(b, state_b, itb, -tangent_impulse)
        impulse_state_a, impulse_state_b = state_a, state_b
        # Positional correction changes no kinetic energy and obeys planar projection.
        correction_normal = normal
        if (a.dynamic and a.plant.mode == "planar3") or (b.dynamic and b.plant.mode == "planar3"):
            correction_normal = torch.stack((normal[0], normal[1], normal.new_zeros(())))
            length = torch.linalg.vector_norm(correction_normal).item()
            correction_normal = correction_normal / length if length > 1e-12 else torch.zeros_like(normal)
        inverse_a = 1 / a.plant.mass.mass_kg if a.dynamic else 0.0
        inverse_b = 1 / b.plant.mass.mass_kg if b.dynamic else 0.0
        total_inverse = inverse_a + inverse_b
        correction = pair_material.position_correction_fraction * contact.penetration_m
        if total_inverse > 0 and correction > 0:
            if state_a is not None:
                position = state_a.position_ned - correction_normal * correction * inverse_a / total_inverse
                state_a = VesselState(position, state_a.q_body_to_ned, state_a.nu_body)
            if state_b is not None:
                position = state_b.position_ned + correction_normal * correction * inverse_b / total_inverse
                state_b = VesselState(position, state_b.q_body_to_ned, state_b.nu_body)
        if state_a is not None:
            states[(a.env_id, a.id)] = state_a
        if state_b is not None:
            states[(b.env_id, b.id)] = state_b
        impulse_on_b = normal_impulse * normal - tangent_impulse * tangent
        impulse_a = _wrench_impulse(a, impulse_state_a, point, -impulse_on_b)
        impulse_b = _wrench_impulse(b, impulse_state_b, point, impulse_on_b)
        tangential_speed=torch.linalg.vector_norm(relative-torch.dot(relative,normal)*normal).item()
        events.append(ContactEvent(contact.id, point, normal, normal_impulse, tangent_impulse,
                                   impulse_a, impulse_b, impulse_a / dt_s, impulse_b / dt_s,
                                   contact.penetration_m,normal_speed,tangential_speed,-impulse_on_b/dt_s))
    return ContactResult(states, contacts, tuple(events))
