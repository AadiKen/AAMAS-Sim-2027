"""Single 6-DOF plant; planar dynamics are a projection of this model."""

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import torch

from bcod_sim.core.errors import NonFiniteStateError, PhysicalValidationError
from bcod_sim.dynamics.coriolis import coriolis_wrench
from bcod_sim.dynamics.crossflow import CrossflowModel, NoCrossflow
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.diagnostics import EXTERNAL_TERMS, WrenchLedger, QuaternionNormalizationDiagnostic, make_ledger
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.restoring import HydrostaticsModel
from bcod_sim.frames.tensor import rotate_world_to_body
from bcod_sim.state.vessel_state import VesselState


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind()
    bw, bx, by, bz = b.unbind()
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw))


def _body_to_world(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    t = 2 * torch.linalg.cross(q[1:], v)
    return v + q[0] * t + torch.linalg.cross(q[1:], t)


def _rpy_to_q(roll: float, pitch: float, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind()
    return torch.atan2(2 * (w*z+x*y), 1 - 2 * (y*y+z*z))


@dataclass(frozen=True)
class PlanarEquilibrium:
    heave_ned_m: float
    roll_rad: float
    pitch_rad: float

    def validate(self) -> None:
        if not all(math.isfinite(x) for x in (self.heave_ned_m, self.roll_rad, self.pitch_rad)):
            raise PhysicalValidationError("Planar equilibrium must be finite")
        if abs(math.cos(self.pitch_rad)) < 1e-6:
            raise PhysicalValidationError("Planar equilibrium pitch is kinematically singular")


@dataclass(frozen=True)
class StepResult:
    state: VesselState
    diagnostics: WrenchLedger
    quaternion_norm_error: float
    normalization_diagnostic: QuaternionNormalizationDiagnostic | None


class Plant6:
    """Immutable mode and model parameters for one vessel/episode."""

    def __init__(self, mass: MassProperties, damping: Damping, hydrostatics: HydrostaticsModel,
                 envelope: OperatingEnvelope, *, mode: Literal["full6", "planar3"] = "full6",
                 planar_equilibrium: PlanarEquilibrium | None = None,
                 crossflow: CrossflowModel | None = None) -> None:
        if mode not in ("full6", "planar3"):
            raise PhysicalValidationError("Unknown dynamics mode")
        if mode == "planar3" and planar_equilibrium is None:
            raise PhysicalValidationError("Planar mode requires vessel-specific equilibrium")
        if planar_equilibrium:
            planar_equilibrium.validate()
        rigid, total = mass.matrices()
        damping.validate(dtype=total.dtype, device=total.device)
        hydrostatics.validate(dtype=total.dtype, device=total.device)
        crossflow = crossflow or NoCrossflow()
        crossflow.validate(dtype=total.dtype, device=total.device)
        envelope.validate(dtype=total.dtype, device=total.device)
        self.mass = mass
        self.damping = damping
        self.hydrostatics = hydrostatics
        self.crossflow = crossflow
        self.envelope = envelope
        self._mode = mode
        self.planar_equilibrium = planar_equilibrium
        self.rigid_mass = rigid
        self.added_mass = mass.added_mass_kg
        self.total_mass = total
        self.active = torch.tensor((0, 1, 5), dtype=torch.long, device=total.device)
        self.active_mass = total.index_select(0, self.active).index_select(1, self.active)

    @property
    def mode(self) -> Literal["full6", "planar3"]:
        return self._mode

    def _validate_external(self, external: Mapping[str, torch.Tensor]) -> None:
        if set(external) != set(EXTERNAL_TERMS):
            raise PhysicalValidationError("All external wrench terms must be explicit")
        for name, value in external.items():
            if (value.shape != (6,) or value.dtype != self.total_mass.dtype or
                value.device != self.total_mass.device or not torch.isfinite(value).all().item()):
                raise PhysicalValidationError(f"Invalid {name} wrench")

    def diagnostics(self, state: VesselState, external: Mapping[str, torch.Tensor],
                    water_velocity_ned: torch.Tensor | None = None) -> WrenchLedger:
        self._validate_external(external)
        nu = state.nu_body
        terms = {name: external[name] for name in EXTERNAL_TERMS}
        terms["rigid_coriolis"] = -coriolis_wrench(self.rigid_mass, nu)
        terms["added_mass_coriolis"] = -coriolis_wrench(self.added_mass, nu)
        water_body = None if water_velocity_ned is None else rotate_world_to_body(water_velocity_ned, state.q_body_to_ned)
        nu_relative = nu.clone()
        if water_body is not None: nu_relative[:3] = nu_relative[:3] - water_body
        damping = self.damping.evaluate(nu_relative)
        terms["linear_damping"], terms["nonlinear_damping"] = self.damping.components(nu_relative)
        hydrostatic = self.hydrostatics.evaluate(state, self.mass.mass_kg, self.mass.cg_frd_m)
        crossflow = self.crossflow.evaluate(state, water_body)
        terms["restoring"] = hydrostatic.tau_body
        terms["crossflow"] = crossflow.tau_body
        return make_ledger(terms, {"tau_hydrostatic": hydrostatic.tau_body,
                                   "tau_crossflow": crossflow.tau_body,
                                   "damping": damping.diagnostics,
                                   "hydrostatics": hydrostatic.diagnostics,
                                   "crossflow": crossflow.diagnostics})

    def acceleration(self, state: VesselState, external: Mapping[str, torch.Tensor],
                     water_velocity_ned: torch.Tensor | None = None) -> torch.Tensor:
        rhs = self.diagnostics(state, external, water_velocity_ned).total
        if self.mode == "full6":
            return torch.linalg.solve(self.total_mass, rhs)
        active_acc = torch.linalg.solve(self.active_mass, rhs.index_select(0, self.active))
        acceleration = torch.zeros_like(state.nu_body)
        return acceleration.index_copy(0, self.active, active_acc)

    def _constrain(self, position: torch.Tensor, q: torch.Tensor, nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "full6":
            return position, q, nu
        equilibrium = self.planar_equilibrium
        assert equilibrium is not None
        position = torch.stack((position[0], position[1], position.new_tensor(equilibrium.heave_ned_m)))
        q = _rpy_to_q(equilibrium.roll_rad, equilibrium.pitch_rad, _yaw(q))
        nu = torch.stack((nu[0], nu[1], nu.new_zeros(()), nu.new_zeros(()), nu.new_zeros(()), nu[5]))
        return position, q, nu

    def step(self, state: VesselState, external: Mapping[str, torch.Tensor], dt_s: float,
             water_velocity_ned: torch.Tensor | None = None) -> StepResult:
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise PhysicalValidationError("Substep must be positive and finite")
        self._validate_external(external)
        self.envelope.check(state, dt_s)
        p0, q0, nu0 = self._constrain(state.position_ned, state.q_body_to_ned, state.nu_body)

        def derivative(p: torch.Tensor, q: torch.Tensor, nu: torch.Tensor):
            q = q / torch.linalg.vector_norm(q)
            p, q, nu = self._constrain(p, q, nu)
            local = VesselState(p, q, nu)
            acc = self.acceleration(local, external, water_velocity_ned)
            if self.mode == "planar3":
                equilibrium = self.planar_equilibrium
                assert equilibrium is not None
                yaw = _yaw(q)
                yaw_dot = nu[5] * math.cos(equilibrium.roll_rad) / math.cos(equilibrium.pitch_rad)
                world_velocity = _body_to_world(nu[:3], q)
                p_dot = torch.stack((world_velocity[0], world_velocity[1], nu.new_zeros(())))
                world_yaw_rate = torch.stack((nu.new_zeros(()), nu.new_zeros(()), nu.new_zeros(()), yaw_dot))
                q_dot = 0.5 * _qmul(world_yaw_rate, q)
                return p_dot, q_dot, acc
            p_dot = _body_to_world(nu[:3], q)
            omega = torch.cat((nu.new_zeros((1,)), nu[3:]))
            q_dot = 0.5 * _qmul(q, omega)
            return p_dot, q_dot, acc

        k1 = derivative(p0, q0, nu0)
        k2 = derivative(p0 + dt_s/2*k1[0], q0 + dt_s/2*k1[1], nu0 + dt_s/2*k1[2])
        k3 = derivative(p0 + dt_s/2*k2[0], q0 + dt_s/2*k2[1], nu0 + dt_s/2*k2[2])
        k4 = derivative(p0 + dt_s*k3[0], q0 + dt_s*k3[1], nu0 + dt_s*k3[2])
        position = p0 + dt_s/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        raw_q = q0 + dt_s/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        nu = nu0 + dt_s/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        if not all(torch.isfinite(x).all().item() for x in (position, raw_q, nu)):
            raise NonFiniteStateError("Integration produced nonfinite state")
        norm_error = abs(torch.linalg.vector_norm(raw_q).item() - 1.0)
        if torch.linalg.vector_norm(raw_q).item() < 1e-12:
            raise NonFiniteStateError("Integrated quaternion collapsed")
        q = raw_q / torch.linalg.vector_norm(raw_q)
        position, q, nu = self._constrain(position, q, nu)
        result = VesselState(position, q, nu)
        self.envelope.check(result, dt_s)
        threshold = 1e-8
        diagnostic = (QuaternionNormalizationDiagnostic("quaternion_normalized", norm_error, threshold)
                      if norm_error > threshold else None)
        return StepResult(result, self.diagnostics(result, external, water_velocity_ned), norm_error, diagnostic)
