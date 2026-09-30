"""Single 6-DOF plant; planar dynamics are a projection of this model."""

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import torch

from bcod_sim.core.errors import NonFiniteStateError, PhysicalValidationError
from bcod_sim.dynamics.coriolis import coriolis_wrench
from bcod_sim.vessel_generation.spec_a.fit import CoefficientSurface
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
                 crossflow: CrossflowModel | None = None,
                 maneuvering_surface: CoefficientSurface | None = None,
                 added_mass_coriolis_enabled: bool = True,
                 steady_coriolis_owner: Literal["bem", "maneuvering_model"] = "bem",
                 surface_min_forward_speed_mps: float = 0.) -> None:
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
        if maneuvering_surface is not None and added_mass_coriolis_enabled:
            raise PhysicalValidationError("CFD/system-ID surface requires added-mass Coriolis disabled")
        if steady_coriolis_owner not in ("bem", "maneuvering_model"):
            raise PhysicalValidationError("Unknown steady added-mass Coriolis owner")
        if steady_coriolis_owner == "maneuvering_model" and added_mass_coriolis_enabled:
            raise PhysicalValidationError("Maneuvering model and BEM cannot both own steady added-mass Coriolis")
        if steady_coriolis_owner == "bem" and not added_mass_coriolis_enabled and maneuvering_surface is None:
            raise PhysicalValidationError("BEM steady Coriolis owner requires the term enabled outside captive surfaces")
        if not math.isfinite(surface_min_forward_speed_mps) or surface_min_forward_speed_mps < 0:
            raise PhysicalValidationError("Invalid surface forward-speed threshold")
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
        self.maneuvering_surface = maneuvering_surface
        self.added_mass_coriolis_enabled = added_mass_coriolis_enabled
        self.steady_coriolis_owner = steady_coriolis_owner
        self.surface_min_forward_speed_mps = surface_min_forward_speed_mps

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
        water_body = None if water_velocity_ned is None else rotate_world_to_body(water_velocity_ned, state.q_body_to_ned)
        nu_relative = nu.clone()
        if water_body is not None: nu_relative[:3] = nu_relative[:3] - water_body
        surface_nu = nu_relative.clone()
        if self.maneuvering_surface is not None:
            reference = nu_relative.new_tensor(self.maneuvering_surface.moment_reference_frd_m)
            surface_nu[:3] += torch.linalg.cross(nu_relative[3:], reference)
        surface_active = (self.maneuvering_surface is not None and
                          surface_nu[0].item() >= self.surface_min_forward_speed_mps and
                          surface_nu[0].item() > 0 and
                          torch.linalg.vector_norm(surface_nu[:2]).item() <=
                          .45*(9.80665*self.maneuvering_surface.length_m)**.5)
        surface_route = "not_configured"
        if self.maneuvering_surface is not None:
            fr = torch.linalg.vector_norm(surface_nu[:2]).item()/(9.80665*self.maneuvering_surface.length_m)**.5
            surface_route = ("out_of_envelope" if fr > .45 else
                             "low_speed_or_reverse_v5" if not surface_active else
                             "extrapolated_fr" if fr >= .30 else "surface")
        # The captive physical Y/N surface includes C_A(nu)nu. Retain the
        # original V5 + C_A path only when the surface is outside its domain.
        # A captive surface owns steady Y/N loads only within its envelope.
        # Outside that envelope, BEM ownership restores C_A(nu)nu unless the
        # package explicitly assigns the total steady hull load to the
        # maneuvering model.
        use_added_coriolis = (not surface_active and self.steady_coriolis_owner == "bem" and
                             (self.added_mass_coriolis_enabled or self.maneuvering_surface is not None))
        terms["added_mass_coriolis"] = (-coriolis_wrench(self.added_mass, nu) if use_added_coriolis
                                         else torch.zeros_like(nu))
        damping = self.damping.evaluate(nu_relative)
        terms["linear_damping"], terms["nonlinear_damping"] = self.damping.components(nu_relative)
        hydrostatic = self.hydrostatics.evaluate(state, self.mass.mass_kg, self.mass.cg_frd_m)
        crossflow = self.crossflow.evaluate(state, water_body)
        if surface_active:
            # Keep the established surge resistance, remove V5 lateral/yaw
            # loads, then add the physical captive Y/N and ΔX exactly once.
            terms["linear_damping"] = terms["linear_damping"].clone()
            terms["nonlinear_damping"] = terms["nonlinear_damping"].clone()
            terms["linear_damping"][[1, 5]] = 0
            terms["nonlinear_damping"][[1, 5]] = 0
            straight_nu = torch.zeros_like(nu_relative)
            straight_nu[0] = surface_nu[0]
            straight_linear, straight_nonlinear = self.damping.components(straight_nu)
            terms["linear_damping"][0] = straight_linear[0]
            terms["nonlinear_damping"][0] = straight_nonlinear[0]
            dx, y, n = self.maneuvering_surface.evaluate(
                *[float(surface_nu[i]) for i in (0, 1, 5)])
            terms["nonlinear_damping"][0] += dx
            # Preserve vertical/roll/pitch crossflow outside the horizontal fit.
            surface_tau = crossflow.tau_body.clone()
            surface_tau[0] = 0
            surface_tau[1], surface_tau[5] = y, n
            # Captive moments and velocity coordinates share the declared CG
            # reference. Plant6's spatial matrices use the body-frame origin.
            physical_horizontal = nu.new_tensor([float(straight_linear[0]+straight_nonlinear[0])+dx, y, 0.])
            surface_tau[3:] += torch.linalg.cross(reference, physical_horizontal)
        else:
            surface_tau = crossflow.tau_body
        terms["restoring"] = hydrostatic.tau_body
        terms["crossflow"] = surface_tau
        return make_ledger(terms, {"tau_hydrostatic": hydrostatic.tau_body,
                                   "tau_crossflow": surface_tau,
                                   "maneuvering_surface_active": surface_active,
                                   "maneuvering_surface_route": surface_route,
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
        projected_q = _rpy_to_q(equilibrium.roll_rad, equilibrium.pitch_rad, _yaw(q))
        # q and -q represent the same rotation, but RK4 derivatives must stay
        # on the input hemisphere when yaw wraps at +/-pi.
        q = torch.where(torch.dot(projected_q, q) < 0, -projected_q, projected_q)
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
