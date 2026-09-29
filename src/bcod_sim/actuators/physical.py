"""Physical actuator set and bounded control allocation for Plant6 FRD loads.

This module does not alter the existing hull model or legacy actuator API.
All predicted moments are the cross product of a device position and force.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares, lsq_linear

from bcod_sim.core.errors import DuplicateIdentityError, PhysicalValidationError


def _vec(value: Any, n: int = 3) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise PhysicalValidationError(f"Expected finite {n}-vector")
    return a


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class VesselMotion:
    velocity_frd: tuple[float, float, float] = (0., 0., 0.)
    angular_velocity_frd: tuple[float, float, float] = (0., 0., 0.)
    current_frd: tuple[float, float, float] = (0., 0., 0.)
    heading_rad: float = 0.
    water_density: float = 1025.

    def local_flow(self, position: np.ndarray) -> np.ndarray:
        if self.water_density <= 0 or not math.isfinite(self.water_density):
            raise PhysicalValidationError("Invalid water density")
        return _vec(self.velocity_frd) + np.cross(_vec(self.angular_velocity_frd), position) - _vec(self.current_frd)


@dataclass
class DeviceState:
    thrust_n: float = 0.
    rpm: float = 0.
    angle_rad: float = 0.
    raw_command: tuple[float, ...] = ()
    target_command: tuple[float, ...] = ()
    saturated: bool = False
    rate_limited: bool = False


@dataclass(frozen=True)
class DeviceLoad:
    id: str
    force_frd: tuple[float, float, float]
    moment_frd: tuple[float, float, float]
    local_inflow_frd: tuple[float, float, float]
    thrust_n: float
    angle_rad: float
    rpm: float
    raw_command: tuple[float, ...]
    target_command: tuple[float, ...]
    saturated: bool
    rate_limited: bool
    status: str


@dataclass(frozen=True)
class WrenchPrediction:
    devices: tuple[DeviceLoad, ...]
    wrench_frd: tuple[float, float, float, float, float, float]


class PhysicalDevice:
    """One force applied at one physical center of pressure."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.id = str(spec["id"])
        self.kind = spec["type"]
        if self.kind not in {"fixed_thruster", "tunnel_thruster", "azimuth_thruster", "rudder"}:
            raise PhysicalValidationError(f"Unsupported actuator type: {self.kind}")
        pose = spec["pose"]
        self.position = _vec(pose["position_frd"])
        self.direction = _vec(pose.get("direction_frd", (1., 0., 0.)))
        length = np.linalg.norm(self.direction)
        if length <= 0:
            raise PhysicalValidationError("Zero actuator direction")
        self.direction /= length
        self.state = DeviceState()
        self.status = spec.get("status", "estimated")
        if self.status not in {"validated_internal", "manufacturer_model", "measured", "calibrated", "estimated", "low_confidence", "unsupported"}:
            raise PhysicalValidationError("Invalid actuator status")
        self.dynamics = spec.get("dynamics", {})
        self.propulsion = spec.get("propulsion", {})
        self.steering = spec.get("steering", {})
        self.geometry = spec.get("geometry", {})
        self.sources = tuple(spec.get("inflow_sources", ()))
        self._validate()

    def _validate(self) -> None:
        if self.kind == "rudder":
            if float(self.geometry.get("area_m2", 0)) <= 0:
                raise PhysicalValidationError("Rudder requires positive area")
            if float(self.geometry.get("aspect_ratio", 0)) <= 0:
                raise PhysicalValidationError("Rudder requires positive aspect ratio")
        else:
            model = self.propulsion.get("model", "power_law")
            if model not in {"power_law", "static_map", "open_water", "power_estimate"}:
                raise PhysicalValidationError(f"Unsupported propulsion model: {model}")
            if model == "static_map":
                pairs = self.propulsion.get("points", ())
                if len(pairs) < 2 or any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(pairs, pairs[1:])):
                    raise PhysicalValidationError("Static map must be monotonically increasing")
            if model == "open_water" and (float(self.propulsion.get("diameter_m", 0)) <= 0 or float(self.propulsion.get("max_rpm", 0)) <= 0):
                raise PhysicalValidationError("Open-water model requires diameter and max RPM")
        for key in ("thrust_tau_s", "rpm_tau_s", "steering_tau_s", "steering_rate_radps", "thrust_rate_nps"):
            if key in self.dynamics and float(self.dynamics[key]) <= 0:
                raise PhysicalValidationError(f"{key} must be positive")

    @property
    def command_bounds(self) -> tuple[tuple[float, float], ...]:
        if self.kind == "rudder":
            return (tuple(self.steering.get("angle_bounds_rad", (-0.61, 0.61))),)
        thrust = tuple(self.propulsion.get("command_bounds", (-1., 1.)))
        if self.kind == "azimuth_thruster":
            return (thrust, tuple(self.steering.get("angle_bounds_rad", (-math.pi, math.pi))))
        return (thrust,)

    def current_command(self) -> tuple[float, ...]:
        if self.kind == "rudder":
            return (self.state.angle_rad,)
        if self.kind == "azimuth_thruster":
            return (self.state.target_command[0] if self.state.target_command else 0., self.state.angle_rad)
        return (self.state.target_command[0] if self.state.target_command else 0.,)

    def _thrust(self, command: float, flow: np.ndarray, density: float, rpm: float | None = None) -> float:
        model = self.propulsion.get("model", "power_law")
        if model == "static_map":
            points = self.propulsion["points"]
            return float(np.interp(command, [p[0] for p in points], [p[1] for p in points]))
        if model == "open_water":
            n = (command * float(self.propulsion["max_rpm"]) if rpm is None else rpm) / 60.
            if abs(n) < 1e-10:
                return 0.
            diameter = float(self.propulsion["diameter_m"])
            axis = self.direction
            advance = float(np.dot(flow, axis))
            j = advance / (abs(n) * diameter)
            if "kt_points" in self.propulsion:
                pts = self.propulsion["kt_points"]
                kt = float(np.interp(j, [p[0] for p in pts], [p[1] for p in pts]))
            else:
                kt = float(np.polynomial.polynomial.polyval(j, self.propulsion["kt_coefficients_ascending"]))
            return density * n * abs(n) * diameter**4 * kt
        if model == "power_estimate":
            power = float(self.propulsion["rated_power_w"])
            diameter = float(self.propulsion["diameter_m"])
            area = math.pi * diameter**2 / 4
            maximum = (2 * density * area * (power * float(self.propulsion.get("efficiency", .55)))**2)**(1/3)
        else:
            maximum = float(self.propulsion.get("max_forward_n", 1.))
        reverse = float(self.propulsion.get("max_reverse_n", maximum))
        exponent = float(self.propulsion.get("exponent", 2.))
        return (maximum if command >= 0 else reverse) * math.copysign(abs(command)**exponent, command)

    def _angle_target(self, requested: float, current: float) -> float:
        bounds = self.command_bounds[-1]
        if self.kind == "azimuth_thruster" and self.steering.get("continuous", False):
            target = current + _wrap(requested - current)
        else:
            target = min(max(requested, bounds[0]), bounds[1])
            if self.kind == "azimuth_thruster":
                for shift in (-2 * math.pi, 2 * math.pi):
                    candidate = target + shift
                    if bounds[0] <= candidate <= bounds[1] and abs(candidate-current) < abs(target-current):
                        target = candidate
        for lo, hi in self.steering.get("forbidden_sectors_rad", ()):
            if lo < target < hi:
                target = lo if abs(target-lo) <= abs(target-hi) else hi
        return target

    def _advance(self, target: float, current: float, dt: float, tau_key: str, rate_key: str) -> tuple[float, bool]:
        if tau_key in self.dynamics:
            target = current + (target-current) * (1-math.exp(-dt/float(self.dynamics[tau_key])))
        limited = False
        if rate_key in self.dynamics:
            delta = float(self.dynamics[rate_key]) * dt
            limited = abs(target-current) > delta
            target = min(max(target, current-delta), current+delta)
        return target, limited

    def predict(self, command: tuple[float, ...], motion: VesselMotion, *, state: DeviceState | None = None,
                source_thrust: dict[str, float] | None = None, dt: float | None = None) -> tuple[DeviceLoad, DeviceState]:
        if len(command) != len(self.command_bounds) or not all(math.isfinite(x) for x in command):
            raise PhysicalValidationError(f"Invalid command for {self.id}")
        old = state or self.state
        if dt is not None and (dt <= 0 or not math.isfinite(dt)):
            raise PhysicalValidationError("dt must be positive")
        flow = motion.local_flow(self.position)
        target = list(command)
        saturated = False
        for i, (lo, hi) in enumerate(self.command_bounds):
            if self.kind == "azimuth_thruster" and i == 1 and self.steering.get("continuous", False):
                continue
            clipped = min(max(target[i], lo), hi)
            saturated |= clipped != target[i]
            target[i] = clipped
        rate_limited = False
        angle = old.angle_rad
        if self.kind in {"rudder", "azimuth_thruster"}:
            index = 0 if self.kind == "rudder" else 1
            desired = self._angle_target(target[index], old.angle_rad)
            angle, rate_limited = self._advance(desired, old.angle_rad, dt, "steering_tau_s", "steering_rate_radps") if dt else (desired, False)
            target[index] = desired
        rpm = old.rpm
        thrust = 0.
        if self.kind == "rudder":
            axial, lateral = float(flow[0]), float(flow[1])
            for source in self.sources:
                if source_thrust and source in source_thrust:
                    p = source_thrust[source]
                    disk_area = float(self.geometry.get("slipstream_disk_area_m2", 0.))
                    overlap = float(self.geometry.get("slipstream_overlap", 1.))
                    scale = float(self.geometry.get("slipstream_scale", 1.))
                    if disk_area > 0 and p > 0:
                        induced = .5 * (math.sqrt(axial*axial + 2*p/(motion.water_density*disk_area))-abs(axial))
                        axial += 2*induced*max(0., min(1., overlap))*max(0., min(2., scale))
            speed2 = axial*axial + lateral*lateral
            alpha = angle - math.atan2(lateral, axial)
            aspect = float(self.geometry["aspect_ratio"])
            lift_gradient = 2*math.pi*aspect/(aspect+2)
            normal = .5*motion.water_density*float(self.geometry["area_m2"])*speed2*lift_gradient*math.sin(alpha)
            normal *= float(self.geometry.get("force_scale", 1.))
            normal *= math.tanh(max(axial, 0.) / .2) if axial < .2 else 1.
            force = np.array((-normal*math.sin(angle), normal*math.cos(angle), 0.))
        else:
            if self.propulsion.get("model") == "open_water" and self.dynamics.get("rpm_tau_s"):
                rpm_target = target[0] * float(self.propulsion["max_rpm"])
                rpm, limited = self._advance(rpm_target, old.rpm, dt, "rpm_tau_s", "rpm_rate_per_s") if dt else (rpm_target, False)
                rate_limited |= limited
                thrust = self._thrust(target[0], flow, motion.water_density, rpm=rpm)
            else:
                desired_thrust = self._thrust(target[0], flow, motion.water_density)
                thrust, limited = self._advance(desired_thrust, old.thrust_n, dt, "thrust_tau_s", "thrust_rate_nps") if dt else (desired_thrust, False)
                rate_limited |= limited
            direction = self.direction
            if self.kind == "azimuth_thruster":
                c, s = math.cos(angle), math.sin(angle)
                direction = np.array((c*direction[0]-s*direction[1], s*direction[0]+c*direction[1], direction[2]))
            force = thrust * direction
        moment = np.cross(self.position, force)
        next_state = DeviceState(thrust, rpm, angle, tuple(command), tuple(target), saturated, rate_limited)
        load = DeviceLoad(self.id, tuple(force), tuple(moment), tuple(flow), thrust, angle, rpm,
                          tuple(command), tuple(target), saturated, rate_limited, self.status)
        return load, next_state


class ActuatorSet:
    def __init__(self, specs: list[dict[str, Any]], *, allocation: dict[str, Any] | None = None):
        self.devices = tuple(PhysicalDevice(s) for s in specs)
        ids = [d.id for d in self.devices]
        if len(ids) != len(set(ids)):
            raise DuplicateIdentityError("Actuator IDs must be unique")
        self.by_id = {d.id: d for d in self.devices}
        for d in self.devices:
            if any(source not in self.by_id or source == d.id for source in d.sources):
                raise PhysicalValidationError("Invalid rudder inflow source")
        self.allocation = allocation or {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ActuatorSet":
        data = yaml.safe_load(Path(path).read_text())
        config = data["actuator_system"]
        if config.get("schema_version") != "manta-actuator-v1":
            raise PhysicalValidationError("Unsupported actuator schema")
        return cls(config["actuators"], allocation=config.get("control_allocation"))

    def predict_wrench(self, commands: dict[str, tuple[float, ...]], motion: VesselMotion,
                       *, dt: float | None = None, commit: bool = False) -> WrenchPrediction:
        if set(commands) != set(self.by_id):
            raise PhysicalValidationError("Exactly one command per actuator is required")
        if commit and dt is None:
            raise PhysicalValidationError("Committing actuator dynamics requires dt")
        loads: dict[str, DeviceLoad] = {}
        states: dict[str, DeviceState] = {}
        for device in self.devices:
            if device.kind == "rudder":
                continue
            loads[device.id], states[device.id] = device.predict(commands[device.id], motion, dt=dt)
        source_thrust = {name: load.thrust_n for name, load in loads.items()}
        for device in self.devices:
            if device.kind == "rudder":
                loads[device.id], states[device.id] = device.predict(commands[device.id], motion, source_thrust=source_thrust, dt=dt)
        if commit:
            for device in self.devices:
                device.state = states[device.id]
        ordered = tuple(loads[d.id] for d in self.devices)
        total = tuple(math.fsum((load.force_frd + load.moment_frd)[i] for load in ordered) for i in range(6))
        return WrenchPrediction(ordered, total)

    def current_commands(self) -> dict[str, tuple[float, ...]]:
        return {d.id: d.current_command() for d in self.devices}

    def flat_bounds(self, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = [], []
        for d in self.devices:
            current = d.current_command()
            for i, (lo, hi) in enumerate(d.command_bounds):
                if dt is not None:
                    rate = d.dynamics.get("steering_rate_radps" if d.kind == "rudder" or (d.kind == "azimuth_thruster" and i == 1) else "command_rate_per_s")
                    if rate is not None:
                        lo, hi = max(lo, current[i]-rate*dt), min(hi, current[i]+rate*dt)
                    if i == 0 and d.kind != "rudder" and "thrust_rate_nps" in d.dynamics and d.propulsion.get("model", "power_law") == "power_law":
                        force = d.state.thrust_n
                        delta_force = float(d.dynamics["thrust_rate_nps"])*dt
                        positive = float(d.propulsion.get("max_forward_n", 1.))
                        negative = float(d.propulsion.get("max_reverse_n", positive))
                        exponent = float(d.propulsion.get("exponent", 2.))
                        def inverse(thrust: float) -> float:
                            gain = positive if thrust >= 0 else negative
                            return math.copysign((abs(thrust)/gain)**(1/exponent), thrust)
                        lo = max(lo, inverse(force-delta_force))
                        hi = min(hi, inverse(force+delta_force))
                lower.append(lo); upper.append(hi)
        return np.asarray(lower), np.asarray(upper)

    def unflatten(self, vector: np.ndarray) -> dict[str, tuple[float, ...]]:
        result = {}; offset = 0
        for d in self.devices:
            count = len(d.command_bounds)
            result[d.id] = tuple(float(x) for x in vector[offset:offset+count])
            offset += count
        return result

    def authority_report(self, motion: VesselMotion, cruise_speed: float = 2.) -> dict[str, Any]:
        report = {}
        for speed in (0., .25*cruise_speed, cruise_speed):
            state = VesselMotion((speed, 0., 0.), motion.angular_velocity_frd, motion.current_frd,
                                 motion.heading_rad, motion.water_density)
            lo, hi = self.flat_bounds()
            center = np.clip(np.zeros_like(lo), lo, hi)
            columns = []
            for i in range(len(center)):
                step = max((hi[i]-lo[i])*1e-4, 1e-6)
                a, b = center.copy(), center.copy()
                a[i] = max(lo[i], a[i]-step); b[i] = min(hi[i], b[i]+step)
                if b[i] == a[i]:
                    columns.append(np.zeros(3)); continue
                wa = self.predict_wrench(self.unflatten(a), state).wrench_frd
                wb = self.predict_wrench(self.unflatten(b), state).wrench_frd
                columns.append((np.asarray(wb)[[0, 1, 5]]-np.asarray(wa)[[0, 1, 5]])/(b[i]-a[i]))
            jac = np.asarray(columns).T
            samples = []
            for i in range(3):
                for sign in (-1, 1):
                    z = center.copy()
                    if jac.size:
                        z = np.where(sign*jac[i] >= 0, hi, lo)
                    samples.append(np.asarray(self.predict_wrench(self.unflatten(z), state).wrench_frd)[[0, 1, 5]])
            report[str(speed)] = {"max_surge_n": max(x[0] for x in samples),
                                  "max_sway_n": max(x[1] for x in samples),
                                  "max_yaw_nm": max(x[2] for x in samples),
                                  "min_yaw_nm": min(x[2] for x in samples),
                                  "jacobian": jac.tolist(), "rank": int(np.linalg.matrix_rank(jac)),
                                  "condition": float(np.linalg.cond(jac)) if jac.size else math.inf}
        return report


@dataclass(frozen=True)
class AllocationResult:
    commands: dict[str, tuple[float, ...]]
    requested_wrench: tuple[float, ...]
    achieved_wrench: tuple[float, ...]
    residual: tuple[float, ...]
    success: bool
    solver_status: str
    saturated: bool


class ControlAllocator:
    def __init__(self, actuators: ActuatorSet):
        self.actuators = actuators
        self.last_valid = actuators.current_commands()

    def allocate(self, requested_wrench: tuple[float, ...], motion: VesselMotion, dt: float) -> AllocationResult:
        desired = _vec(requested_wrench, 6)
        indices = np.array((0, 1, 5)) if np.allclose(desired[[2, 3, 4]], 0) else np.arange(6)
        lo, hi = self.actuators.flat_bounds(dt)
        previous = np.concatenate([self.last_valid[d.id] for d in self.actuators.devices])
        previous = np.clip(previous, lo, hi)
        weight = np.asarray(self.actuators.allocation.get("weights", [1.]*6), dtype=float)[indices]
        delta_penalty = float(self.actuators.allocation.get("delta_penalty", 1e-6))
        effort_penalty = float(self.actuators.allocation.get("effort_penalty", 1e-8))
        fixed = all(d.kind in {"fixed_thruster", "tunnel_thruster"} for d in self.actuators.devices)
        linear = fixed and all(d.propulsion.get("model", "power_law") == "power_law" and
                               float(d.propulsion.get("exponent", 2.)) == 1. and
                               float(d.propulsion.get("max_forward_n", 1.)) == float(d.propulsion.get("max_reverse_n", d.propulsion.get("max_forward_n", 1.)))
                               for d in self.actuators.devices)
        try:
            if linear:
                cols = []
                for d in self.actuators.devices:
                    gain = float(d.propulsion.get("max_forward_n", 1.))
                    f = gain*d.direction
                    cols.append(np.r_[f, np.cross(d.position, f)][indices])
                matrix = np.asarray(cols).T
                a = np.vstack((weight[:, None]*matrix, math.sqrt(delta_penalty)*np.eye(len(lo)), math.sqrt(effort_penalty)*np.eye(len(lo))))
                b = np.r_[weight*desired[indices], math.sqrt(delta_penalty)*previous, np.zeros(len(lo))]
                solution = lsq_linear(a, b, bounds=(lo, hi), method="trf")
                vector, success, status = solution.x, bool(solution.success), f"fixed:{solution.status}"
            else:
                seed = previous.copy()
                offset = 0
                for device in self.actuators.devices:
                    if device.kind != "rudder" and abs(seed[offset]) < 1e-8:
                        seed[offset] = np.clip(.2, lo[offset], hi[offset])
                    offset += len(device.command_bounds)
                def residual(z: np.ndarray) -> np.ndarray:
                    wrench = np.asarray(self.actuators.predict_wrench(self.actuators.unflatten(z), motion).wrench_frd)
                    return np.r_[weight*(wrench[indices]-desired[indices]),
                                 math.sqrt(delta_penalty)*(z-previous), math.sqrt(effort_penalty)*z]
                solution = least_squares(residual, seed, bounds=(lo, hi), max_nfev=400)
                vector = solution.x
                success = bool(solution.success or np.linalg.norm(residual(vector)[:len(indices)]) < 1e-2)
                status = f"nonlinear:{solution.status}"
            if not success or not np.all(np.isfinite(vector)):
                raise RuntimeError(status)
            commands = self.actuators.unflatten(vector)
            self.last_valid = commands
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            commands = self.last_valid
            success, status = False, f"fallback:{exc}"
        achieved = np.asarray(self.actuators.predict_wrench(commands, motion).wrench_frd)
        residual_vector = desired-achieved
        saturated = bool(np.any(np.isclose(vector if success else previous, lo)) or np.any(np.isclose(vector if success else previous, hi)))
        return AllocationResult(commands, tuple(desired), tuple(achieved), tuple(residual_vector), success, status, saturated)


@dataclass
class SpeedHeadingController:
    effective_surge_mass_kg: float
    yaw_inertia_kgm2: float
    surge_damping_nspm: float
    yaw_damping_nms: float
    speed_frequency_radps: float = .3
    heading_frequency_radps: float = .3
    damping_ratio: float = 1.
    speed_integral: float = 0.

    def desired_wrench(self, desired_speed: float, desired_heading: float, motion: VesselMotion,
                       dt: float, allocation_feedback: AllocationResult | None = None) -> tuple[float, ...]:
        if dt <= 0:
            raise PhysicalValidationError("Controller dt must be positive")
        kp = max(0., 2*self.damping_ratio*self.speed_frequency_radps*self.effective_surge_mass_kg-self.surge_damping_nspm)
        ki = self.effective_surge_mass_kg*self.speed_frequency_radps**2
        error = desired_speed-motion.velocity_frd[0]
        if allocation_feedback is None or abs(allocation_feedback.residual[0]) < max(1., .05*abs(allocation_feedback.requested_wrench[0])):
            self.speed_integral += error*dt
        else:
            self.speed_integral -= .2*allocation_feedback.residual[0]/max(ki, 1e-9)*dt
        self.speed_integral = float(np.clip(self.speed_integral, -1e6/max(ki, 1e-9), 1e6/max(ki, 1e-9)))
        surge = kp*error + ki*self.speed_integral
        kpsi = self.yaw_inertia_kgm2*self.heading_frequency_radps**2
        kr = max(0., 2*self.damping_ratio*self.heading_frequency_radps*self.yaw_inertia_kgm2-self.yaw_damping_nms)
        yaw = kpsi*_wrap(desired_heading-motion.heading_rad)-kr*motion.angular_velocity_frd[2]
        return (surge, 0., 0., 0., 0., yaw)


class ActuatorPipeline:
    """Control updates at allocation rate; device dynamics update every physics step."""

    def __init__(self, actuators: ActuatorSet, controller: SpeedHeadingController | None = None,
                 allocation_period_s: float = .1):
        self.actuators = actuators
        self.allocator = ControlAllocator(actuators)
        self.controller = controller
        self.allocation_period_s = allocation_period_s
        self.elapsed = math.inf
        self.commands = actuators.current_commands()
        self.allocation_feedback: AllocationResult | None = None

    def step(self, mode: str, action: Any, motion: VesselMotion, dt: float) -> WrenchPrediction:
        if mode == "DIRECT_ACTUATOR":
            self.commands = action
        elif mode in {"DESIRED_WRENCH", "DESIRED_SPEED_HEADING"}:
            self.elapsed += dt
            if self.elapsed >= self.allocation_period_s:
                if mode == "DESIRED_SPEED_HEADING":
                    if self.controller is None:
                        raise PhysicalValidationError("Speed/heading mode requires controller")
                    desired = self.controller.desired_wrench(action[0], action[1], motion, self.elapsed, self.allocation_feedback)
                else:
                    desired = tuple(action)
                self.allocation_feedback = self.allocator.allocate(desired, motion, self.elapsed)
                self.commands = self.allocation_feedback.commands
                self.elapsed = 0.
        else:
            raise PhysicalValidationError(f"Unknown action mode: {mode}")
        return self.actuators.predict_wrench(self.commands, motion, dt=dt, commit=True)
