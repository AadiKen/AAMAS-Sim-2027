"""The single authoritative per-environment episode stepping path."""

from dataclasses import dataclass, field
import copy
import math
from typing import Callable, Mapping

import torch

from bcod_sim.actuators.allocation import ActuatorBank
from bcod_sim.actuators.autopilot import HeadingSpeedAutopilot, HighLevelCommand
from bcod_sim.actuators.base import Actuator, ActuatorState
from bcod_sim.actuators.thruster import FixedThruster
from bcod_sim.collision.broadphase import CollisionBody
from bcod_sim.collision.shapes import Box, Capsule, CollisionShape, ConvexHull, Sphere, bounding_radius
from bcod_sim.collision.solver import ContactEvent, ContactMaterial, resolve_contacts
from bcod_sim.collision.world_adapter import seabed_collision_bodies, world_collision_bodies
from bcod_sim.config.resolver import ResolvedExperiment
from bcod_sim.core.environment_loads import ENVIRONMENT_TERMS, EnvironmentLoadModel
from bcod_sim.core.errors import (BCODSimError, ExternalDataCoverageError,
                                  ExternalDataUnavailableError, NonFiniteStateError,
                                  PhysicalValidationError)
from bcod_sim.core.lifecycle import AgentStatus, DirectAction, TerminationReason
from bcod_sim.frames.transforms import rpy_to_quaternion
from bcod_sim.dynamics.crossflow import NoCrossflow
from bcod_sim.rl.observation import ObservationContract
from bcod_sim.scenario.generator import ResolvedScenario, ScenarioTemplate, generate
from bcod_sim.sensors.base import Sensor, SensorContext, SensorPacket
from bcod_sim.sensors.scheduler import SensorScheduler
from bcod_sim.state.vessel_state import VesselState
from bcod_sim.tasks.base import RewardReport, TaskEvaluation, compose_reward
from bcod_sim.tasks.registry import build_task
from bcod_sim.world.wake import GaussianWakeEmitter, WakeEmission
from bcod_sim.world.world import ParametricWorld
from bcod_sim.world.environment import Environment
from bcod_sim.config.hashing import content_hash


@dataclass(frozen=True)
class ConstantScriptedController:
    action: DirectAction

    def command(self, step_index: int, state: VesselState) -> DirectAction:
        return self.action


@dataclass(frozen=True)
class EpisodeVessel:
    instance_id: str
    vessel_id: int
    plant: object
    collision_shape: CollisionShape
    actuators: tuple[Actuator, ...]
    sensors: tuple[Sensor, ...]
    environment_loads: EnvironmentLoadModel
    wind_loads: object | None = None
    wave_loads: object | None = None
    autopilot: HeadingSpeedAutopilot | None = None
    scripted_controller: ConstantScriptedController | None = None
    wake_emitter: GaussianWakeEmitter | None = None


@dataclass(frozen=True)
class EpisodeFrame:
    master_step: int
    sim_time_s: float
    states: Mapping[str, VesselState]
    observations: Mapping[str, Mapping[str, object]]
    pending_observations: tuple[str, ...]
    delivered_packets: tuple[SensorPacket, ...]
    reward: RewardReport | None
    task_evaluation: TaskEvaluation | None
    contact_events: tuple[ContactEvent, ...]
    terminated: bool
    termination_reason: TerminationReason | None
    grounding_telemetry: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    observation_freshness: Mapping[str, Mapping[str, Mapping[str, object]]] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeCheckpoint:
    scenario_hash: str
    master_step: int
    states: Mapping[str, VesselState]
    statuses: Mapping[str, AgentStatus]
    actuator_states: Mapping[tuple[int, int, str], ActuatorState]
    held_commands: Mapping[tuple[int, int, str], object]
    latest_packets: Mapping[tuple[int, str], SensorPacket]
    last_propulsion: Mapping[str, torch.Tensor]
    scheduler_last_step: int
    scheduler_pending: Mapping[int, list[SensorPacket]]
    scheduler_disabled_owners: frozenset[tuple[int, int]]
    wake_current: tuple[WakeEmission, ...]
    wake_generation: int
    task_snapshot: object
    terminated: bool
    termination_reason: TerminationReason | None
    sensor_states: Mapping[tuple[int, int, str], object] = field(default_factory=dict)


class EpisodeEngine:
    """Headless, web, validation, and future RL adapters all call this step method."""

    def __init__(self, resolved: ResolvedExperiment, vessels: tuple[EpisodeVessel, ...],
                 *, observation_contracts: Mapping[str, ObservationContract] | None = None,
                 contact_material: ContactMaterial | None = None, world: object | None = None) -> None:
        if not isinstance(resolved, ResolvedExperiment):
            raise TypeError("Episode engine requires ResolvedExperiment")
        self.resolved = resolved
        if resolved.config.world.source.kind == "parametric":
            if world is not None:
                raise PhysicalValidationError("Parametric configuration cannot inject a real-world bundle")
            self.world = ParametricWorld.from_resolved(resolved, env_id=0)
        else:
            from bcod_sim.data_sources.world_bundle import RealWorldBundle
            if not isinstance(world, RealWorldBundle) or world.env_id != 0:
                raise ExternalDataUnavailableError("Real-world configuration requires a resolved local world bundle")
            definitions = [d for d in resolved.definitions if d.kind == "data_product"]
            expected_hash = definitions[0].payload.get("bundle_hash") if len(definitions) == 1 else None
            if not expected_hash or expected_hash != world.content_hash:
                raise PhysicalValidationError("Real-world bundle hash does not match resolved data product")
            self.world = world
        self.vessels = {v.instance_id: v for v in vessels}
        expected = {v.instance_id for v in resolved.config.vessels}
        if len(self.vessels) != len(vessels) or set(self.vessels) != expected:
            raise PhysicalValidationError("Episode vessel identities must exactly match resolved config")
        numeric_ids = [v.vessel_id for v in vessels]
        if len(numeric_ids) != len(set(numeric_ids)) or any(v < 0 for v in numeric_ids):
            raise PhysicalValidationError("Episode vessel numeric IDs must be unique and nonnegative")
        self.config_vessels = {v.instance_id: v for v in resolved.config.vessels}
        all_actuators, all_sensors = [], []
        substep_s = resolved.config.simulation.master_dt_s / resolved.config.simulation.dynamics_substeps
        for vessel in vessels:
            authoring = self.config_vessels[vessel.instance_id]
            if vessel.plant.mode != resolved.config.simulation.dynamics_mode:
                raise PhysicalValidationError("All vessels must use the resolved environment dynamics mode")
            if not vessel.plant.envelope.min_substep_s <= substep_s <= vessel.plant.envelope.max_substep_s:
                raise PhysicalValidationError("Dynamics substep violates vessel operating envelope")
            if vessel.environment_loads is None:
                raise PhysicalValidationError("Every vessel requires explicit environment loads")
            if authoring.controller.mode == "high_level" and vessel.autopilot is None:
                raise PhysicalValidationError("High-level vessel requires autopilot")
            if authoring.controller.mode == "scripted" and vessel.scripted_controller is None:
                raise PhysicalValidationError("Scripted vessel requires controller")
            if vessel.wake_emitter is not None and (vessel.wake_emitter.env_id != 0 or
                                                    vessel.wake_emitter.source_vessel_id != vessel.vessel_id):
                raise PhysicalValidationError("Wake emitter owner mismatch")
            for component in (*vessel.actuators, *vessel.sensors):
                if component.config.env_id != 0 or component.config.owner_vessel_id != vessel.vessel_id:
                    raise PhysicalValidationError("Component owner mismatch")
            environment = Environment(self.world)
            for sensor in vessel.sensors:
                requirements = frozenset(getattr(sensor, "requirements", ()))
                environment.require(item for item in requirements if not item.startswith("vehicle."))
                if sensor.config.config_fingerprint is not None:
                    reference = next((ref for ref in authoring.sensors
                                      if ref.partition("@")[0] == sensor.config.instance_id), None)
                    definition = next((d for d in resolved.definitions if d.kind == "sensor" and
                                       reference == f"{d.id}@{d.version}"), None)
                    if definition is None:
                        raise PhysicalValidationError(f"Runtime sensor {sensor.config.instance_id} is not resolved")
                    expected_fingerprint = content_hash({"id": definition.id, "version": definition.version,
                        "source": definition.source, "payload": dict(definition.payload), "env_id": 0,
                        "owner_vessel_id": vessel.vessel_id})
                    if sensor.config.fingerprint != expected_fingerprint:
                        raise PhysicalValidationError(
                            f"Runtime sensor fingerprint mismatch: {sensor.config.instance_id}")
            all_actuators.extend(vessel.actuators)
            all_sensors.extend(vessel.sensors)
        self.all_actuators = tuple(all_actuators)
        self.all_sensors = tuple(all_sensors)
        self.contact_material = contact_material or ContactMaterial()
        self.observation_contracts = dict(observation_contracts or {})
        if set(self.observation_contracts) - expected:
            raise PhysicalValidationError("Observation contract for unknown vessel")
        task_defs = [d for d in resolved.definitions if d.kind == "task"]
        if len(task_defs) != 1:
            raise PhysicalValidationError("Resolved experiment must contain exactly one task definition")
        self.task = build_task(task_defs[0])
        scenario_defs = [d for d in resolved.definitions if d.kind == "scenario_template"]
        if resolved.config.scenario_template:
            if len(scenario_defs) != 1:
                raise PhysicalValidationError("Scenario template resolution mismatch")
            self.template = ScenarioTemplate.model_validate(dict(scenario_defs[0].payload))
        else:
            self.template = ScenarioTemplate(id="inline_fixed", version="1")
        self.scenario: ResolvedScenario | None = None
        self.states: dict[str, VesselState] = {}
        self.statuses: dict[str, AgentStatus] = {}
        self.actuator_states: dict[tuple[int, int, str], ActuatorState] = {}
        self.held_commands: dict[tuple[int, int, str], object] = {}
        self.latest_packets: dict[tuple[int, str], SensorPacket] = {}
        self.last_propulsion: dict[str, torch.Tensor] = {}
        self.scheduler: SensorScheduler | None = None
        self.master_step = 0
        self.terminated = False
        self.termination_reason: TerminationReason | None = None

    def _state_snapshot(self) -> dict[str, VesselState]:
        return {name: state.clone() for name, state in self.states.items()}

    def _load_terms(self, vessel: EpisodeVessel, state: VesselState, sim_time_s: float) -> dict[str, torch.Tensor]:
        sample = self.world.sample(state.position_ned.unsqueeze(0), sim_time_s=sim_time_s,
                                   env_id=0, receiver_vessel_id=vessel.vessel_id)
        terms = dict(vessel.environment_loads.evaluate(state, sample))
        if vessel.wind_loads is not None: terms["wind"] = vessel.wind_loads.evaluate(state,sample).tau_body
        if vessel.wave_loads is not None: terms["wave"] = vessel.wave_loads.evaluate(state,sample).tau_body
        if set(terms) != set(ENVIRONMENT_TERMS):
            raise PhysicalValidationError("Environment load model must return every declared wrench term")
        for key, value in terms.items():
            if (value.shape != (6,) or value.dtype != state.nu_body.dtype or value.device != state.nu_body.device or
                not torch.isfinite(value).all().item()):
                raise PhysicalValidationError(f"Invalid {key} environment wrench")
        return terms

    def _external(self, vessel: EpisodeVessel, state: VesselState, sim_time_s: float,
                  propulsion: torch.Tensor) -> dict[str, torch.Tensor]:
        result = self._load_terms(vessel, state, sim_time_s)
        result["propulsion"] = propulsion
        result["contact"] = state.nu_body.new_zeros((6,))  # contact is a separate impulse stage
        result["manual"] = state.nu_body.new_zeros((6,))
        return result

    def _water_velocity_ned(self, vessel: EpisodeVessel, state: VesselState, sim_time_s: float) -> torch.Tensor | None:
        if isinstance(vessel.plant.crossflow, NoCrossflow):
            return None
        sample = self.world.sample(state.position_ned.unsqueeze(0), sim_time_s=sim_time_s,
                                   env_id=0, receiver_vessel_id=vessel.vessel_id)
        if sample.current_valid is not None and not bool(sample.current_valid[0]):
            raise ExternalDataCoverageError("Vessel current query is below seabed or in a dry cell")
        return sample.current_ned_mps[0] + sample.wake_ned_mps[0] + sample.wave_orbital_ned_mps[0]

    def _grounding_telemetry(self, events: tuple[ContactEvent,...], sim_time_s: float):
        result={}
        for name,vessel in self.vessels.items():
            state=self.states[name]; sample=self.world.sample(state.position_ned[None,:],sim_time_s=sim_time_s,env_id=0,receiver_vessel_id=vessel.vessel_id)
            bottom=None if sample.bottom_ned_z_m is None else float(sample.bottom_ned_z_m[0])
            if bottom is None: continue
            shape=vessel.collision_shape
            if isinstance(shape,Sphere): lowest=float(state.position_ned[2])+shape.radius_m
            elif isinstance(shape,Box):
                from bcod_sim.frames.tensor import rotate_body_to_world
                axes=(state.position_ned.new_tensor((1.,0,0)),state.position_ned.new_tensor((0,1.,0)),state.position_ned.new_tensor((0,0,1.)))
                lowest=float(state.position_ned[2])+sum(extent*abs(float(rotate_body_to_world(axis,state.q_body_to_ned)[2])) for axis,extent in zip(axes,shape.half_extents_m))
            else: lowest=float(state.position_ned[2])+bounding_radius(shape)
            contacts=tuple(event for event in events if event.contact_id.startswith(f"0:vessel:{name}:") and "world:seabed" in event.contact_id)
            collision=getattr(getattr(getattr(self.world,"bathymetry",None),"spec",None),"collision",None) or getattr(getattr(self.world,"spec",None),"seabed_collision",None)
            result[name]={"seabed_ned_z_m":bottom,"under_keel_clearance_m":bottom-lowest,"contact_active":bool(contacts),"contacts":contacts,
                          "collision_resolution_m":None if collision is None else collision.resolution_m,
                          "collision_tile_size_m":None if collision is None else collision.tile_size_m}
        return result

    def _sensor_contexts(self, sim_time_s: float) -> dict[tuple[int, int], SensorContext]:
        contexts = {}
        for name, vessel in self.vessels.items():
            if not self.statuses[name].physical_active:
                continue
            state = self.states[name]
            propulsion = self.last_propulsion.get(name, state.nu_body.new_zeros((6,)))
            external = self._external(vessel, state, sim_time_s, propulsion)
            acceleration = vessel.plant.acceleration(state, external, self._water_velocity_ned(vessel, state, sim_time_s))
            contexts[(0, vessel.vessel_id)] = SensorContext(state, self.world, sim_time_s,
                                                              acceleration[:3], acceleration[3:])
        return contexts

    def _deliver(self, packets: tuple[SensorPacket, ...], current_time_s: float) -> tuple[
            dict[str, Mapping[str, object]], tuple[str, ...], dict[str, Mapping[str, Mapping[str, object]]]]:
        for packet in packets:
            self.latest_packets[(packet.owner_vessel_id, packet.sensor_id)] = packet
        observations, pending, freshness = {}, [], {}
        for name, contract in self.observation_contracts.items():
            if not self.statuses[name].rl_active:
                continue
            vessel_id = self.vessels[name].vessel_id
            source_packets = {sensor_id: packet for (owner, sensor_id), packet in self.latest_packets.items()
                              if owner == vessel_id}
            if any(field.sensor_id not in source_packets for field in contract.fields):
                pending.append(name)
            else:
                observations[name] = contract.assemble(source_packets)
                sensor_configs = {sensor.config.instance_id: sensor.config
                                  for sensor in self.vessels[name].sensors}
                freshness[name] = {sensor_id: packet.freshness(
                    current_time_s, sensor_configs[sensor_id].max_age_s)
                    for sensor_id, packet in source_packets.items() if sensor_id in sensor_configs}
        return observations, tuple(sorted(pending)), freshness

    def reset(self, *, seed: int | None = None, episode_index: int = 0) -> EpisodeFrame:
        effective_seed = self.resolved.config.experiment.seed if seed is None else seed
        self.scenario = generate(self.resolved, self.template, seed=effective_seed,
                                 episode_index=episode_index)
        self.world.wake = type(self.world.wake)()
        self.states = {}
        self.statuses = {}
        for spawn in self.scenario.spawns:
            vessel = self.vessels[spawn.instance_id]
            dtype, device = vessel.plant.total_mass.dtype, vessel.plant.total_mass.device
            q = rpy_to_quaternion(*spawn.rpy_rad)
            state = VesselState(torch.tensor(spawn.position_ned_m, dtype=dtype, device=device),
                                torch.tensor(q, dtype=dtype, device=device),
                                torch.tensor(spawn.nu_body, dtype=dtype, device=device))
            self.states[spawn.instance_id] = state
            mode = self.config_vessels[spawn.instance_id].controller.mode
            controller = {"direct_actuator": "policy_direct", "high_level": "policy_high_level",
                          "scripted": "scripted"}[mode]
            self.statuses[spawn.instance_id] = AgentStatus(mode != "scripted", True, controller)
        self.actuator_states = {(0, a.config.owner_vessel_id, a.config.instance_id): ActuatorState()
                                for a in self.all_actuators}
        self.held_commands = {}
        self.latest_packets = {}
        self.last_propulsion = {name: state.nu_body.new_zeros((6,)) for name, state in self.states.items()}
        self.scheduler = SensorScheduler(self.all_sensors,
            master_dt_s=self.resolved.config.simulation.master_dt_s)
        self.master_step = 0
        self.terminated = False
        self.termination_reason = None
        self.task.reset(self.states)
        delivered = self.scheduler.tick(0, self._sensor_contexts(0.0))
        observations, pending, freshness = self._deliver(delivered, 0.0)
        return EpisodeFrame(0, 0.0, self._state_snapshot(), observations, pending, delivered,
                            None, None, (), False, None, observation_freshness=freshness)

    def _action_commands(self, actions: Mapping[str, DirectAction | HighLevelCommand],
                         *, policy_tick: bool) -> None:
        active = {name for name, status in self.statuses.items() if status.rl_active}
        if policy_tick and set(actions) != active:
            raise PhysicalValidationError("Policy tick requires one action for every active RL agent")
        if not policy_tick and actions:
            raise PhysicalValidationError("Actions supplied outside common policy tick")
        for name, vessel in self.vessels.items():
            status = self.statuses[name]
            if not status.physical_active:
                continue
            action = None
            if status.controller == "scripted":
                assert vessel.scripted_controller is not None
                action = vessel.scripted_controller.command(self.master_step, self.states[name])
            elif status.rl_active and policy_tick:
                action = actions[name]
            if action is None:
                continue
            if status.controller == "policy_high_level":
                if not isinstance(action, HighLevelCommand):
                    raise PhysicalValidationError("High-level controller requires HighLevelCommand")
                from bcod_sim.dynamics.plant6 import _yaw
                thrusters = tuple(a for a in vessel.actuators if isinstance(a, FixedThruster))
                if len(thrusters) != len(vessel.actuators):
                    raise PhysicalValidationError("High-level adapter supports fixed thrusters only")
                commands = vessel.autopilot.commands(action,
                    current_speed_mps=self.states[name].nu_body[0].item(),
                    current_heading_rad=_yaw(self.states[name].q_body_to_ned).item(),
                    thrusters=thrusters)
            else:
                if not isinstance(action, DirectAction):
                    raise PhysicalValidationError("Direct controller requires DirectAction")
                commands = dict(action.commands)
            expected = {a.config.instance_id for a in vessel.actuators}
            if set(commands) != expected:
                raise PhysicalValidationError(f"Actuator commands do not match vessel {name}")
            for actuator_id, command in commands.items():
                self.held_commands[(0, vessel.vessel_id, actuator_id)] = copy.deepcopy(command)

    def disable_agent(self, name: str, *, reason: str) -> None:
        if name not in self.statuses or not self.statuses[name].rl_active or not reason:
            raise PhysicalValidationError("Can only disable an active RL agent with reason")
        status = self.statuses[name]
        behavior = self.resolved.config.task.disabled_agent_behavior
        if behavior == "terminate_episode":
            self.terminated, self.termination_reason = True, TerminationReason.AGENT_DISABLED
            self.statuses[name] = AgentStatus(False, status.physical_active, status.controller, reason)
        elif behavior == "deactivate_remove_physical":
            self.statuses[name] = AgentStatus(False, False, status.controller, reason)
            assert self.scheduler is not None
            self.scheduler.disable_owner((0, self.vessels[name].vessel_id))
        elif behavior == "continue_scripted":
            if self.vessels[name].scripted_controller is None:
                raise PhysicalValidationError("Continue-scripted behavior requires fallback controller")
            self.statuses[name] = AgentStatus(False, True, "scripted", reason)
        else:
            self.statuses[name] = AgentStatus(False, True, status.controller, reason)

    def step(self, actions: Mapping[str, DirectAction | HighLevelCommand]) -> EpisodeFrame:
        if self.scenario is None or self.scheduler is None or self.terminated:
            raise PhysicalValidationError("Episode must be active before step")
        config = self.resolved.config.simulation
        dt = config.master_dt_s
        dt_sub = dt / config.dynamics_substeps
        try:
            self._action_commands(actions, policy_tick=self.master_step % config.policy_every_n_master_steps == 0)
            self.world.wake.begin_step()
            contact_events: list[ContactEvent] = []
            for substep in range(config.dynamics_substeps):
                sub_time = self.master_step * dt + substep * dt_sub
                active_actuators = tuple(a for a in self.all_actuators if
                    self.statuses[next(name for name, vessel in self.vessels.items()
                                       if vessel.vessel_id == a.config.owner_vessel_id)].physical_active)
                bank = ActuatorBank(active_actuators)
                keys = set(bank.actuators)
                if not keys.issubset(self.held_commands):
                    raise PhysicalValidationError("Missing held actuator command")
                bank_result = bank.step({key: self.held_commands[key] for key in keys},
                                        {key: self.actuator_states[key] for key in keys}, dt_sub)
                self.actuator_states.update(bank_result.states)
                stepped = {}
                for name in sorted(self.vessels):
                    if not self.statuses[name].physical_active:
                        continue
                    vessel, state = self.vessels[name], self.states[name]
                    wrench = bank_result.wrenches_by_owner.get((0, vessel.vessel_id), (0.,)*6)
                    propulsion = state.nu_body.new_tensor(wrench)
                    self.last_propulsion[name] = propulsion.clone()
                    external = self._external(vessel, state, sub_time, propulsion)
                    water_velocity = self._water_velocity_ned(vessel, state, sub_time)
                    stepped[name] = vessel.plant.step(state, external, dt_sub, water_velocity_ned=water_velocity).state
                self.states.update(stepped)
                contact_bodies = tuple(CollisionBody(0, f"vessel:{name}", vessel.collision_shape,
                                   self.states[name], vessel.plant) for name, vessel in sorted(self.vessels.items())
                                   if self.statuses[name].physical_active)
                contact_bodies += world_collision_bodies(self.world, sim_time_s=sub_time + dt_sub)
                contact_bodies += seabed_collision_bodies(self.world, vessel_positions=tuple(
                    tuple(float(x) for x in self.states[name].position_ned) for name in sorted(stepped)))
                contact_result = resolve_contacts(contact_bodies, material=self.contact_material, dt_s=dt_sub)
                contact_events.extend(contact_result.events)
                for name in stepped:
                    self.states[name] = contact_result.states[(0, f"vessel:{name}")]
                for name, vessel in sorted(self.vessels.items()):
                    if self.statuses[name].physical_active and vessel.wake_emitter is not None:
                        emission = vessel.wake_emitter.emit(self.states[name], weight=1/config.dynamics_substeps)
                        self.world.wake.emit(emission, substep_index=substep)
            self.world.wake.swap()
            self.master_step += 1
            sim_time_s = self.master_step * dt
            delivered = self.scheduler.tick(self.master_step, self._sensor_contexts(sim_time_s))
            observations, pending, freshness = self._deliver(delivered, sim_time_s)
            active = tuple(sorted(name for name, status in self.statuses.items() if status.rl_active))
            evaluation = self.task.evaluate(self.states, active)
            reward = compose_reward(evaluation, self.resolved.config.task.reward, active)
            if evaluation.success:
                self.terminated, self.termination_reason = True, TerminationReason.TASK_SUCCESS
            elif evaluation.failure:
                self.terminated, self.termination_reason = True, TerminationReason.TASK_FAILURE
            elif self.master_step >= self.scenario.max_master_steps:
                self.terminated, self.termination_reason = True, TerminationReason.TIME_LIMIT
            return EpisodeFrame(self.master_step, sim_time_s, self._state_snapshot(), observations,
                                pending, delivered, reward, evaluation, tuple(contact_events),
                                self.terminated, self.termination_reason,
                                self._grounding_telemetry(tuple(contact_events),sim_time_s), freshness)
        except BCODSimError as exc:
            self.terminated = True
            if isinstance(exc, (ExternalDataUnavailableError, ExternalDataCoverageError)):
                self.termination_reason = TerminationReason.EXTERNAL_DATA_FAILURE
            elif isinstance(exc, NonFiniteStateError):
                self.termination_reason = TerminationReason.PHYSICS_FAILURE
            else:
                self.termination_reason = TerminationReason.CONTRACT_FAILURE
            raise

    def checkpoint(self) -> EpisodeCheckpoint:
        if self.scenario is None or self.scheduler is None or self.world.wake._next is not None:
            raise PhysicalValidationError("Checkpoint requires a master-step boundary")
        return EpisodeCheckpoint(self.scenario.content_hash, self.master_step, self._state_snapshot(),
            copy.deepcopy(self.statuses), copy.deepcopy(self.actuator_states),
            copy.deepcopy(self.held_commands), copy.deepcopy(self.latest_packets),
            {name: wrench.clone() for name, wrench in self.last_propulsion.items()},
            self.scheduler.last_step, copy.deepcopy(self.scheduler.pending),
            frozenset(self.scheduler.disabled_owners), self.world.wake._current,
            self.world.wake.generation, self.task.snapshot(), self.terminated, self.termination_reason,
            {key: copy.deepcopy(sensor.snapshot()) for key, sensor in self.scheduler.sensors.items()
             if callable(getattr(sensor, "snapshot", None))})

    def restore(self, checkpoint: EpisodeCheckpoint) -> None:
        if self.scenario is None or self.scheduler is None or self.scenario.content_hash != checkpoint.scenario_hash:
            raise PhysicalValidationError("Checkpoint scenario hash mismatch")
        if self.world.wake._next is not None:
            raise PhysicalValidationError("Restore requires a master-step boundary")
        self.master_step = checkpoint.master_step
        self.states = {name: state.clone() for name, state in checkpoint.states.items()}
        self.statuses = copy.deepcopy(dict(checkpoint.statuses))
        self.actuator_states = copy.deepcopy(dict(checkpoint.actuator_states))
        self.held_commands = copy.deepcopy(dict(checkpoint.held_commands))
        self.latest_packets = copy.deepcopy(dict(checkpoint.latest_packets))
        self.last_propulsion = {name: wrench.clone() for name, wrench in checkpoint.last_propulsion.items()}
        self.scheduler.last_step = checkpoint.scheduler_last_step
        self.scheduler.pending = copy.deepcopy(dict(checkpoint.scheduler_pending))
        self.scheduler.disabled_owners = set(checkpoint.scheduler_disabled_owners)
        for key, state in checkpoint.sensor_states.items():
            sensor = self.scheduler.sensors[key]
            sensor.restore(copy.deepcopy(state))
        self.world.wake._current = checkpoint.wake_current
        self.world.wake.generation = checkpoint.wake_generation
        self.task.restore(checkpoint.task_snapshot)
        self.terminated, self.termination_reason = checkpoint.terminated, checkpoint.termination_reason
