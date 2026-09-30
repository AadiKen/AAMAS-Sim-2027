"""Fixed local-runner worker. Invoked only with a persisted job request."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from bcod_sim.actuators.autopilot import HighLevelCommand
from bcod_sim.config.resolver import resolve
from bcod_sim.core.errors import PhysicalValidationError, PolicyContractMismatchError
from bcod_sim.frames.geodesy import geodetic_to_ned
from bcod_sim.logging.recorder import RunRecorder
from bcod_sim.policy.bundle import PolicyBundle
from bcod_sim.policy.runtime import ONNXPolicyRuntime
from bcod_sim.web.runner import write_json
from bcod_sim.web.runtime_factory import build_engine
from bcod_sim.web.service import build_registry, serialize_frame


def _sensor_observation(engine, frame, name: str) -> list[float]:
    by_type = {}
    for (owner, _), packet in engine.latest_packets.items():
        if owner == engine.vessels[name].vessel_id:
            by_type[packet.sensor_type] = packet
    if not {"gps", "imu", "sonar"}.issubset(by_type):
        raise PolicyContractMismatchError("Navigation policy requires delivered GPS, IMU and sonar packets")
    gps, imu, sonar = (by_type[k] for k in ("gps", "imu", "sonar"))
    gps_definition = next(d for d in engine.resolved.definitions
        if d.kind == "sensor" and d.payload.get("kind") == "gps")
    origin = gps_definition.payload["origin_wgs84_rad_m"]
    lat_lon_alt = tuple(float(x) for x in gps.values["wgs84_lat_lon_alt"])
    north, east, _ = geodetic_to_ned(*lat_lon_alt, tuple(origin))
    velocity = [float(x) for x in gps.values["velocity_ned_mps"]]
    yaw_rate = float(imu.values["angular_rate_mount_radps"][2])
    range_min = min(float(x) for x in sonar.values["range_m"])
    task = next(d for d in engine.resolved.definitions if d.kind == "task")
    goal = task.payload["target_ned_m"]
    result = [north, east, velocity[0], velocity[1], yaw_rate, range_min,
              float(goal[0]), float(goal[1])]
    if not all(math.isfinite(x) for x in result):
        raise PolicyContractMismatchError("Navigation policy observation is nonfinite")
    return result


def _policy_runtime(root: Path, policy_id: str, engine):
    if not policy_id or not policy_id.replace("-", "").replace("_", "").isalnum():
        raise PolicyContractMismatchError("Invalid policy ID")
    bundle_root = root/"policies"/policy_id
    manifest = json.loads((bundle_root/"manifest.json").read_text())
    bundle = PolicyBundle.load(bundle_root,
        expected_observation_hash=manifest["observation_contract_hash"],
        expected_action_hash=manifest["action_contract_hash"])
    obs = bundle.observation_contract; action = bundle.action_contract
    if obs.get("adapter") != "navigation_sensors_v1" or action.get("adapter") != "high_level_v1":
        raise PolicyContractMismatchError("Unsupported web policy adapter")
    if obs.get("required_sensors") != ["gps", "imu", "sonar"] or obs.get("features") != [
        "north_m", "east_m", "velocity_north_mps", "velocity_east_mps", "yaw_rate_radps",
        "sonar_min_range_m", "goal_north_m", "goal_east_m"]:
        raise PolicyContractMismatchError("Navigation observation contract mismatch")
    if action.get("fields") != ["desired_speed_mps", "desired_heading_rad"]:
        raise PolicyContractMismatchError("High-level action contract mismatch")
    vessel = engine.resolved.config.vessels[0]
    sensor_kinds = {d.payload.get("kind") for d in engine.resolved.definitions
                    if d.kind == "sensor" and f"{d.id}@{d.version}" in vessel.sensors}
    if not {"gps", "imu", "sonar"}.issubset(sensor_kinds) or vessel.controller.mode != "high_level":
        raise PolicyContractMismatchError("Vessel sensors or control interface are incompatible")
    return bundle, ONNXPolicyRuntime(bundle)


def run(request_path: Path, project_root: Path, artifact_root: Path) -> None:
    job = request_path.parent
    request = json.loads(request_path.read_text())
    run_id = request["run_id"]
    status_path = job/"status.json"
    status = json.loads(status_path.read_text())
    status.update(state="RUNNING", started_at=datetime.now(timezone.utc).isoformat(), pid=__import__("os").getpid())
    write_json(status_path, status)
    stage = "resolve"
    try:
        resolved = resolve(request["config"], build_registry(request["definitions"]))
        if resolved.content_hash != request["experiment_hash"]:
            raise PhysicalValidationError("Experiment hash changed after submission")
        stage = "construct"
        engine = build_engine(resolved)
        stage = "reset"
        frame = engine.reset(seed=request["seed"])
        name = resolved.config.vessels[0].instance_id
        task = next(d for d in resolved.definitions if d.kind == "task")
        goal = task.payload["target_ned_m"]
        bundle = runtime = None
        if request["mode"] == "policy_evaluation":
            stage = "policy_compatibility"
            bundle, runtime = _policy_runtime(artifact_root, request["policy_id"], engine)
        stage = "record"
        recorder = RunRecorder(artifact_root, run_id=run_id, resolved=resolved,
            scenario=engine.scenario, initial_frame=frame, project_root=project_root,
            policy=bundle, observation_contract_hash=bundle.manifest.observation_contract_hash if bundle else None,
            action_contract_hash=bundle.manifest.action_contract_hash if bundle else None)
        success = False; collisions = 0; reward_total = 0.0
        with (job/"frames.jsonl").open("w", buffering=1) as frames, \
             (job/"metrics.jsonl").open("w", buffering=1) as metrics, \
             (job/"logs.jsonl").open("w", buffering=1) as logs:
            def emit(stream, value): stream.write(json.dumps(value, allow_nan=False)+"\n")
            emit(logs, {"time": datetime.now(timezone.utc).isoformat(), "message": "Simulator reset"})
            emit(frames, {**serialize_frame(frame), "action": None, "observation": None,
                          "events": [], "actuator_commands": None})
            while not frame.terminated:
                if (job/"stop.requested").exists():
                    emit(logs, {"time": datetime.now(timezone.utc).isoformat(), "message": "Graceful stop requested"})
                    break
                observation = None
                stage = "control"
                if runtime is None:
                    state = frame.states[name]
                    north, east = (float(x) for x in state.position_ned[:2])
                    heading = math.atan2(float(goal[1])-east, float(goal[0])-north)
                    command = HighLevelCommand(1.0, heading)
                else:
                    observation = _sensor_observation(engine, frame, name)
                    output = runtime.infer({"observation": np.asarray([observation], dtype=np.float32)})["action"]
                    values = np.asarray(output).reshape(-1)
                    if values.shape != (2,) or not np.isfinite(values).all() or not (0 <= values[0] <= 2) or abs(values[1]) > math.pi:
                        raise PolicyContractMismatchError("Policy action is outside high-level bounds")
                    command = HighLevelCommand(float(values[0]), float(values[1]))
                action = {name: command}
                stage = "simulator_step"
                frame = engine.step(action)
                stage = "record"
                recorder.record_frame(frame, actions=action)
                success |= bool(frame.task_evaluation and frame.task_evaluation.success)
                collisions += len(frame.contact_events)
                reward_total += sum(frame.reward.per_agent_total.values()) if frame.reward else 0.0
                commands = {component: vars(value) if hasattr(value, "__dict__") else str(value)
                            for (env_id, vessel_id, component), value in engine.held_commands.items()
                            if env_id == 0 and vessel_id == engine.vessels[name].vessel_id
                            and component != "__physical_action__"}
                emit(frames, {**serialize_frame(frame), "action": {name: vars(command)},
                    "observation": observation, "events": [event.contact_id for event in frame.contact_events],
                    "actuator_commands": commands,
                    "control_diagnostics": str(engine.last_actuator_diagnostics.get(name, ""))})
                emit(metrics, {"step": frame.master_step, "time_s": frame.sim_time_s,
                    "reward": sum(frame.reward.per_agent_total.values()) if frame.reward else 0.0,
                    "success": success, "collisions": collisions})
                if frame.master_step % 5 == 0 or frame.terminated:
                    status.update(step=frame.master_step, sim_time_s=frame.sim_time_s,
                        reward_total=reward_total, success=success, collisions=collisions)
                    write_json(status_path, status)
                if request["step_delay_s"]: time.sleep(request["step_delay_s"])
            artifact = recorder.close(final_frame=frame, external_stop=not frame.terminated)
            status.update(state="COMPLETED" if frame.terminated else "STOPPED",
                step=frame.master_step, sim_time_s=frame.sim_time_s, reward_total=reward_total,
                success=success, collisions=collisions, artifact_path=str(artifact),
                termination_reason=frame.termination_reason.value if frame.termination_reason else "external_stop",
                ended_at=datetime.now(timezone.utc).isoformat())
            emit(logs, {"time": status["ended_at"], "message": f"Run {status['state'].lower()}"})
            write_json(status_path, status)
    except BaseException as exc:
        category = ("POLICY_COMPATIBILITY_ERROR" if isinstance(exc, PolicyContractMismatchError) or stage == "policy_compatibility" else
                    "CONFIGURATION_ERROR" if stage == "resolve" else
                    "CONTROL_INTERFACE_ERROR" if stage == "control" else
                    "SIMULATOR_ERROR" if stage in {"construct", "reset", "simulator_step"} else
                    "COMPUTE_ERROR")
        status.update(state="FAILED", error=f"{type(exc).__name__}: {exc}",
                      failure_category=category, failure_stage=stage,
                      ended_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        with (job/"logs.jsonl").open("a") as logs:
            logs.write(json.dumps({"time": status["ended_at"], "message": status["error"]})+"\n")
        traceback.print_exc()
        raise


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("worker requires request, project root, artifact root")
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
