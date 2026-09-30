"""Small, explicit qualification ladder over the authoritative episode engine."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bcod_sim import __version__
from bcod_sim.actuators.autopilot import HighLevelCommand
from bcod_sim.config.resolver import resolve
from bcod_sim.web.runtime_factory import VesselRuntime, build_engine
from bcod_sim.web.service import build_registry


def _result(category: str, name: str, status: str, *, message: str = "",
            metrics: dict | None = None, path: str = "", code: str = "") -> dict:
    return {"category": category, "name": name, "status": status,
            "message": message, "metrics": metrics or {}, "path": path, "code": code}


def qualify(config: dict[str, Any], definitions: list[dict[str, Any]], *,
            artifact_root: str | Path | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    checks: list[dict] = []
    try:
        resolved = resolve(config, build_registry(definitions))
        checks.append(_result("schema", "Resolved experiment", "PASS"))
    except Exception as exc:
        checks.append(_result("schema", "Resolved experiment", "FAIL", message=str(exc), code="CONFIG_SCHEMA"))
        return {"status": "FAIL", "checks": checks, "timestamp": now,
                "software_version": __version__, "schema_version": 1}

    for definition in resolved.definitions:
        if definition.kind != "vessel":
            continue
        try:
            vessel = VesselRuntime.model_validate(dict(definition.payload))
            inertia = np.asarray(vessel.inertia_cg_kg_m2, dtype=float)
            positive = np.allclose(inertia, inertia.T, atol=1e-9) and np.linalg.eigvalsh(inertia).min() > 0
            checks.append(_result("semantic", f"{definition.id} mass/inertia",
                "PASS" if positive else "FAIL", path=f"/definitions/{definition.id}/payload/inertia_cg_kg_m2",
                code="INERTIA_NOT_POSITIVE_DEFINITE" if not positive else "",
                metrics={"mass_kg": vessel.mass_kg, "minimum_inertia_eigenvalue": float(np.linalg.eigvalsh(inertia).min())}))
        except Exception as exc:
            checks.append(_result("semantic", f"{definition.id} physical parameters", "FAIL",
                                  message=str(exc), code="VESSEL_PHYSICAL_SCHEMA"))

    task_definition = next((d for d in resolved.definitions if d.kind == "task"), None)
    for vessel in resolved.config.vessels:
        sensor_kinds = {str(d.payload.get("kind")) for d in resolved.definitions
                        if d.kind == "sensor" and f"{d.id}@{d.version}" in vessel.sensors}
        if task_definition and task_definition.payload.get("kind") == "waypoint":
            missing = sorted({"gps", "imu", "sonar"} - sensor_kinds)
            checks.append(_result("cross_object", f"{vessel.instance_id} navigation sensors",
                "FAIL" if missing else "PASS", message=f"Missing sensors: {', '.join(missing)}" if missing else "",
                path=f"/vessels/{vessel.instance_id}/sensors", code="MISSING_NAVIGATION_SENSOR" if missing else ""))
        vessel_definition = next((d for d in resolved.definitions if d.kind == "vessel" and
                                  f"{d.id}@{d.version}" == vessel.definition), None)
        collision = vessel_definition.payload.get("collision", {}) if vessel_definition else {}
        radius = (float(collision.get("radius_m", 0)) if collision.get("kind") == "sphere" else
                  math.sqrt(sum(float(x)**2 for x in collision.get("half_extents_m", (0, 0, 0)))))
        position = vessel.spawn.ned_m
        overlaps = []
        for obstacle in resolved.config.world.static_entities:
            if not obstacle.collision_enabled: continue
            shape = obstacle.shape
            obstacle_radius = (shape.radius_m if shape.kind == "sphere" else
                               math.sqrt(sum(x*x for x in shape.half_extents_m)))
            distance = math.sqrt(sum((a-b)**2 for a, b in zip(position, obstacle.position_ned_m)))
            if distance < radius + obstacle_radius: overlaps.append(obstacle.id)
        bottom = resolved.config.world.bathymetry
        grounded = bool(bottom and bottom.kind == "flat" and position[2] + radius >= bottom.bottom_ned_z_m)
        checks.append(_result("cross_object", f"{vessel.instance_id} spawn clearance",
            "FAIL" if overlaps or grounded else "PASS",
            message=f"Overlap: {', '.join(overlaps)}" if overlaps else "Vessel intersects seabed" if grounded else "",
            path=f"/vessels/{vessel.instance_id}/spawn/ned_m",
            code="INVALID_SPAWN" if overlaps or grounded else ""))

    if any(c["status"] == "FAIL" for c in checks):
        outcome = "FAIL"
    else:
        try:
            engine = build_engine(resolved)
            initial = engine.reset()
            checks.append(_result("runtime", "T0 construction/reset", "PASS"))
            if len(resolved.config.vessels) != 1 or resolved.config.vessels[0].controller.mode != "high_level":
                checks.append(_result("runtime", "T1 control response", "NOT_RUN",
                    message="MVP response test supports one high-level vessel"))
            else:
                name = resolved.config.vessels[0].instance_id
                def trajectory(command):
                    episode = build_engine(resolved); episode.reset()
                    frames = []
                    for _ in range(min(10, resolved.config.simulation.max_master_steps)):
                        frame = episode.step({name: command}); frames.append(frame)
                        if frame.terminated: break
                    return frames
                zero = trajectory(HighLevelCommand(0, 0))
                surge = trajectory(HighLevelCommand(1, 0))
                yaw = trajectory(HighLevelCommand(0, .5))
                zero_speed = abs(float(zero[-1].states[name].nu_body[0]))
                surge_speed = float(surge[-1].states[name].nu_body[0])
                yaw_rate = float(yaw[-1].states[name].nu_body[5])
                checks.append(_result("runtime", "T1 zero command stability",
                    "PASS" if zero_speed < .05 else "FAIL", code="ZERO_COMMAND_DRIFT" if zero_speed >= .05 else "",
                    metrics={"surge_speed_mps": zero_speed}))
                checks.append(_result("runtime", "T1 positive surge response",
                    "PASS" if surge_speed > .01 else "FAIL", code="SURGE_SIGN" if surge_speed <= .01 else "",
                    metrics={"surge_speed_mps": surge_speed}))
                checks.append(_result("runtime", "T1 positive yaw response",
                    "PASS" if yaw_rate > .01 else "FAIL", code="YAW_SIGN" if yaw_rate <= .01 else "",
                    metrics={"yaw_rate_radps": yaw_rate}))
                repeat = trajectory(HighLevelCommand(1, 0))
                first = [tuple(frame.states[name].position_ned.tolist()) for frame in surge]
                second = [tuple(frame.states[name].position_ned.tolist()) for frame in repeat]
                checks.append(_result("runtime", "T1 deterministic repeat",
                    "PASS" if first == second else "FAIL", code="NONDETERMINISTIC_TRAJECTORY" if first != second else ""))
                task = next(d for d in resolved.definitions if d.kind == "task")
                if task.payload.get("kind") == "waypoint":
                    goal = task.payload["target_ned_m"]
                    baseline = build_engine(resolved); current = baseline.reset()
                    completed = False; contacts = 0
                    for _ in range(min(200, resolved.config.simulation.max_master_steps)):
                        north, east = (float(x) for x in current.states[name].position_ned[:2])
                        heading = math.atan2(float(goal[1])-east, float(goal[0])-north)
                        current = baseline.step({name: HighLevelCommand(1.0, heading)})
                        completed |= bool(current.task_evaluation and current.task_evaluation.success)
                        contacts += len(current.contact_events)
                        if current.terminated: break
                    passed = completed and contacts == 0
                    checks.append(_result("runtime", "T2 scripted navigation", "PASS" if passed else "FAIL",
                        code="SCRIPTED_BASELINE_FAILED" if not passed else "",
                        metrics={"steps": current.master_step, "success": completed, "contacts": contacts}))
                else:
                    checks.append(_result("runtime", "T2 scripted navigation", "NOT_RUN",
                        message="Baseline implemented only for waypoint task"))
                checks.append(_result("runtime", "T3 learning smoke", "NOT_RUN"))
        except Exception as exc:
            constructed = any(c["name"] == "T0 construction/reset" and c["status"] == "PASS" for c in checks)
            checks.append(_result("runtime", "T1 control response" if constructed else "T0 construction/reset",
                "FAIL", message=str(exc), code="CONTROL_INTERFACE" if constructed else "SIMULATOR_CONSTRUCTION"))
        outcome = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS"

    report = {"status": outcome, "checks": checks, "timestamp": now,
              "experiment_hash": resolved.content_hash, "schema_version": 1,
              "software_version": __version__,
              "objects": [{"kind": d.kind, "id": d.id, "version": d.version,
                           "content_hash": d.content_hash} for d in resolved.definitions]}
    if artifact_root is not None:
        root = Path(artifact_root)/"qualifications"; root.mkdir(parents=True, exist_ok=True)
        (root/f"{resolved.content_hash}.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report
