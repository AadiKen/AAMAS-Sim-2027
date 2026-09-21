"""Application service shared by HTTP and direct headless callers."""

from base64 import b64decode
import json
from pathlib import Path
import tempfile
from typing import Mapping

from bcod_sim.actuators.autopilot import HighLevelCommand
from bcod_sim.actuators.thruster import ThrustCommand
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.core.lifecycle import DirectAction
from bcod_sim.frames.transforms import ned_to_web
from bcod_sim.logging.artifacts import resolved_artifact
from bcod_sim.logging.recorder import RunRecorder
from bcod_sim.policy.bundle import PolicyBundle, REQUIRED_FILES
from bcod_sim.web.runtime_factory import build_engine


def build_registry(definitions: list[dict]) -> Registry:
    registry = Registry()
    for definition in definitions:
        if set(definition) != {"kind", "id", "version", "payload", "source"}:
            raise PhysicalValidationError("Definition has unknown or missing fields")
        registry.register(**definition)
    return registry


def serialize_frame(frame) -> dict:
    return {"master_step": frame.master_step, "sim_time_s": frame.sim_time_s,
        "states": {name: {"position_ned_m": state.position_ned.tolist(),
            "position_display_m": list(ned_to_web(tuple(state.position_ned.tolist()))),
            "q_body_to_ned": state.q_body_to_ned.tolist(), "nu_body": state.nu_body.tolist()}
            for name, state in sorted(frame.states.items())},
        "reward": dict(frame.reward.per_agent_total) if frame.reward else {},
        "terminated": frame.terminated,
        "termination_reason": frame.termination_reason.value if frame.termination_reason else None}


class SimulationService:
    def __init__(self, artifact_root: str | Path, *, project_root: str | Path) -> None:
        self.artifact_root, self.project_root = Path(artifact_root), Path(project_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.resources: dict[str, dict[str, dict]] = {name: {} for name in ("vessels", "worlds", "scenarios")}
        self.policies: dict[str, PolicyBundle] = {}; self.runs: dict[str, dict] = {}

    def store(self, family: str, identity: str, document: dict) -> dict:
        if family not in self.resources or not identity or identity in self.resources[family]:
            raise PhysicalValidationError("Resource family/identity is invalid or duplicate")
        self.resources[family][identity] = json.loads(json.dumps(document))
        return self.resources[family][identity]

    def validate(self, config: dict, definitions: list[dict]) -> dict:
        resolved = resolve(config, build_registry(definitions))
        return resolved_artifact(resolved)

    def run(self, *, run_id: str, config: dict, definitions: list[dict], actions: list[dict],
            seed: int | None = None, episode_index: int = 0) -> dict:
        if run_id in self.runs or (self.artifact_root/f"run_{run_id}").exists():
            raise PhysicalValidationError("Run identity already exists")
        resolved = resolve(config, build_registry(definitions)); engine = build_engine(resolved)
        initial = engine.reset(seed=seed, episode_index=episode_index)
        recorder = RunRecorder(self.artifact_root, run_id=run_id, resolved=resolved, scenario=engine.scenario,
                               initial_frame=initial, project_root=self.project_root)
        frames = [serialize_frame(initial)]
        frame = initial
        try:
            for action_payload in actions:
                typed = self._actions(engine, action_payload)
                frame = engine.step(typed); recorder.record_frame(frame, actions=typed); frames.append(serialize_frame(frame))
                if frame.terminated: break
        except Exception:
            if recorder.thread.is_alive():
                try: recorder.close(final_frame=frame, external_stop=not frame.terminated)
                except Exception: pass
            raise
        root = recorder.close(final_frame=frame, external_stop=not engine.terminated)
        result = {"run_id": run_id, "config_hash": resolved.content_hash, "frames": frames,
                  "artifact_path": str(root)}
        self.runs[run_id] = result
        return result

    def _actions(self, engine, payload: Mapping[str, object]):
        if set(payload) != {name for name, status in engine.statuses.items() if status.rl_active}:
            raise PhysicalValidationError("Action payload must match active agents")
        result = {}
        for name, value in payload.items():
            mode = engine.config_vessels[name].controller.mode
            if mode == "direct_actuator":
                if not isinstance(value, dict): raise PhysicalValidationError("Direct action must be an object")
                result[name] = DirectAction(tuple((key, ThrustCommand(float(command)))
                                                  for key, command in sorted(value.items())))
            elif mode == "high_level":
                if not isinstance(value, dict) or set(value) != {"desired_speed_mps", "desired_heading_rad"}:
                    raise PhysicalValidationError("High-level action fields mismatch")
                result[name] = HighLevelCommand(float(value["desired_speed_mps"]), float(value["desired_heading_rad"]))
            else: raise PhysicalValidationError("Scripted agents do not accept policy actions")
        return result

    def upload_policy(self, policy_id: str, files: Mapping[str, str], observation_hash: str, action_hash: str) -> dict:
        if set(files) != set(REQUIRED_FILES) or not policy_id or policy_id in self.policies:
            raise PhysicalValidationError("Policy upload file set or identity is invalid")
        root = self.artifact_root/"policies"/policy_id; root.mkdir(parents=True, exist_ok=False)
        try:
            for name, encoded in files.items(): (root/name).write_bytes(b64decode(encoded, validate=True))
            bundle = PolicyBundle.load(root, expected_observation_hash=observation_hash, expected_action_hash=action_hash)
        except Exception:
            import shutil; shutil.rmtree(root, ignore_errors=True); raise
        self.policies[policy_id] = bundle
        return {"policy_id": policy_id, "version": bundle.manifest.version,
                "model_sha256": bundle.manifest.model_sha256}
