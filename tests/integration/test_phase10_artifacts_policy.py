from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import threading

import numpy as np
import onnx
from onnx import TensorProto, helper
import pyarrow.parquet as pq
import pytest
import torch

from bcod_sim.actuators.base import ActuatorConfig, Bounds
from bcod_sim.actuators.thruster import FixedThruster, ThrustCommand
from bcod_sim.collision.shapes import Sphere
from bcod_sim.config.hashing import content_hash
from bcod_sim.config.registry import Registry
from bcod_sim.config.resolver import resolve
from bcod_sim.core.engine import EpisodeEngine, EpisodeVessel
from bcod_sim.core.environment_loads import ExplicitZeroLoads
from bcod_sim.core.errors import ConfigSchemaError, LoggingBackpressureError, PolicyContractMismatchError, UnknownReferenceError
from bcod_sim.core.lifecycle import DirectAction, TerminationReason
from bcod_sim.dynamics.damping import Damping
from bcod_sim.dynamics.matrices import MassProperties
from bcod_sim.dynamics.operating_envelope import OperatingEnvelope
from bcod_sim.dynamics.plant6 import Plant6
from bcod_sim.dynamics.restoring import Hydrostatics
from bcod_sim.logging.artifacts import load_resolved_artifact, load_scenario_artifact
from bcod_sim.logging.metrics import MetricSet, MetricsRegistry
from bcod_sim.logging.recorder import RunRecorder
from bcod_sim.policy.bundle import PolicyBundle
from bcod_sim.policy.runtime import ONNXPolicyRuntime


def make_engine(*, metrics=("reward", "success", "collision_count", "distance_traveled", "time_to_completion", "control_effort")):
    registry = Registry(); registry.register("vessel", "vessel", "1", {"mass": 10}, "test")
    registry.register("task", "waypoint", "1", {"kind": "waypoint", "agent_id": "agent",
        "target_ned_m": [100, 0, 0], "radius_m": 0.1}, "test")
    resolved = resolve({"schema_version": 1, "experiment": {"id": "log", "seed": 12},
        "simulation": {"dynamics_mode": "full6", "master_dt_s": 0.1, "dynamics_substeps": 1,
                       "policy_every_n_master_steps": 1, "max_master_steps": 2},
        "world": {"source": {"kind": "parametric"}, "environment": {
            "current": {"kind": "uniform", "ned_mps": [0, 0, 0]},
            "wind": {"kind": "uniform", "ned_mps": [0, 0, 0]}, "waves": {"kind": "calm"},
            "visibility_m": 1000}, "bathymetry": {"kind": "flat", "bottom_ned_z_m": 20,
                                                   "vertical_datum": "MSL"}},
        "vessels": [{"instance_id": "agent", "definition": "vessel@1",
            "controller": {"mode": "direct_actuator"}, "spawn": {"ned_m": [0, 0, 0], "rpy_rad": [0, 0, 0]}}],
        "task": {"type": "waypoint@1", "reward": {"individual_weight": 1, "team_weight": 0},
                 "disabled_agent_behavior": "deactivate_keep_physical"},
        "logging": {"metrics": list(metrics), "states": True, "sensor_payloads": False,
                    "queue_capacity": 8, "backpressure": "block"}}, registry)
    v = lambda x: torch.tensor(x, dtype=torch.float64)
    mass = MassProperties(10, v((0, 0, 0)), torch.diag(v((4, 5, 6))), torch.zeros((6, 6), dtype=torch.float64))
    plant = Plant6(mass, Damping(v((0,)*6), v((0,)*6)), Hydrostatics(10*9.80665, v((0, 0, 0))),
                   OperatingEnvelope(v((1e4,)*6), max_substep_s=0.2))
    prop = FixedThruster(ActuatorConfig("prop", 0, 1, (0, 0, 0), (1, 0, 0, 0), Bounds(-100, 100)))
    return EpisodeEngine(resolved, (EpisodeVessel("agent", 1, plant, Sphere(0.2), (prop,), (),
                                                  ExplicitZeroLoads()),))


def zero_action(): return DirectAction((("prop", ThrustCommand(0)),))


def write_policy(root: Path):
    root.mkdir()
    x = helper.make_tensor_value_info("observation", TensorProto.FLOAT, [None, 2])
    y = helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, 2])
    graph = helper.make_graph([helper.make_node("Identity", ["observation"], ["action"])], "identity", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]); model.ir_version = 10
    onnx.save(model, root/"model.onnx")
    documents = {
        "observation_contract.json": {"model_inputs": ["observation"],
            "fields": [{"name": "xy", "shape": [2], "dtype": "float32"}]},
        "action_contract.json": {"model_outputs": ["action"],
            "fields": [{"name": "thrust", "shape": [2], "dtype": "float32"}]},
        "preprocessing.json": {"operation": "identity"},
        "normalization.json": {"mean": [0, 0], "scale": [1, 1], "source": "training"},
        "training_provenance.json": {"simulator_commit": "abc123", "training_config_hash": "1"*64,
            "trainer_commit": "def456", "trainer_version": "1.0", "seeds": [3]},
    }
    for name, document in documents.items(): (root/name).write_text(json.dumps(document))
    manifest = {"policy_id": "demo", "version": "1", "model_format": "onnx",
        "model_sha256": hashlib.sha256((root/"model.onnx").read_bytes()).hexdigest(),
        "observation_contract_hash": content_hash(documents["observation_contract.json"]),
        "action_contract_hash": content_hash(documents["action_contract.json"]),
        "simulator_commit": "abc123", "training_config_hash": "1"*64,
        "trainer_commit": "def456", "trainer_version": "1.0", "seeds": [3],
        "preprocessing_sha256": content_hash(documents["preprocessing.json"]),
        "normalization_provenance": content_hash(documents["normalization.json"]),
        "training_provenance_sha256": content_hash(documents["training_provenance.json"])}
    (root/"manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_policy_bundle_contract_validation_and_onnx_demo(tmp_path):
    manifest = write_policy(tmp_path/"policy")
    bundle = PolicyBundle.load(tmp_path/"policy", expected_observation_hash=manifest["observation_contract_hash"],
                               expected_action_hash=manifest["action_contract_hash"])
    runtime = ONNXPolicyRuntime(bundle)
    result = runtime.infer({"observation": np.array([[1, 2]], dtype=np.float32)})
    assert result["action"].tolist() == [[1, 2]]
    with pytest.raises(PolicyContractMismatchError, match="Observation"):
        PolicyBundle.load(tmp_path/"policy", expected_observation_hash="0"*64,
                          expected_action_hash=manifest["action_contract_hash"])
    (tmp_path/"policy"/"model.onnx").write_bytes(b"tampered")
    with pytest.raises(PolicyContractMismatchError, match="checksum"):
        PolicyBundle.load(tmp_path/"policy", expected_observation_hash=manifest["observation_contract_hash"],
                          expected_action_hash=manifest["action_contract_hash"])


def test_run_artifact_is_complete_integrity_checked_and_reconstructable(tmp_path):
    engine = make_engine(); initial = engine.reset(seed=99, episode_index=4)
    recorder = RunRecorder(tmp_path, run_id="fixture", resolved=engine.resolved, scenario=engine.scenario,
        initial_frame=initial, project_root=Path(__file__).parents[2])
    frame1 = engine.step({"agent": zero_action()}); recorder.record_frame(frame1, actions={"agent": zero_action()})
    frame2 = engine.step({"agent": zero_action()}); recorder.record_frame(frame2, actions={"agent": zero_action()})
    root = recorder.close(final_frame=frame2)
    assert frame2.termination_reason == TerminationReason.TIME_LIMIT
    required = {"manifest.json", "config.resolved.yaml", "provenance", "metrics.parquet", "events.jsonl",
                "states.parquet", "sensors", "scenario.resolved.yaml"}
    assert {path.name for path in root.iterdir()} == required
    restored = load_resolved_artifact(root/"config.resolved.yaml")
    scenario = load_scenario_artifact(root/"scenario.resolved.yaml")
    assert restored.content_hash == engine.resolved.content_hash
    assert restored.definitions[0].payload == engine.resolved.definitions[0].payload
    assert scenario == engine.scenario
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["termination_reason"] == "time_limit" and manifest["backend"] == "torch"
    software = json.loads((root/"provenance"/"software.json").read_text())
    assert manifest["dependency_lock_hash"] == software["dependency_lock_sha256"]
    assert manifest["git_commit"] and len(software["source_tree_sha256"]) == 64
    assert pq.read_table(root/"metrics.parquet").num_rows > 0
    assert pq.read_table(root/"states.parquet").num_rows == 2


def test_artifact_tampering_fails_integrity_validation(tmp_path):
    engine = make_engine(metrics=()); initial = engine.reset()
    recorder = RunRecorder(tmp_path, run_id="tamper", resolved=engine.resolved, scenario=engine.scenario,
                           initial_frame=initial, project_root=Path(__file__).parents[2])
    engine.step({"agent": zero_action()}); final = engine.step({"agent": zero_action()})
    root = recorder.close(final_frame=final)
    document = (root/"config.resolved.yaml").read_text().replace("mass: 10", "mass: 11")
    (root/"config.resolved.yaml").write_text(document)
    with pytest.raises(ConfigSchemaError, match="integrity"):
        load_resolved_artifact(root/"config.resolved.yaml")


def test_metrics_registry_extension_unknown_identity_and_components():
    class Custom:
        def reset(self, frame): pass
        def update(self, frame, actions=None):
            return ({"step": frame.master_step, "metric": "custom", "scope": "team", "value": 2.0},)
    registry = MetricsRegistry(); registry.register("custom", Custom)
    engine = make_engine(metrics=()); initial = engine.reset()
    metrics = MetricSet(registry.build(("custom", "reward"))); metrics.reset(initial)
    frame = engine.step({"agent": zero_action()})
    rows = metrics.update(frame)
    assert any(row["metric"] == "custom" for row in rows)
    assert any(row["metric"] == "reward.waypoint_progress" for row in rows)
    with pytest.raises(UnknownReferenceError): registry.build(("missing",))


def test_backpressure_drop_is_reported_and_terminate_is_explicit():
    recorder = object.__new__(RunRecorder); recorder.worker_error = None; recorder.queue = queue.Queue(1)
    recorder.queue.put(("event", {})); recorder.lock = threading.Lock(); recorder.dropped = 0
    recorder.backpressure = "drop_noncritical_with_event"
    recorder.submit("metric", {"value": 1})
    assert recorder.dropped == 1
    recorder.backpressure = "terminate"
    with pytest.raises(LoggingBackpressureError): recorder.submit("metric", {"value": 1})
