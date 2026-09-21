"""Bounded asynchronous run recorder with explicit backpressure behavior."""

from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import re
import threading
from typing import Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from bcod_sim.config.resolver import ResolvedExperiment
from bcod_sim.core.engine import EpisodeFrame
from bcod_sim.core.errors import LoggingBackpressureError, PhysicalValidationError
from bcod_sim.logging.manifest import RunManifest, manifest_skeleton
from bcod_sim.logging.artifacts import resolved_artifact, scenario_artifact
from bcod_sim.logging.metrics import MetricSet, MetricsRegistry
from bcod_sim.logging.provenance import runtime_provenance, software_provenance
from bcod_sim.policy.bundle import PolicyBundle
from bcod_sim.scenario.generator import ResolvedScenario


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=str)+"\n")
    temporary.replace(path)


class RunRecorder:
    def __init__(self, parent: str | Path, *, run_id: str, resolved: ResolvedExperiment,
                 scenario: ResolvedScenario, initial_frame: EpisodeFrame, project_root: str | Path,
                 policy: PolicyBundle | None = None,
                 observation_contract_hash: str | None = None,
                 action_contract_hash: str | None = None,
                 external_data: tuple[Mapping[str, object], ...] = ()) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise PhysicalValidationError("Run ID is invalid")
        queue_capacity = resolved.config.logging.queue_capacity
        backpressure = resolved.config.logging.backpressure
        if resolved.config.world.source.kind == "real_world" and not external_data:
            raise PhysicalValidationError("Real-world run recording requires external-data provenance")
        if policy is not None and (observation_contract_hash != policy.manifest.observation_contract_hash or
                                   action_contract_hash != policy.manifest.action_contract_hash):
            raise PhysicalValidationError("Run contract hashes must match the loaded policy bundle")
        self.root = Path(parent)/f"run_{run_id}"
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root/"provenance").mkdir(); (self.root/"sensors").mkdir()
        self.queue: queue.Queue = queue.Queue(queue_capacity)
        self.backpressure, self.dropped = backpressure, 0
        self.lock = threading.Lock(); self.worker_error: BaseException | None = None
        self.metric_rows: list[dict] = []; self.state_rows: list[dict] = []
        self.log_states = resolved.config.logging.states
        self.sensor_payloads = resolved.config.logging.sensor_payloads
        self.metrics = MetricSet(MetricsRegistry().build(resolved.config.logging.metrics))
        self.metrics.reset(initial_frame)
        (self.root/"config.resolved.yaml").write_text(yaml.safe_dump(resolved_artifact(resolved), sort_keys=True))
        (self.root/"scenario.resolved.yaml").write_text(yaml.safe_dump(scenario_artifact(scenario), sort_keys=True))
        software = software_provenance(project_root)
        runtime = runtime_provenance(initial_frame.states[next(iter(initial_frame.states))].nu_body.dtype,
                                     initial_frame.states[next(iter(initial_frame.states))].nu_body.device)
        _write_json(self.root/"provenance"/"software.json", {**software, **runtime})
        _write_json(self.root/"provenance"/"assets.json", {key: value for key, value in
            manifest_skeleton(resolved, run_id=run_id).asset_hashes.items()})
        _write_json(self.root/"provenance"/"external_data.json", list(external_data))
        base = manifest_skeleton(resolved, run_id=run_id)
        self.manifest = base.model_copy(update={
            "git_commit": software["git_commit"], "dirty_patch_hash": software["dirty_patch_sha256"],
            "dependency_lock_hash": software["dependency_lock_sha256"], "backend": runtime["backend"],
            "device": runtime["device"], "dtype": runtime["dtype"],
            "external_data": tuple({str(k): str(v) for k, v in row.items()} for row in external_data),
            "seed_streams": {"scenario": scenario.episode_seed},
            "policy_provenance": ({"policy_id": policy.manifest.policy_id,
                                   "version": policy.manifest.version,
                                   "model_sha256": policy.manifest.model_sha256} if policy else None),
            "trainer_provenance": ({str(k): str(v) for k, v in policy.training_provenance.items()}
                                   if policy else None),
            "observation_contract_hash": observation_contract_hash,
            "action_contract_hash": action_contract_hash,
        })
        _write_json(self.root/"manifest.json", self.manifest.model_dump(mode="json"))
        self.thread = threading.Thread(target=self._worker, name=f"bcod-log-{run_id}", daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        try:
            with (self.root/"events.jsonl").open("w") as events, (self.root/"sensors"/"packets.jsonl").open("w") as sensors:
                while True:
                    item = self.queue.get()
                    try:
                        if item is None: break
                        kind, payload = item
                        if kind == "metric": self.metric_rows.append(payload)
                        elif kind == "state": self.state_rows.append(payload)
                        elif kind == "sensor": sensors.write(json.dumps(payload, sort_keys=True, default=str)+"\n")
                        else: events.write(json.dumps(payload, sort_keys=True, default=str)+"\n")
                    finally:
                        self.queue.task_done()
                with self.lock: dropped = self.dropped
                if dropped:
                    events.write(json.dumps({"event": "logging_backpressure_drop", "dropped_noncritical": dropped})+"\n")
        except BaseException as exc:
            self.worker_error = exc

    def submit(self, kind: str, payload: Mapping[str, object], *, critical: bool = False) -> None:
        if self.worker_error: raise LoggingBackpressureError("Recorder worker failed") from self.worker_error
        item = (kind, dict(payload))
        if self.backpressure == "block" or critical:
            self.queue.put(item); return
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            if self.backpressure == "terminate":
                raise LoggingBackpressureError("Recorder queue capacity exceeded")
            with self.lock: self.dropped += 1

    def record_frame(self, frame: EpisodeFrame, *, actions: Mapping[str, object] | None = None) -> None:
        for row in self.metrics.update(frame, actions):
            value = row["value"]
            self.submit("metric", {**row, "value": float(value)})
        if self.log_states:
            for name, state in sorted(frame.states.items()):
                self.submit("state", {"step": frame.master_step, "sim_time_s": frame.sim_time_s,
                    "agent": name, "position_ned": state.position_ned.tolist(),
                    "q_body_to_ned": state.q_body_to_ned.tolist(), "nu_body": state.nu_body.tolist()})
        for event in frame.contact_events:
            self.submit("event", {"event": "collision_contact", "step": frame.master_step,
                "contact_id": event.contact_id, "normal_impulse_ns": event.normal_impulse_ns,
                "friction_impulse_ns": event.friction_impulse_ns}, critical=True)
        if self.sensor_payloads:
            for packet in frame.delivered_packets:
                self.submit("sensor", {"step": frame.master_step, "sensor_id": packet.sensor_id,
                    "owner_vessel_id": packet.owner_vessel_id, "values": packet.values})

    def close(self, *, final_frame: EpisodeFrame, external_stop: bool = False) -> Path:
        if not final_frame.terminated and not external_stop:
            raise PhysicalValidationError("Closing an active run requires explicit external_stop")
        self.queue.put(None); self.thread.join()
        if self.worker_error: raise LoggingBackpressureError("Recorder worker failed") from self.worker_error
        metric_schema = pa.schema([("step", pa.int64()), ("metric", pa.string()),
                                   ("scope", pa.string()), ("value", pa.float64())])
        state_schema = pa.schema([("step", pa.int64()), ("sim_time_s", pa.float64()), ("agent", pa.string()),
            ("position_ned", pa.list_(pa.float64(), 3)), ("q_body_to_ned", pa.list_(pa.float64(), 4)),
            ("nu_body", pa.list_(pa.float64(), 6))])
        pq.write_table(pa.Table.from_pylist(self.metric_rows, schema=metric_schema), self.root/"metrics.parquet")
        if self.log_states:
            pq.write_table(pa.Table.from_pylist(self.state_rows, schema=state_schema), self.root/"states.parquet")
        reason = (final_frame.termination_reason.value if final_frame.termination_reason else
                  "external_stop" if external_stop else None)
        self.manifest = self.manifest.model_copy(update={"ended_utc": datetime.now(timezone.utc),
            "termination_reason": reason})
        _write_json(self.root/"manifest.json", self.manifest.model_dump(mode="json"))
        return self.root
