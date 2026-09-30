"""Local process adapter with a fixed, validated simulator job protocol."""

from datetime import datetime, timezone
import json
import os
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys

from bcod_sim.config.resolver import resolve
from bcod_sim.core.errors import PhysicalValidationError
from bcod_sim.web.service import build_registry


RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class LocalRunner:
    def __init__(self, artifact_root: str | Path, project_root: str | Path):
        self.root = Path(artifact_root).resolve()/"jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root).resolve()
        self.processes: dict[str, subprocess.Popen] = {}

    def describe(self) -> dict:
        return {"runner_id": "local", "adapter": "local-process-v1", "state": "ONLINE",
                "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "resources": {"cpu_count": os.cpu_count(), "gpu": "unreported",
                              "memory_bytes": self._memory_bytes()}}

    @staticmethod
    def _memory_bytes() -> int | None:
        try: return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, AttributeError): return None

    def _job(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise PhysicalValidationError("Invalid run ID")
        return self.root/run_id

    def submit(self, run_id: str, config: dict, definitions: list[dict], *,
               mode: str = "scripted_waypoint", seed: int | None = None,
               step_delay_s: float = 0.0, policy_id: str | None = None) -> dict:
        if mode not in {"scripted_waypoint", "policy_evaluation"}:
            raise PhysicalValidationError("Unsupported local runner mode")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise PhysicalValidationError("Seed must be nonnegative")
        if not 0 <= step_delay_s <= .1:
            raise PhysicalValidationError("Step delay must be between 0 and 0.1 seconds")
        resolved = resolve(config, build_registry(definitions))
        task = next((d for d in resolved.definitions if d.kind == "task"), None)
        if len(resolved.config.vessels) != 1 or task is None or task.payload.get("kind") != "waypoint":
            raise PhysicalValidationError("Local MVP runner supports one vessel and waypoint task")
        if resolved.config.simulation.max_master_steps > 10000:
            raise PhysicalValidationError("Local MVP run is bounded to 10000 master steps")
        if mode == "scripted_waypoint" and resolved.config.vessels[0].controller.mode != "high_level":
            raise PhysicalValidationError("Scripted waypoint baseline needs high_level control")
        if mode == "policy_evaluation" and not policy_id:
            raise PhysicalValidationError("Policy evaluation requires policy_id")
        job = self._job(run_id)
        try:
            job.mkdir()
        except FileExistsError as exc:
            raise PhysicalValidationError("Run ID already exists") from exc
        request = {"schema_version": 1, "run_id": run_id, "config": config,
                   "definitions": definitions, "mode": mode, "seed": seed,
                   "step_delay_s": step_delay_s, "policy_id": policy_id,
                   "experiment_hash": resolved.content_hash}
        write_json(job/"request.json", request)
        write_json(job/"status.json", {"run_id": run_id, "state": "QUEUED", "step": 0,
            "experiment_hash": resolved.content_hash, "submitted_at": datetime.now(timezone.utc).isoformat(),
            "runner": "local-process-v1", "compute_target": "local", "seed": seed if seed is not None else config["experiment"]["seed"]})
        output = (job/"worker.log").open("ab", buffering=0)
        try:
            process = subprocess.Popen([sys.executable, "-m", "bcod_sim.web.worker",
                str(job/"request.json"), str(self.project_root), str(self.root.parent)],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                cwd=self.project_root, start_new_session=True)
        except Exception:
            write_json(job/"status.json", {"run_id": run_id, "state": "FAILED", "step": 0,
                "error": "Runner process could not start", "failure_category": "COMPUTE_ERROR"})
            raise
        finally:
            output.close()
        self.processes[run_id] = process
        return self.status(run_id)

    def status(self, run_id: str) -> dict:
        job = self._job(run_id)
        path = job/"status.json"
        if not path.is_file():
            raise FileNotFoundError(run_id)
        result = json.loads(path.read_text())
        process = self.processes.get(run_id)
        if process is not None and process.poll() is not None and result["state"] in {"QUEUED", "RUNNING"}:
            result.update(state="FAILED", error="Runner exited without final status",
                          exit_code=process.returncode, failure_category="COMPUTE_ERROR")
            write_json(path, result)
        return result

    def list(self) -> list[dict]:
        return [self.status(path.name) for path in sorted(self.root.iterdir())
                if path.is_dir() and RUN_ID.fullmatch(path.name) and (path/"status.json").is_file()]

    def experiment(self, run_id: str) -> dict:
        path = self._job(run_id)/"request.json"
        if not path.is_file():
            raise FileNotFoundError(run_id)
        request = json.loads(path.read_text())
        return {"config": request["config"], "definitions": request["definitions"]}

    def artifacts(self, run_id: str) -> dict:
        status = self.status(run_id)
        root = Path(status.get("artifact_path", ""))
        if status["state"] not in {"COMPLETED", "STOPPED"} or not root.is_dir():
            return {"artifacts": [], "checkpoints": []}
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return {"artifacts": files, "checkpoints": []}

    def stop(self, run_id: str) -> dict:
        job = self._job(run_id)
        if not (job/"status.json").is_file():
            raise FileNotFoundError(run_id)
        if self.status(run_id)["state"] not in {"QUEUED", "RUNNING"}:
            raise PhysicalValidationError("Only an active job can be stopped")
        (job/"stop.requested").write_text("stop\n")
        return self.status(run_id)

    def kill(self, run_id: str) -> dict:
        process = self.processes.get(run_id)
        if process is None or process.poll() is not None:
            raise PhysicalValidationError("Live local process is unavailable for hard kill")
        process.kill(); process.wait(timeout=5)
        status = self.status(run_id)
        status.update(state="KILLED", ended_at=datetime.now(timezone.utc).isoformat())
        write_json(self._job(run_id)/"status.json", status)
        return status

    def page(self, run_id: str, name: str, *, offset: int = 0, limit: int = 200) -> dict:
        if name not in {"frames.jsonl", "metrics.jsonl", "logs.jsonl"}:
            raise PhysicalValidationError("Unsupported stream")
        if offset < 0 or not 1 <= limit <= 1000:
            raise PhysicalValidationError("Invalid stream page")
        path = self._job(run_id)/name
        if not path.is_file():
            return {"items": [], "next_offset": offset}
        rows = []
        with path.open() as stream:
            for index, line in enumerate(stream):
                if index < offset: continue
                if len(rows) == limit: break
                rows.append(json.loads(line))
        return {"items": rows, "next_offset": offset + len(rows)}

    def trajectory(self, run_id: str, *, max_points: int = 1000) -> dict:
        if not 2 <= max_points <= 1000:
            raise PhysicalValidationError("Invalid trajectory point limit")
        status = self.status(run_id)
        path = self._job(run_id)/"frames.jsonl"
        if not path.is_file(): return {"frames": []}
        stride = max(1, math.ceil((status.get("step", 0)+1)/max_points))
        frames = []
        with path.open() as stream:
            for index, line in enumerate(stream):
                if index % stride: continue
                row = json.loads(line)
                frames.append({"master_step": row["master_step"], "states": {
                    agent: {"position_display_m": value["position_display_m"]}
                    for agent, value in row["states"].items()}})
        return {"frames": frames, "stride": stride}
