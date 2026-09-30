"""FastAPI surface over canonical BCOD-Sim application services."""

import json
from pathlib import Path
from typing import Any
import yaml

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from bcod_sim.core.errors import BCODSimError
from bcod_sim.web.service import SimulationService
from bcod_sim.web import config_service
from bcod_sim.web.qualification import qualify
from bcod_sim.web.runner import LocalRunner
from bcod_sim.web.components import sensor_types, actuator_types
from bcod_sim.web.vessel_assembly import normalize as normalize_vessel_assembly
from bcod_sim.config.resolver import _UniqueLoader, _no_duplicate_pairs


class StrictRequest(BaseModel): model_config = ConfigDict(extra="forbid")


class ResourceRequest(StrictRequest):
    id: str = Field(min_length=1)
    document: dict[str, Any]


class ValidationRequest(StrictRequest):
    config: dict[str, Any]
    definitions: list[dict[str, Any]]


class RunRequest(ValidationRequest):
    run_id: str = Field(min_length=1)
    actions: list[dict[str, Any]]
    seed: int | None = Field(default=None, ge=0)
    episode_index: int = Field(default=0, ge=0)


class PolicyUploadRequest(StrictRequest):
    policy_id: str = Field(min_length=1)
    files_base64: dict[str, str]
    observation_contract_hash: str
    action_contract_hash: str


class ConfigRequest(StrictRequest):
    document: dict[str, Any]


class DiffRequest(StrictRequest):
    left: dict[str, Any]
    right: dict[str, Any]


class QualificationRequest(StrictRequest):
    config: dict[str, Any]
    definitions: list[dict[str, Any]]


class JobRequest(QualificationRequest):
    run_id: str = Field(min_length=1)
    mode: str = "scripted_waypoint"
    seed: int | None = Field(default=None, ge=0)
    step_delay_s: float = Field(default=0, ge=0, le=.1)
    policy_id: str | None = None


class ImportRequest(StrictRequest):
    content: str = Field(max_length=2_000_000)
    format: str


class ExportRequest(QualificationRequest):
    format: str


class ConfigExportRequest(ConfigRequest):
    format: str


class VesselAssemblyExportRequest(StrictRequest):
    document: dict[str, Any]
    format: str


def create_app(*, artifact_root: str | Path, project_root: str | Path,
               frontend_dist: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="BCOD-Sim API", version="1.0")
    service = SimulationService(artifact_root, project_root=project_root)
    app.state.service = service
    runner = LocalRunner(artifact_root, project_root)
    app.state.runner = runner

    @app.exception_handler(BCODSimError)
    async def bcod_error(_: Request, exc: BCODSimError):
        return JSONResponse(status_code=422, content={"error": type(exc).__name__, "detail": str(exc)})

    @app.get("/v1/health")
    def health(): return {"status": "ok", "physics_location": "backend"}

    def config_routes(kind: str):
        def normalize(body: ConfigRequest):
            return config_service.serialize(config_service.parse(kind, body.document))
        def validate_config(body: ConfigRequest):
            return config_service.validate(kind, body.document)
        def migrate_config(body: ConfigRequest):
            return config_service.migrate(kind, body.document)
        def diff_config(body: DiffRequest):
            return {"changes": config_service.diff(kind, body.left, body.right)}
        def import_config(body: ImportRequest):
            if body.format not in {"json", "yaml"}:
                return JSONResponse(status_code=422, content={"detail": "Use json or yaml"})
            try:
                raw = (json.loads(body.content, object_pairs_hook=_no_duplicate_pairs,
                                  parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite: {x}"))) if body.format == "json"
                       else yaml.load(body.content, Loader=_UniqueLoader))
                document = raw if isinstance(raw, dict) and set(raw) == {"schema_version", "payload"} else {
                    "schema_version": 1, "payload": raw}
                return config_service.serialize(config_service.parse(kind, document))
            except Exception as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})
        def export_config(body: ConfigExportRequest):
            if body.format not in {"json", "yaml"}:
                return JSONResponse(status_code=422, content={"detail": "Use json or yaml"})
            document = config_service.serialize(config_service.parse(kind, body.document))
            content = (json.dumps(document, indent=2, sort_keys=True)+"\n" if body.format == "json"
                       else yaml.safe_dump(document, sort_keys=True))
            return {"content": content, "format": body.format,
                    "content_hash": config_service.validate(kind, document)["content_hash"]}
        return normalize, validate_config, migrate_config, diff_config, import_config, export_config

    for config_kind in ("vessel", "environment", "scenario", "task"):
        normalize, validate_config, migrate_config, diff_config, import_config, export_config = config_routes(config_kind)
        base = f"/v1/configs/{config_kind}"
        app.add_api_route(f"{base}/normalize", normalize, methods=["POST"], name=f"normalize_{config_kind}")
        app.add_api_route(f"{base}/validate", validate_config, methods=["POST"], name=f"validate_{config_kind}")
        app.add_api_route(f"{base}/migrate", migrate_config, methods=["POST"], name=f"migrate_{config_kind}")
        app.add_api_route(f"{base}/diff", diff_config, methods=["POST"], name=f"diff_{config_kind}")
        app.add_api_route(f"{base}/import", import_config, methods=["POST"], name=f"import_{config_kind}")
        app.add_api_route(f"{base}/export", export_config, methods=["POST"], name=f"export_{config_kind}")

    def resource_routes(family: str):
        def list_resources(): return service.resources[family]
        def create_resource(body: ResourceRequest): return service.store(family, body.id, body.document)
        return list_resources, create_resource

    for family in ("vessels", "worlds", "scenarios"):
        list_resources, create_resource = resource_routes(family)
        app.add_api_route(f"/v1/{family}", list_resources, methods=["GET"], name=f"list_{family}")
        app.add_api_route(f"/v1/{family}", create_resource, methods=["POST"], name=f"create_{family}")

    @app.post("/v1/validate")
    def validate(body: ValidationRequest): return service.validate(body.config, body.definitions)

    @app.post("/v1/qualify")
    def qualification(body: QualificationRequest):
        return qualify(body.config, body.definitions, artifact_root=service.artifact_root)

    @app.get("/v1/presets/navigation")
    def navigation_preset():
        return json.loads((Path(project_root)/"tests/fixtures/web/experiment_navigation.json").read_text())

    @app.post("/v1/experiments/parse")
    def parse_experiment(body: ImportRequest):
        if body.format not in {"json", "yaml"}:
            return JSONResponse(status_code=422, content={"detail": "Use json or yaml"})
        try:
            raw = (json.loads(body.content, object_pairs_hook=_no_duplicate_pairs,
                             parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite: {x}"))) if body.format == "json"
                   else yaml.load(body.content, Loader=_UniqueLoader))
            if not isinstance(raw, dict) or set(raw) != {"config", "definitions"} or not isinstance(raw["definitions"], list):
                raise ValueError("Complete experiment requires config and definitions")
            from bcod_sim.web.service import build_registry
            from bcod_sim.config.resolver import resolve
            resolved = resolve(raw["config"], build_registry(raw["definitions"]))
            return {"config": resolved.config.model_dump(mode="json"), "definitions": raw["definitions"]}
        except Exception as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/v1/experiments/serialize")
    def serialize_experiment(body: ExportRequest):
        if body.format not in {"json", "yaml"}:
            return JSONResponse(status_code=422, content={"detail": "Use json or yaml"})
        from bcod_sim.web.service import build_registry
        from bcod_sim.config.resolver import resolve
        resolved = resolve(body.config, build_registry(body.definitions))
        output = {"config": resolved.config.model_dump(mode="json"), "definitions": body.definitions}
        content = (json.dumps(output, indent=2, sort_keys=True)+"\n" if body.format == "json"
                   else yaml.safe_dump(output, sort_keys=True))
        return {"content": content, "format": body.format, "experiment_hash": resolved.content_hash}

    @app.post("/v1/vessel-assembly/parse")
    def parse_vessel_assembly(body: ImportRequest):
        if body.format not in {"json", "yaml"}:
            return JSONResponse(status_code=422, content={"detail": "Use json or yaml"})
        try:
            raw = (json.loads(body.content, object_pairs_hook=_no_duplicate_pairs) if body.format == "json"
                   else yaml.load(body.content, Loader=_UniqueLoader))
            return normalize_vessel_assembly(raw)
        except Exception as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/v1/vessel-assembly/serialize")
    def serialize_vessel_assembly(body: VesselAssemblyExportRequest):
        if body.format not in {"json", "yaml"}:
            return JSONResponse(status_code=422, content={"detail": "Use json or yaml"})
        try:
            document = normalize_vessel_assembly(body.document)
            content = (json.dumps(document, indent=2, sort_keys=True)+"\n" if body.format == "json"
                       else yaml.safe_dump(document, sort_keys=True))
            return {"content": content, "format": body.format}
        except Exception as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/v1/component-types/sensors")
    def sensors(): return sensor_types()

    @app.get("/v1/component-types/actuators")
    def actuators(): return actuator_types()

    @app.post("/v1/jobs")
    def submit_job(body: JobRequest):
        return runner.submit(body.run_id, body.config, body.definitions, mode=body.mode,
            seed=body.seed, step_delay_s=body.step_delay_s, policy_id=body.policy_id)

    @app.get("/v1/runners")
    def runners(): return [runner.describe()]

    @app.get("/v1/jobs")
    def list_jobs(): return runner.list()

    @app.get("/v1/jobs/{run_id}")
    def job_status(run_id: str):
        try: return runner.status(run_id)
        except FileNotFoundError: return JSONResponse(status_code=404, content={"detail": "Job not found"})

    @app.get("/v1/jobs/{run_id}/experiment")
    def job_experiment(run_id: str):
        try: return runner.experiment(run_id)
        except FileNotFoundError: return JSONResponse(status_code=404, content={"detail": "Job not found"})

    @app.get("/v1/jobs/{run_id}/artifacts")
    def job_artifacts(run_id: str):
        try: return runner.artifacts(run_id)
        except FileNotFoundError: return JSONResponse(status_code=404, content={"detail": "Job not found"})

    @app.get("/v1/jobs/{run_id}/trajectory")
    def job_trajectory(run_id: str, max_points: int = 1000):
        try: return runner.trajectory(run_id, max_points=max_points)
        except FileNotFoundError: return JSONResponse(status_code=404, content={"detail": "Job not found"})

    @app.post("/v1/jobs/{run_id}/stop")
    def stop_job(run_id: str):
        try: return runner.stop(run_id)
        except FileNotFoundError: return JSONResponse(status_code=404, content={"detail": "Job not found"})

    @app.post("/v1/jobs/{run_id}/kill")
    def kill_job(run_id: str): return runner.kill(run_id)

    def stream_route(name: str):
        def page(run_id: str, offset: int = 0, limit: int = 200):
            return runner.page(run_id, f"{name}.jsonl", offset=offset, limit=limit)
        return page

    for stream_name in ("frames", "metrics", "logs"):
        page = stream_route(stream_name)
        app.add_api_route(f"/v1/jobs/{{run_id}}/{stream_name}", page, methods=["GET"],
                          name=f"job_{stream_name}")

    @app.post("/v1/runs")
    def run(body: RunRequest):
        return service.run(run_id=body.run_id, config=body.config, definitions=body.definitions,
                           actions=body.actions, seed=body.seed, episode_index=body.episode_index)

    @app.get("/v1/runs")
    def runs(): return {key: {"run_id": value["run_id"], "config_hash": value["config_hash"]}
                        for key, value in service.runs.items()}

    @app.get("/v1/runs/{run_id}")
    def run_detail(run_id: str):
        if run_id not in service.runs: return JSONResponse(status_code=404, content={"detail": "Run not found"})
        return service.runs[run_id]

    @app.get("/v1/runs/{run_id}/stream")
    def run_stream(run_id: str):
        if run_id not in service.runs: return JSONResponse(status_code=404, content={"detail": "Run not found"})
        def events():
            for frame in service.runs[run_id]["frames"]:
                yield f"event: frame\ndata: {json.dumps(frame, separators=(',', ':'))}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/policies")
    def upload_policy(body: PolicyUploadRequest):
        return service.upload_policy(body.policy_id, body.files_base64, body.observation_contract_hash,
                                     body.action_contract_hash)

    @app.get("/v1/policies")
    def policies():
        root = service.artifact_root/"policies"
        return {path.name: {"policy_id": manifest["policy_id"], "version": manifest["version"]}
                for path in root.iterdir() if path.is_dir() and (path/"manifest.json").is_file()
                for manifest in [json.loads((path/"manifest.json").read_text())]} if root.is_dir() else {}

    @app.get("/v1/artifacts/{run_id}/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str):
        if run_id in service.runs:
            location = service.runs[run_id]["artifact_path"]
        else:
            try: location = runner.status(run_id).get("artifact_path")
            except FileNotFoundError: location = None
        if not location: return JSONResponse(status_code=404, content={"detail": "Run not found"})
        root = Path(location).resolve(); target = (root/artifact_path).resolve()
        if root not in target.parents or not target.is_file():
            return JSONResponse(status_code=404, content={"detail": "Artifact not found"})
        return FileResponse(target)

    if frontend_dist is not None and Path(frontend_dist).is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="web-client")
    return app
