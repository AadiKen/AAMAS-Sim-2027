"""FastAPI surface over canonical BCOD-Sim application services."""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from bcod_sim.core.errors import BCODSimError
from bcod_sim.web.service import SimulationService


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


def create_app(*, artifact_root: str | Path, project_root: str | Path,
               frontend_dist: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="BCOD-Sim API", version="1.0")
    service = SimulationService(artifact_root, project_root=project_root)
    app.state.service = service

    @app.exception_handler(BCODSimError)
    async def bcod_error(_: Request, exc: BCODSimError):
        return JSONResponse(status_code=422, content={"error": type(exc).__name__, "detail": str(exc)})

    @app.get("/v1/health")
    def health(): return {"status": "ok", "physics_location": "backend"}

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
    def policies(): return {key: {"policy_id": value.manifest.policy_id, "version": value.manifest.version}
                            for key, value in service.policies.items()}

    @app.get("/v1/artifacts/{run_id}/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str):
        if run_id not in service.runs: return JSONResponse(status_code=404, content={"detail": "Run not found"})
        root = Path(service.runs[run_id]["artifact_path"]).resolve(); target = (root/artifact_path).resolve()
        if root not in target.parents or not target.is_file():
            return JSONResponse(status_code=404, content={"detail": "Artifact not found"})
        return FileResponse(target)

    if frontend_dist is not None and Path(frontend_dist).is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="web-client")
    return app
