"""Default ASGI entry point: uvicorn bcod_sim.web.main:app."""

import os
from pathlib import Path

from bcod_sim.web.api import create_app


PROJECT_ROOT = Path(os.environ.get("BCOD_SIM_PROJECT_ROOT", Path.cwd())).resolve()
ARTIFACT_ROOT = Path(os.environ.get("BCOD_SIM_ARTIFACT_ROOT", PROJECT_ROOT/".bcod_runs")).resolve()
FRONTEND_DIST = Path(os.environ.get("BCOD_SIM_FRONTEND_DIST", PROJECT_ROOT/"web_client"/"dist")).resolve()

app = create_app(artifact_root=ARTIFACT_ROOT, project_root=PROJECT_ROOT, frontend_dist=FRONTEND_DIST)
