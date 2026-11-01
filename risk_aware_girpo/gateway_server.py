from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from risk_aware_girpo.gateway import GatewayService, TaskCatalog, load_catalog_rows
from risk_aware_girpo.openhands_runner import ToolShieldRunner


class ResetRequest(BaseModel):
    task_id: str
    seed: int


class StepRequest(BaseModel):
    session_id: str
    action: str


class CloseRequest(BaseModel):
    session_id: str


def create_app() -> FastAPI:
    tasks_root = Path(os.environ["TASKS_ROOT"])
    manifests = [Path(value) for value in os.environ["TASK_MANIFESTS"].split(os.pathsep) if value]
    work_root = Path(os.environ.get("GATEWAY_WORK_ROOT", "work/gateway_sessions"))
    service = GatewayService(
        TaskCatalog(tasks_root, load_catalog_rows(manifests)), work_root, ToolShieldRunner
    )
    app = FastAPI(title="MT-AgentRisk OpenHands Gateway")

    @app.post("/reset")
    def reset(request: ResetRequest):
        try:
            return service.reset(request.task_id, request.seed)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/step")
    def step(request: StepRequest):
        try:
            return service.step(request.session_id, request.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown session: {request.session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/close")
    def close(request: CloseRequest):
        try:
            service.close(request.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown session: {request.session_id}") from exc
        return {"closed": True}

    return app
