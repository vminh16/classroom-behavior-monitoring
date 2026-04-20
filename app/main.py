from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .mediamtx import MediaMTXManager
from .runtime import InferenceManager, PROJECT_ROOT
from .schemas import StartSessionRequest, StopSessionResponse


APP_ROOT = PROJECT_ROOT / "app"
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    media_server = MediaMTXManager(APP_ROOT)
    manager = InferenceManager(media_server=media_server)
    app.state.manager = manager
    yield
    manager.shutdown()


app = FastAPI(
    title="Classroom Monitoring UI",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")


def _manager(request: Request) -> InferenceManager:
    return request.app.state.manager


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    manager = _manager(request)
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "options": manager.get_options(),
            "state": manager.snapshot()["state"],
        },
    )


@app.get("/api/options")
def options(request: Request):
    return JSONResponse(_manager(request).get_options())


@app.get("/api/status")
def status(request: Request):
    return JSONResponse(_manager(request).snapshot())


@app.post("/api/session/start")
def start_session(request: Request, payload: StartSessionRequest):
    manager = _manager(request)
    try:
        snapshot = manager.start(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(snapshot)


@app.post("/api/session/stop", response_model=StopSessionResponse)
def stop_session(request: Request):
    snapshot = _manager(request).stop()
    return StopSessionResponse(ok=True, state=snapshot["state"]["session_state"])


@app.get("/api/media/status")
def media_status(request: Request):
    return JSONResponse(_manager(request).snapshot()["media"])


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}
