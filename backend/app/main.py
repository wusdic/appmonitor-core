"""FastAPI entrypoint. Builds the runtime, starts the live pipeline, mounts the
API and serves the static frontend."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api import routes
from .pipeline.build import Runtime

FRONTEND_DIR = os.environ.get(
    "APPMON_FRONTEND_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

app = FastAPI(title="AppMonitor Core", version="1.0",
              description="组织信息化业务系统画像平台 — raw/derived metric, behaviour & signature libraries")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    warmup = int(os.environ.get("APPMON_WARMUP_TICKS", "180"))
    period = float(os.environ.get("APPMON_LIVE_PERIOD_S", "3.0"))
    runtime = Runtime(warmup_ticks=warmup, live_period_s=period)
    runtime.start()
    routes.RUNTIME = runtime


app.include_router(routes.router)


@app.get("/")
def index():
    if os.path.isdir(FRONTEND_DIR) and os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return RedirectResponse(url="/app/")
    return {"service": "appmonitor-core", "frontend": "not found", "api": "/api"}


if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
