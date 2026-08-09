from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.registry.engine_registry import bootstrap_default_engines
from infra.filesystem_paths import ensure_project_directories
from webui import routes_step1, routes_step2, routes_step3
from webui.template_engine import WEBUI_DIR


def create_app() -> FastAPI:
    ensure_project_directories()
    bootstrap_default_engines()

    app = FastAPI(title="OneManDubbing")
    app.mount("/static", StaticFiles(directory=str(WEBUI_DIR / "static")), name="static")

    app.include_router(routes_step1.router)
    app.include_router(routes_step2.router)
    app.include_router(routes_step3.router)

    return app


app = create_app()
