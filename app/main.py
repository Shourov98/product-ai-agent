from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
