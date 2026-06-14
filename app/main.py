from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import get_settings
from app.services.import_service import ImportService
from app.services.product_service import ProductService


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)
    app.include_router(api_router, prefix="/api")
    @app.on_event("startup")
    async def warm_service_caches() -> None:
        ProductService()
        ImportService()

    @app.on_event("shutdown")
    async def close_service_caches() -> None:
        ImportService.reset_shared_state()
        ProductService.reset_shared_state()
    return app


app = create_app()
