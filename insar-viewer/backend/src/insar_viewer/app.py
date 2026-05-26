"""FastAPI application factory."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes_export import router as export_router
from .api.routes_metadata import router as metadata_router
from .api.routes_overlay import router as overlay_router
from .api.routes_pixel import router as pixel_router
from .api.routes_points import router as points_router
from .api.routes_project import router as project_router

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent.parent.parent.parent.parent / "frontend" / "dist"


def create_app(dev_mode: bool = False) -> FastAPI:
    app = FastAPI(title="InSAR Viewer", version="1.0.0", docs_url="/api/docs")

    if dev_mode:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(project_router)
    app.include_router(pixel_router)
    app.include_router(points_router)
    app.include_router(overlay_router)
    app.include_router(metadata_router)
    app.include_router(export_router)

    # Serve built frontend in production
    if not dev_mode and FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
        logger.info("Serving frontend from %s", FRONTEND_DIST)
    elif not dev_mode:
        logger.warning("Frontend dist not found at %s — run npm run build first.", FRONTEND_DIST)

    return app
