"""Project metadata and settings endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..project.state import get_project, is_loaded
from ..settings import (
    AppSettings,
    RecentProject,
    load_recent,
    load_settings,
    save_settings,
)

router = APIRouter(tags=["metadata"])


@router.get("/api/metadata")
def metadata() -> dict[str, Any]:
    if not is_loaded():
        raise HTTPException(status_code=404, detail="No project loaded.")
    proj = get_project()
    p = proj.parameters
    return {
        "parameters": p.model_dump(),
        "manifest": proj.manifest.model_dump() if proj.manifest else None,
        "ncFile": proj.nc_path.name,
        "pois": [poi.model_dump() for poi in p.pois],
    }


@router.get("/api/settings", response_model=AppSettings)
def get_settings() -> AppSettings:
    return load_settings()


@router.put("/api/settings")
def put_settings(body: AppSettings) -> AppSettings:
    save_settings(body)
    return body


@router.get("/api/recent")
def recent_projects() -> list[RecentProject]:
    return load_recent()
