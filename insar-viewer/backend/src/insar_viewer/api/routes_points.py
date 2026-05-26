"""Bulk canvas points endpoint."""
from fastapi import APIRouter, HTTPException
from ..project.state import get_project
from ..rendering.canvas_payload import build_points_payload

router = APIRouter(prefix="/api", tags=["points"])


@router.get("/points")
def points():
    try:
        get_project()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return build_points_payload()
