"""Server-rendered PNG overlay endpoint."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..project.state import get_project
from ..rendering.png import overlay_png_bytes

router = APIRouter(tags=["overlay"])


@router.get("/overlay/{key}/{date_index}.png")
def overlay(key: str, date_index: int) -> Response:
    try:
        proj = get_project()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from ..project.registry import png_keys
    if key not in png_keys() and key not in proj.available_keys:
        raise HTTPException(status_code=404, detail=f"Layer '{key}' not found.")

    try:
        content = overlay_png_bytes(key, date_index)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
