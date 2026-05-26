"""Project open/close/info endpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..project import state
from ..project.registry import ALL_SPECS, canvas_keys, png_keys
from ..settings import add_recent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/project", tags=["project"])

BASEMAP_SPECS = [
    {
        "key": "esri_satellite",
        "label": "Esri Satellite",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles &copy; Esri",
        "defaultEnabled": True,
        "defaultOpacity": 1.0,
        "maxZoom": 18,
    },
    {
        "key": "esri_hillshade",
        "label": "Esri Hillshade",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles &copy; Esri",
        "defaultEnabled": False,
        "defaultOpacity": 1.0,
        "maxZoom": 13,
    },
    {
        "key": "openstreetmap",
        "label": "OpenStreetMap",
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "&copy; OpenStreetMap contributors",
        "defaultEnabled": False,
        "defaultOpacity": 0.9,
        "maxZoom": 19,
    },
]

_LAYER_DEFAULTS: dict[str, dict[str, Any]] = {
    "sbas_velocity_masked": {"defaultEnabled": True, "defaultOpacity": 0.87},
    "sbas_velocity_raw": {"defaultEnabled": False, "defaultOpacity": 0.82},
    "sbas_displacement_masked": {"defaultEnabled": False, "defaultOpacity": 0.86},
    "coherence_median": {"defaultEnabled": False, "defaultOpacity": 0.78},
    "valid_pixel_mask": {"defaultEnabled": False, "defaultOpacity": 0.65},
    "dem": {"defaultEnabled": False, "defaultOpacity": 0.72},
    "sbas_rmse_masked": {"defaultEnabled": False, "defaultOpacity": 0.72},
}


class OpenRequest(BaseModel):
    path: str


@router.post("/open")
def open_project(body: OpenRequest) -> dict[str, Any]:
    try:
        proj = state.open_project(Path(body.path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load project: %s", body.path)
        raise HTTPException(status_code=500, detail=f"Failed to load project: {exc}") from exc

    from ..rendering.png import invalidate_cache
    invalidate_cache()

    add_recent(
        path=str(proj.root_dir),
        name=proj.parameters.project,
    )

    return _build_info(proj)


@router.get("/info")
def project_info() -> dict[str, Any]:
    if not state.is_loaded():
        raise HTTPException(status_code=404, detail="No project loaded.")
    return _build_info(state.get_project())


@router.post("/close")
def close_project() -> dict[str, str]:
    state.close_project()
    return {"status": "closed"}


def _build_info(proj: Any) -> dict[str, Any]:
    canvas = set(canvas_keys())
    png = set(png_keys())

    data_layers = []
    for spec in ALL_SPECS:
        if spec.key not in proj.available_keys:
            continue
        defs = _LAYER_DEFAULTS.get(spec.key, {"defaultEnabled": False, "defaultOpacity": 0.75})
        lo, hi = proj.layer_ranges.get(spec.key, (0.0, 1.0))
        data_layers.append({
            "key": spec.key,
            "label": spec.display_name,
            "kind": "canvas" if spec.key in canvas else "png",
            "units": spec.units,
            "temporal": spec.dimensions == "temporal",
            "symmetric": spec.symmetric,
            "defaultEnabled": defs["defaultEnabled"],
            "defaultOpacity": defs["defaultOpacity"],
            "valueRange": [round(lo, 4), round(hi, 4)],
            "colormap": spec.default_colormap,
        })

    # AOI layer always appended if WKT exists
    if proj.aoi_lons:
        data_layers.append({
            "key": "aoi_original",
            "label": "AOI boundary",
            "kind": "aoi",
            "units": "",
            "temporal": False,
            "symmetric": False,
            "defaultEnabled": True,
            "defaultOpacity": 1.0,
            "valueRange": None,
            "colormap": "",
        })

    dates = [d.strftime("%Y-%m-%d") for d in proj.dates]
    p = proj.parameters

    return {
        "projectName": p.project,
        "orbit": p.orbit,
        "dateRange": {
            "start": p.time_window.start,
            "end": p.time_window.end,
        },
        "sceneCount": p.scenes.count,
        "dates": dates,
        "defaultDateIndex": len(dates) - 1 if dates else 0,
        "pois": [poi.model_dump() for poi in p.pois],
        "center": [proj.center_lat, proj.center_lon],
        "bounds": [
            [float(proj.lat_edges[0]), float(proj.lon_edges[0])],
            [float(proj.lat_edges[-1]), float(proj.lon_edges[-1])],
        ],
        "aoi": (
            [[float(lat), float(lon)] for lon, lat in zip(proj.aoi_lons, proj.aoi_lats)]
            if proj.aoi_lons else None
        ),
        "baseLayers": BASEMAP_SPECS,
        "dataLayers": data_layers,
        "cohThresholdDefault": 0.30,
        "cohSliderStep": 0.05,
        "ncFile": proj.nc_path.name,
    }
