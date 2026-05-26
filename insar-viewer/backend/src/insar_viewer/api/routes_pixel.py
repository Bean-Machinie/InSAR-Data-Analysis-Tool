"""Pixel time-series endpoint."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from ..project.state import get_project

router = APIRouter(prefix="/api", tags=["pixel"])


def _finite_or_none(v: float) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


@router.get("/pixel")
def pixel(lat: float, lon: float) -> dict[str, Any]:
    try:
        proj = get_project()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    lat_arr, lon_arr = proj.latitudes, proj.longitudes
    dy = abs(lat_arr[1] - lat_arr[0]) if len(lat_arr) > 1 else 1e-4
    dx = abs(lon_arr[1] - lon_arr[0]) if len(lon_arr) > 1 else 1e-4

    if lat < lat_arr[0] - dy or lat > lat_arr[-1] + dy:
        return {"found": False, "reason": "outside grid"}
    if lon < lon_arr[0] - dx or lon > lon_arr[-1] + dx:
        return {"found": False, "reason": "outside grid"}

    row = int(np.argmin(np.abs(lat_arr - lat)))
    col = int(np.argmin(np.abs(lon_arr - lon)))

    if not proj.spatial_mask[row, col]:
        return {"found": False, "reason": "no data at this location"}

    ps = proj.pixel_series
    raw_arr = ps.get("raw")
    if raw_arr is None:
        return {"found": False, "reason": "no displacement data"}

    raw_vals = [_finite_or_none(v) for v in raw_arr[:, row, col]]
    if not any(v is not None for v in raw_vals):
        return {"found": False, "reason": "no finite displacement values"}

    n = len(proj.dates)

    seg_arr = ps.get("segmented")
    seg_id_arr = ps.get("segment_id")
    vtm_arr = ps.get("valid_time_mask")
    cpd_arr = ps.get("coh_per_date")

    segmented_vals = [
        _finite_or_none(v)
        for v in (seg_arr[:, row, col] if seg_arr is not None else [float("nan")] * n)
    ]
    seg_ids = [
        int(v)
        for v in (seg_id_arr[:, row, col] if seg_id_arr is not None else [0] * n)
    ]
    valid_mask = [
        int(v)
        for v in (vtm_arr[:, row, col] if vtm_arr is not None else [0] * n)
    ]
    coh_per_date = [
        _finite_or_none(v)
        for v in (cpd_arr[:, row, col] if cpd_arr is not None else [float("nan")] * n)
    ]

    unique_segs = sorted({s for s in seg_ids if s > 0})
    vel_arr = proj.static_values.get("sbas_velocity_masked")
    velocity_val = _finite_or_none(vel_arr[row, col]) if vel_arr is not None else None
    coh_arr = proj.static_values.get("coherence_median")
    coh_val = _finite_or_none(coh_arr[row, col]) if coh_arr is not None else None
    mask_arr = proj.static_values.get("valid_pixel_mask")
    below_static = bool(mask_arr is not None and mask_arr[row, col] < 0.5)

    return {
        "found": True,
        "lat": float(lat_arr[row]),
        "lon": float(lon_arr[col]),
        "cellBounds": [
            [float(proj.lat_edges[row]), float(proj.lon_edges[col])],
            [float(proj.lat_edges[row + 1]), float(proj.lon_edges[col + 1])],
        ],
        "dates": [d.strftime("%Y-%m-%d") for d in proj.dates],
        "velocity_mm_yr": velocity_val,
        "coherence_median": coh_val,
        "valid_epoch_count": sum(1 for m in valid_mask if m > 0),
        "total_epoch_count": n,
        "segment_count": len(unique_segs),
        "has_gap": len(unique_segs) > 1,
        "below_static_mask": below_static,
        "series": {
            "raw": raw_vals,
            "segmented": segmented_vals,
            "segment_id": seg_ids,
            "valid_time_mask": valid_mask,
            "coh_per_date": coh_per_date,
        },
    }
