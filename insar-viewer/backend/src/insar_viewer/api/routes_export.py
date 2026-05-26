"""Export endpoints: viewport CSV/GeoJSON, time-series CSV."""
from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from ..project.state import get_project

router = APIRouter(prefix="/api/export", tags=["export"])


class ViewportBounds(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    layer_key: str = "sbas_velocity_masked"
    date_index: int = 0


@router.post("/viewport-csv")
def viewport_csv(body: ViewportBounds) -> Response:
    try:
        proj = get_project()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    rows, cols = np.where(proj.spatial_mask)
    lat_arr, lon_arr = proj.latitudes, proj.longitudes

    in_view = [
        (r, c) for r, c in zip(rows, cols)
        if body.lat_min <= lat_arr[r] <= body.lat_max
        and body.lon_min <= lon_arr[c] <= body.lon_max
    ]

    key = body.layer_key
    is_temporal = key in proj.temporal_values
    if is_temporal:
        n = proj.temporal_values[key].shape[0]
        idx = max(0, min(n - 1, body.date_index))
        value_arr = proj.temporal_values[key][idx]
    elif key in proj.static_values:
        value_arr = proj.static_values[key]
    else:
        raise HTTPException(status_code=404, detail=f"Layer '{key}' not available.")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["lat", "lon", key])
    for r, c in in_view:
        v = value_arr[r, c]
        writer.writerow([round(float(lat_arr[r]), 6), round(float(lon_arr[c]), 6),
                         round(float(v), 4) if math.isfinite(float(v)) else ""])

    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="insar_viewport_{key}.csv"'},
    )


@router.get("/timeseries-csv")
def timeseries_csv(lat: float, lon: float) -> Response:
    try:
        proj = get_project()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    lat_arr, lon_arr = proj.latitudes, proj.longitudes
    row = int(np.argmin(np.abs(lat_arr - lat)))
    col = int(np.argmin(np.abs(lon_arr - lon)))

    if not proj.spatial_mask[row, col]:
        raise HTTPException(status_code=404, detail="No data at this location.")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "displacement_raw_mm", "displacement_segmented_mm", "coherence"])

    raw_arr = proj.pixel_series.get("raw")
    seg_arr = proj.pixel_series.get("segmented")
    coh_arr = proj.pixel_series.get("coh_per_date")

    for i, d in enumerate(proj.dates):
        rv = raw_arr[i, row, col] if raw_arr is not None else float("nan")
        sv = seg_arr[i, row, col] if seg_arr is not None else float("nan")
        cv = coh_arr[i, row, col] if coh_arr is not None else float("nan")

        def fmt(x: float) -> str:
            return str(round(float(x), 4)) if math.isfinite(float(x)) else ""

        writer.writerow([d.strftime("%Y-%m-%d"), fmt(rv), fmt(sv), fmt(cv)])

    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="insar_timeseries.csv"'},
    )
