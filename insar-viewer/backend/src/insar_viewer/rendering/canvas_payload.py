"""Build the JSON payload for client-side canvas rendering."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..project.state import get_project


def _f(v: float) -> float | None:
    try:
        fv = float(v)
        return round(fv, 3) if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


def build_points_payload() -> dict[str, Any]:
    proj = get_project()
    rows, cols = np.where(proj.spatial_mask)
    n = int(len(rows))

    if n == 0:
        return {
            "count": 0, "lats": [], "lons": [],
            "vel_raw": None, "vel_masked": None,
            "coherence": None, "disp": None,
            "cellLat": 0.001, "cellLon": 0.001,
        }

    def static(key: str) -> list[float | None] | None:
        arr = proj.static_values.get(key)
        return [_f(arr[r, c]) for r, c in zip(rows, cols)] if arr is not None else None

    def temporal(key: str) -> list[list[float | None]] | None:
        arr = proj.temporal_values.get(key)
        if arr is None:
            return None
        return [[_f(arr[t, r, c]) for r, c in zip(rows, cols)] for t in range(arr.shape[0])]

    lat_arr, lon_arr = proj.latitudes, proj.longitudes
    cell_lat = float(abs(lat_arr[1] - lat_arr[0])) if len(lat_arr) > 1 else 0.001
    cell_lon = float(abs(lon_arr[1] - lon_arr[0])) if len(lon_arr) > 1 else 0.001

    return {
        "count": n,
        "lats": [round(float(lat_arr[r]), 6) for r in rows],
        "lons": [round(float(lon_arr[c]), 6) for c in cols],
        "vel_raw": static("sbas_velocity_raw"),
        "vel_masked": static("sbas_velocity_masked"),
        "coherence": static("coherence_median"),
        "disp": temporal("sbas_displacement_masked"),
        "cellLat": round(cell_lat, 7),
        "cellLon": round(cell_lon, 7),
    }
