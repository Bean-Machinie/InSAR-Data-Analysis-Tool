"""InSAR SBAS Results Viewer

Flask + Leaflet + Plotly.js viewer for one SBAS processing run.

Inputs (from the product folder):
    results_tight.nc              — main NetCDF with all maps and time-series cubes
    parameters.json               — AOI WKT, processing parameters, POI list
    sbas_results_metadata.json    — units, thresholds, array shapes, notes

Run:
    python insar_deformation_viewer.py Data/project_D_results_only
    open http://127.0.0.1:8050

Dependencies:
    pip install flask xarray netcdf4 pandas matplotlib pillow numpy
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from flask import Flask, Response, jsonify, request
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image

from insar_project import (
    ProjectPaths,
    add_project_dir_argument,
    find_netcdf_files,
    resolve_project_paths,
)


# ── Runtime constants ──────────────────────────────────────────────────────────

DEFAULT_ZOOM = 13
DEFAULT_COH_THRESHOLD = 0.30   # matches processing default (parameters.json coh_threshold)
COH_SLIDER_STEP = 0.05
SEGMENT_COLORS = ["#2166ac", "#e08214", "#4dac26", "#9e0142", "#abd9e9"]

# Variables loaded only for pixel-level queries (not rendered as map overlays)
PIXEL_SERIES_VARS: dict[str, str] = {
    "raw": "sbas_displacement_raw",
    "segmented": "sbas_displacement_segmented_same_pixel",
    "segment_id": "sbas_segment_id",
    "valid_time_mask": "sbas_valid_time_mask",
    "coh_per_date": "coherence_per_date",
}


# ── Spec dataclasses ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TileLayerSpec:
    key: str
    label: str
    url: str
    attribution: str
    default_enabled: bool
    default_opacity: float
    max_zoom: int


@dataclass(frozen=True)
class DataLayerSpec:
    key: str
    label: str
    variable: str | None
    kind: str            # 'scalar' | 'mask' | 'aoi'
    units: str
    colormap: str
    default_enabled: bool
    default_opacity: float
    temporal: bool = False
    symmetric: bool = False


BASEMAP_SPECS: tuple[TileLayerSpec, ...] = (
    TileLayerSpec(
        key="esri_satellite",
        label="Esri Satellite",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Tiles &copy; Esri",
        default_enabled=True,
        default_opacity=1.0,
        max_zoom=18,
    ),
    TileLayerSpec(
        key="esri_hillshade",
        label="Esri Hillshade",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        attribution="Tiles &copy; Esri",
        default_enabled=False,
        default_opacity=1.0,
        max_zoom=13,
    ),
    TileLayerSpec(
        key="openstreetmap",
        label="OpenStreetMap",
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="&copy; OpenStreetMap contributors",
        default_enabled=False,
        default_opacity=0.9,
        max_zoom=19,
    ),
)

REQUESTED_DATA_LAYERS: tuple[DataLayerSpec, ...] = (
    DataLayerSpec(
        key="sbas_velocity_masked",
        label="Velocity — masked (mm/yr)",
        variable="sbas_velocity_masked",
        kind="scalar",
        units="mm/yr",
        colormap="RdBu_r",
        default_enabled=True,
        default_opacity=0.87,
        symmetric=True,
    ),
    DataLayerSpec(
        key="sbas_velocity_raw",
        label="Velocity — raw (mm/yr)",
        variable="sbas_velocity_raw",
        kind="scalar",
        units="mm/yr",
        colormap="RdBu_r",
        default_enabled=False,
        default_opacity=0.82,
        symmetric=True,
    ),
    DataLayerSpec(
        key="sbas_displacement_masked",
        label="Displacement — masked (mm)",
        variable="sbas_displacement_masked",
        kind="scalar",
        units="mm",
        colormap="RdBu_r",
        default_enabled=False,
        default_opacity=0.86,
        temporal=True,
        symmetric=True,
    ),
    DataLayerSpec(
        key="coherence_median",
        label="Coherence (median)",
        variable="coherence_median",
        kind="scalar",
        units="",
        colormap="viridis",
        default_enabled=False,
        default_opacity=0.78,
    ),
    DataLayerSpec(
        key="valid_pixel_mask",
        label="Valid pixel mask",
        variable="valid_pixel_mask",
        kind="mask",
        units="",
        colormap="mask",
        default_enabled=False,
        default_opacity=0.65,
    ),
    DataLayerSpec(
        key="dem",
        label="DEM (terrain elevation, m)",
        variable="dem",
        kind="scalar",
        units="m",
        colormap="terrain",
        default_enabled=False,
        default_opacity=0.72,
    ),
    DataLayerSpec(
        key="aoi_original",
        label="AOI boundary",
        variable=None,
        kind="aoi",
        units="",
        colormap="",
        default_enabled=True,
        default_opacity=1.0,
    ),
)


# ── ViewerData ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ViewerData:
    dataset_path: Path
    title: str
    dates: pd.DatetimeIndex
    latitudes: np.ndarray
    longitudes: np.ndarray
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    spatial_mask: np.ndarray                  # [lat, lon] bool
    layer_specs: tuple[DataLayerSpec, ...]
    layer_ranges: dict[str, tuple[float, float]]
    static_values: dict[str, np.ndarray]      # [lat, lon] for overlay rendering
    temporal_values: dict[str, np.ndarray]    # [date, lat, lon] for overlay rendering
    pixel_series: dict[str, np.ndarray]       # [date, lat, lon] for pixel queries only
    pois: tuple[dict, ...]
    metadata: dict[str, Any]
    parameters: dict[str, Any]
    aoi_lons: np.ndarray | None
    aoi_lats: np.ndarray | None
    center_lat: float
    center_lon: float


VIEWER_DATA: ViewerData | None = None


# ── Data loading ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_dir_argument(parser)
    parser.add_argument("--dataset", type=Path, help="Override NetCDF path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def default_dataset_path(project_paths: ProjectPaths) -> Path:
    for candidate in (
        project_paths.product_dir / "results_tight.nc",
        project_paths.product_dir / "results_aoi_masked.nc",
        project_paths.product_dir / "results_wide.nc",
    ):
        if candidate.exists():
            return candidate
    nc_files = find_netcdf_files(project_paths)
    if nc_files:
        return nc_files[0]
    raise FileNotFoundError(f"No NetCDF dataset found under {project_paths.product_dir}")


def load_json_safe(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def parse_polygon_wkt(wkt: str | None) -> list[tuple[float, float]]:
    if not wkt:
        return []
    match = re.match(r"^\s*POLYGON\s*\(\((.+)\)\)\s*$", wkt, flags=re.IGNORECASE)
    if not match:
        return []
    coords: list[tuple[float, float]] = []
    for point in match.group(1).split(","):
        parts = point.strip().split()
        if len(parts) < 2:
            return []
        coords.append((float(parts[0]), float(parts[1])))
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    if values.size == 1:
        d = 0.0001
        return np.array([values[0] - d / 2, values[0] + d / 2])
    midpoints = (values[:-1] + values[1:]) / 2
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def build_spatial_mask(dataset: xr.Dataset) -> np.ndarray:
    # Use sbas_velocity_raw as the extent mask: covers all processed pixels
    # regardless of AOI or coherence threshold (the broadest meaningful extent).
    for var_name in ("sbas_velocity_raw", "sbas_velocity_masked"):
        if var_name in dataset.data_vars:
            arr = dataset[var_name].transpose("lat", "lon").values
            mask = np.isfinite(arr)
            if np.any(mask):
                return mask

    # Fallback: any pixel finite in any data variable
    candidates = []
    for spec in REQUESTED_DATA_LAYERS:
        if spec.variable is None or spec.variable not in dataset.data_vars:
            continue
        var = dataset[spec.variable]
        if "date" in var.dims:
            arr = var.transpose("date", "lat", "lon").values
            candidates.append(np.isfinite(arr).any(axis=0))
        else:
            arr = var.transpose("lat", "lon").values
            candidates.append(np.isfinite(arr))

    if not candidates:
        raise ValueError("No spatial data layers found in dataset.")
    return np.logical_or.reduce(candidates)


def robust_range(values: np.ndarray, mask: np.ndarray, symmetric: bool) -> tuple[float, float]:
    flat = values[:, mask] if values.ndim == 3 else values[mask]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        limit = float(np.nanpercentile(np.abs(flat), 98))
        if not math.isfinite(limit) or limit == 0:
            limit = 1.0
        return -limit, limit
    low = float(np.nanpercentile(flat, 2))
    high = float(np.nanpercentile(flat, 98))
    if low == high:
        low, high = float(np.nanmin(flat)), float(np.nanmax(flat))
    if low == high:
        high = low + 1.0
    return low, high


def _parse_poi_latlon_from_csv(path: Path) -> tuple[float | None, float | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                m = re.search(r"lon\s*=\s*([-\d.]+).*lat\s*=\s*([-\d.]+)", line)
                if m:
                    return float(m.group(2)), float(m.group(1))
    except OSError:
        pass
    return None, None


def load_pois(project_paths: ProjectPaths, parameters: dict) -> tuple[dict, ...]:
    pois: list[dict] = []
    for poi in parameters.get("pois", []):
        name = poi.get("name", "")
        lat = poi.get("lat")
        lon = poi.get("lon")
        if not name or lat is None or lon is None:
            continue
        has_csv = (project_paths.timeseries_dir / f"{name}.csv").exists()
        pois.append({"name": name, "lat": float(lat), "lon": float(lon), "has_csv": has_csv})

    existing = {p["name"] for p in pois}
    if project_paths.timeseries_dir.exists():
        for csv_path in sorted(project_paths.timeseries_dir.glob("*.csv")):
            if csv_path.stem not in existing:
                lat, lon = _parse_poi_latlon_from_csv(csv_path)
                if lat is not None and lon is not None:
                    pois.append({"name": csv_path.stem, "lat": lat, "lon": lon, "has_csv": True})

    return tuple(pois)


def load_viewer_data(dataset_path: Path, project_paths: ProjectPaths) -> ViewerData:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    parameters = load_json_safe(project_paths.parameters_path)
    metadata = load_json_safe(project_paths.metadata_path)

    aoi_wkt = (parameters.get("aoi") or {}).get("raw_wkt")
    aoi_coords = parse_polygon_wkt(aoi_wkt)
    aoi_lons = np.array([lon for lon, _ in aoi_coords]) if aoi_coords else None
    aoi_lats = np.array([lat for _, lat in aoi_coords]) if aoi_coords else None

    with xr.open_dataset(dataset_path) as raw_ds:
        dataset = raw_ds.sortby("lat").sortby("lon").load()

    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError("Dataset must have lat/lon coordinates.")

    latitudes = dataset["lat"].values.astype("float64")
    longitudes = dataset["lon"].values.astype("float64")
    spatial_mask = build_spatial_mask(dataset)

    if not np.any(spatial_mask):
        raise ValueError("No pixels inside the spatial mask.")

    layer_specs = tuple(
        spec for spec in REQUESTED_DATA_LAYERS
        if spec.kind == "aoi" or (spec.variable is not None and spec.variable in dataset.data_vars)
    )

    static_values: dict[str, np.ndarray] = {}
    temporal_values: dict[str, np.ndarray] = {}
    layer_ranges: dict[str, tuple[float, float]] = {}

    for spec in layer_specs:
        if spec.variable is None:
            continue
        var = dataset[spec.variable]
        if spec.temporal:
            arr = var.transpose("date", "lat", "lon").values.astype("float64")
            temporal_values[spec.key] = arr
        else:
            arr = var.transpose("lat", "lon").values.astype("float64")
            static_values[spec.key] = arr
        if spec.kind == "mask":
            layer_ranges[spec.key] = (0.0, 1.0)
        else:
            layer_ranges[spec.key] = robust_range(arr, spatial_mask, spec.symmetric)

    if "date" in dataset.coords:
        dates = pd.DatetimeIndex(pd.to_datetime(dataset["date"].values))
    else:
        dates = pd.DatetimeIndex([])

    # Pixel-level query arrays — not used for overlay rendering
    pixel_series: dict[str, np.ndarray] = {}
    for key, var_name in PIXEL_SERIES_VARS.items():
        if var_name not in dataset.data_vars:
            continue
        var = dataset[var_name]
        if "date" not in var.dims:
            continue
        arr = var.transpose("date", "lat", "lon").values
        pixel_series[key] = arr.astype("float64") if arr.dtype.kind == "f" else arr.astype("int32")

    mask_rows, mask_cols = np.where(spatial_mask)
    center_lat = float(np.nanmean(latitudes[mask_rows]))
    center_lon = float(np.nanmean(longitudes[mask_cols]))
    title = str(dataset.attrs.get("title") or dataset_path.stem)
    pois = load_pois(project_paths, parameters)

    return ViewerData(
        dataset_path=dataset_path,
        title=title,
        dates=dates,
        latitudes=latitudes,
        longitudes=longitudes,
        lat_edges=coordinate_edges(latitudes),
        lon_edges=coordinate_edges(longitudes),
        spatial_mask=spatial_mask,
        layer_specs=layer_specs,
        layer_ranges=layer_ranges,
        static_values=static_values,
        temporal_values=temporal_values,
        pixel_series=pixel_series,
        pois=pois,
        metadata=metadata,
        parameters=parameters,
        aoi_lons=aoi_lons,
        aoi_lats=aoi_lats,
        center_lat=center_lat,
        center_lon=center_lon,
    )


# ── Runtime accessors ──────────────────────────────────────────────────────────

def require_viewer_data() -> ViewerData:
    if VIEWER_DATA is None:
        raise RuntimeError("Viewer data not loaded.")
    return VIEWER_DATA


def spec_by_key(key: str) -> DataLayerSpec:
    for spec in require_viewer_data().layer_specs:
        if spec.key == key:
            return spec
    raise KeyError(key)


def clamp_date_index(idx: int | None) -> int:
    data = require_viewer_data()
    n = len(data.dates)
    if not n:
        return 0
    if idx is None:
        return n - 1
    return max(0, min(n - 1, int(idx)))


def finite_or_none(v: float) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def format_number(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


# ── Overlay rendering ──────────────────────────────────────────────────────────

def _rgba_to_uint8(colors: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgba = np.clip(colors * 255, 0, 255).astype("uint8")
    rgba[..., 3] = np.clip(alpha * 255, 0, 255).astype("uint8")
    return rgba


def _png_bytes(rgba: np.ndarray) -> bytes:
    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@lru_cache(maxsize=128)
def overlay_png_bytes(key: str, date_index: int) -> bytes:
    data = require_viewer_data()
    spec = spec_by_key(key)
    if spec.kind == "aoi":
        raise ValueError("AOI is a vector layer, not an image overlay.")

    values = (
        data.temporal_values[key][clamp_date_index(date_index)]
        if spec.temporal
        else data.static_values[key]
    )

    if spec.kind == "mask":
        mask_vals = np.where(np.isfinite(values), values > 0.5, False)
        rgba = np.zeros(values.shape + (4,), dtype="uint8")
        rgba[data.spatial_mask & ~mask_vals] = [154, 167, 173, 185]
        rgba[data.spatial_mask & mask_vals] = [33, 166, 122, 225]
    else:
        visible = data.spatial_mask & np.isfinite(values)
        low, high = data.layer_ranges[key]
        norm = (
            TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
            if spec.symmetric
            else Normalize(vmin=low, vmax=high)
        )
        cmap = colormaps.get_cmap(spec.colormap)
        normalized = np.zeros(values.shape, dtype="float64")
        normalized[visible] = norm(values[visible])
        colors = cmap(normalized)
        alpha = np.where(visible, 1.0, 0.0)
        rgba = _rgba_to_uint8(colors, alpha)

    return _png_bytes(rgba)


@lru_cache(maxsize=64)
def coh_filtered_velocity_png_bytes(threshold_pct: int) -> bytes:
    """Render sbas_velocity_raw, masking out pixels where coherence_median < threshold."""
    data = require_viewer_data()
    velocity = data.static_values.get("sbas_velocity_raw")
    coherence = data.static_values.get("coherence_median")
    if velocity is None or coherence is None:
        raise ValueError("sbas_velocity_raw or coherence_median not available.")

    threshold = threshold_pct / 100.0
    visible = data.spatial_mask & np.isfinite(velocity) & (coherence >= threshold)

    low, high = data.layer_ranges.get(
        "sbas_velocity_raw",
        data.layer_ranges.get("sbas_velocity_masked", (-10.0, 10.0)),
    )
    norm = TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
    cmap = colormaps.get_cmap("RdBu_r")

    normalized = np.zeros(velocity.shape, dtype="float64")
    normalized[visible] = norm(velocity[visible])
    colors = cmap(normalized)
    alpha = np.where(visible, 1.0, 0.0)
    return _png_bytes(_rgba_to_uint8(colors, alpha))


def _build_legend(spec: DataLayerSpec) -> dict | None:
    if spec.kind in ("aoi", "mask"):
        return None
    data = require_viewer_data()
    low, high = data.layer_ranges.get(spec.key, (0.0, 1.0))
    cmap_colors = {
        "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
        "terrain": ["#333399", "#006600", "#c8a450", "#ffffff"],
    }
    colors = cmap_colors.get(spec.colormap, ["#08306b", "#f7f7f7", "#67000d"])
    return {"low": format_number(low), "high": format_number(high), "colors": colors}


# ── API payload builders ───────────────────────────────────────────────────────

def build_viewer_payload() -> dict:
    data = require_viewer_data()
    vel_range = data.layer_ranges.get(
        "sbas_velocity_raw",
        data.layer_ranges.get("sbas_velocity_masked", (-10.0, 10.0)),
    )
    return {
        "title": data.title,
        "dataset": data.dataset_path.name,
        "center": [data.center_lat, data.center_lon],
        "bounds": [
            [float(data.lat_edges[0]), float(data.lon_edges[0])],
            [float(data.lat_edges[-1]), float(data.lon_edges[-1])],
        ],
        "defaultZoom": DEFAULT_ZOOM,
        "dates": [d.strftime("%Y-%m-%d") for d in data.dates],
        "defaultDateIndex": clamp_date_index(None),
        "cohThresholdDefault": DEFAULT_COH_THRESHOLD,
        "cohSliderStep": COH_SLIDER_STEP,
        "velocityRange": list(vel_range),
        "baseLayers": [
            {
                "key": s.key, "label": s.label, "url": s.url,
                "attribution": s.attribution,
                "defaultEnabled": s.default_enabled,
                "defaultOpacity": s.default_opacity,
                "maxZoom": s.max_zoom,
            }
            for s in BASEMAP_SPECS
        ],
        "dataLayers": [
            {
                "key": s.key, "label": s.label, "kind": s.kind,
                "units": s.units, "temporal": s.temporal,
                "defaultEnabled": s.default_enabled,
                "defaultOpacity": s.default_opacity,
                "legend": _build_legend(s),
            }
            for s in data.layer_specs
        ],
        "aoi": (
            [[float(lat), float(lon)] for lon, lat in zip(data.aoi_lons, data.aoi_lats)]
            if data.aoi_lons is not None else None
        ),
        "pois": list(data.pois),
        "segmentColors": SEGMENT_COLORS,
    }


def coherence_grid_payload() -> dict:
    data = require_viewer_data()
    coh = data.static_values.get("coherence_median")
    if coh is None:
        return {"available": False}
    flat = coh.ravel()
    mask_flat = data.spatial_mask.ravel()
    return {
        "available": True,
        "shape": list(coh.shape),
        "values": [round(float(v), 4) if math.isfinite(float(v)) else None for v in flat],
        "valid_mask": [int(v) for v in mask_flat],
        "total_valid": int(np.sum(mask_flat)),
    }


def cell_index_from_latlon(lat: float, lon: float) -> tuple[int, int] | None:
    data = require_viewer_data()
    lat_arr, lon_arr = data.latitudes, data.longitudes
    dy = abs(lat_arr[1] - lat_arr[0]) if len(lat_arr) > 1 else 1e-4
    dx = abs(lon_arr[1] - lon_arr[0]) if len(lon_arr) > 1 else 1e-4
    if lat < lat_arr[0] - dy or lat > lat_arr[-1] + dy:
        return None
    if lon < lon_arr[0] - dx or lon > lon_arr[-1] + dx:
        return None
    i = int(np.argmin(np.abs(lat_arr - lat)))
    j = int(np.argmin(np.abs(lon_arr - lon)))
    return i, j


def pixel_payload(lat: float, lon: float) -> dict:
    data = require_viewer_data()
    index = cell_index_from_latlon(lat, lon)
    if index is None:
        return {"found": False, "reason": "outside grid"}
    row, col = index
    if not data.spatial_mask[row, col]:
        return {"found": False, "reason": "no data at this location"}

    ps = data.pixel_series
    raw_arr = ps.get("raw")
    if raw_arr is None:
        return {"found": False, "reason": "no displacement data"}

    raw_vals = [finite_or_none(v) for v in raw_arr[:, row, col]]
    if not any(v is not None for v in raw_vals):
        return {"found": False, "reason": "no finite displacement values"}

    n = len(data.dates)

    seg_arr = ps.get("segmented")
    seg_id_arr = ps.get("segment_id")
    vtm_arr = ps.get("valid_time_mask")
    cpd_arr = ps.get("coh_per_date")

    segmented_vals = [finite_or_none(v) for v in (seg_arr[:, row, col] if seg_arr is not None else [float("nan")] * n)]
    seg_ids = [int(v) for v in (seg_id_arr[:, row, col] if seg_id_arr is not None else [0] * n)]
    valid_mask = [int(v) for v in (vtm_arr[:, row, col] if vtm_arr is not None else [0] * n)]
    coh_per_date = [finite_or_none(v) for v in (cpd_arr[:, row, col] if cpd_arr is not None else [float("nan")] * n)]

    unique_segs = sorted({s for s in seg_ids if s > 0})
    seg_count = len(unique_segs)
    valid_count = sum(1 for m in valid_mask if m > 0)

    vel_arr = data.static_values.get("sbas_velocity_masked")
    velocity_val = finite_or_none(vel_arr[row, col]) if vel_arr is not None else None

    coh_arr = data.static_values.get("coherence_median")
    coh_val = finite_or_none(coh_arr[row, col]) if coh_arr is not None else None

    mask_arr = data.static_values.get("valid_pixel_mask")
    below_static_mask = bool(mask_arr is not None and mask_arr[row, col] < 0.5)

    return {
        "found": True,
        "lat": float(data.latitudes[row]),
        "lon": float(data.longitudes[col]),
        "cellBounds": [
            [float(data.lat_edges[row]), float(data.lon_edges[col])],
            [float(data.lat_edges[row + 1]), float(data.lon_edges[col + 1])],
        ],
        "dates": [d.strftime("%Y-%m-%d") for d in data.dates],
        "velocity_mm_yr": velocity_val,
        "coherence_median": coh_val,
        "valid_epoch_count": valid_count,
        "total_epoch_count": n,
        "segment_count": seg_count,
        "has_gap": seg_count > 1,
        "below_static_mask": below_static_mask,
        "series": {
            "raw": raw_vals,
            "segmented": segmented_vals,
            "segment_id": seg_ids,
            "valid_time_mask": valid_mask,
            "coh_per_date": coh_per_date,
        },
    }


def metadata_payload() -> dict:
    data = require_viewer_data()
    return {
        "metadata": data.metadata,
        "parameters": data.parameters,
        "dataset": data.dataset_path.name,
        "pois": list(data.pois),
    }


def points_payload() -> dict:
    """All valid pixel coordinates + velocity/coherence/displacement values for canvas rendering."""
    data = require_viewer_data()
    rows, cols = np.where(data.spatial_mask)
    n = int(len(rows))
    if n == 0:
        return {"count": 0, "lats": [], "lons": [],
                "vel_raw": None, "vel_masked": None, "coherence": None, "disp": None,
                "cellLat": 0.001, "cellLon": 0.001}

    def _f(v: float) -> float | None:
        return round(float(v), 3) if math.isfinite(float(v)) else None

    def static(key: str) -> list | None:
        arr = data.static_values.get(key)
        return [_f(arr[r, c]) for r, c in zip(rows, cols)] if arr is not None else None

    def temporal(key: str) -> list[list] | None:
        arr = data.temporal_values.get(key)
        if arr is None:
            return None
        return [[_f(arr[t, r, c]) for r, c in zip(rows, cols)] for t in range(arr.shape[0])]

    lat_arr, lon_arr = data.latitudes, data.longitudes
    cell_lat = float(abs(lat_arr[1] - lat_arr[0])) if len(lat_arr) > 1 else 0.001
    cell_lon = float(abs(lon_arr[1] - lon_arr[0])) if len(lon_arr) > 1 else 0.001

    return {
        "count": n,
        "lats": [round(float(lat_arr[r]), 6) for r in rows],
        "lons": [round(float(lon_arr[c]), 6) for c in cols],
        "vel_raw":   static("sbas_velocity_raw"),
        "vel_masked": static("sbas_velocity_masked"),
        "coherence":  static("coherence_median"),
        "disp":       temporal("sbas_displacement_masked"),
        "cellLat": round(cell_lat, 7),
        "cellLon": round(cell_lon, 7),
    }


# ── Flask app ──────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(APP_HTML, mimetype="text/html")

    @app.get("/api/viewer")
    def viewer_api():
        return jsonify(build_viewer_payload())

    @app.get("/api/pixel")
    def pixel_api():
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
        return jsonify(pixel_payload(lat, lon))

    @app.get("/api/coherence_grid")
    def coherence_grid_api():
        return jsonify(coherence_grid_payload())

    @app.get("/api/metadata")
    def metadata_api():
        return jsonify(metadata_payload())

    @app.get("/overlay/<key>/<int:date_index>.png")
    def overlay_api(key: str, date_index: int) -> Response:
        content = overlay_png_bytes(key, clamp_date_index(date_index))
        return Response(content, mimetype="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/points")
    def points_api():
        return jsonify(points_payload())

    return app


# ── Embedded HTML/CSS/JS ───────────────────────────────────────────────────────

APP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>InSAR SBAS Viewer</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --top-h: 48px;
      --bottom-h: 56px;
      --left-w: 224px;
      --right-w: 296px;
      --bg: #0b1825;
      --panel: #0f2033;
      --panel2: #162840;
      --border: rgba(255,255,255,0.07);
      --border2: rgba(255,255,255,0.12);
      --text: #cce0f0;
      --text2: #6a91ae;
      --accent: #29b6f6;
      --accent2: #00c896;
      --danger: #ef5350;
      --warn: #ffb74d;
      --radius: 8px;
      --tr: 0.18s ease;
    }

    html, body { height: 100%; overflow: hidden; font-family: "Inter","Segoe UI",system-ui,sans-serif; background: var(--bg); color: var(--text); font-size: 13px; }

    /* ── Top bar ── */
    #top-bar {
      position: fixed; top: 0; left: 0; right: 0; height: var(--top-h); z-index: 1000;
      display: flex; align-items: center; gap: 10px; padding: 0 14px;
      background: rgba(11,24,37,0.97); border-bottom: 1px solid var(--border2);
      backdrop-filter: blur(10px);
    }
    #sidebar-toggle {
      display: grid; place-items: center; width: 32px; height: 32px;
      border: 1px solid var(--border2); border-radius: 6px; background: transparent;
      color: var(--text2); cursor: pointer; font-size: 16px; flex-shrink: 0;
      transition: color var(--tr), background var(--tr);
    }
    #sidebar-toggle:hover { color: var(--accent); background: rgba(41,182,246,0.08); }
    #top-title { font-size: 15px; font-weight: 700; color: var(--text); letter-spacing: -0.2px; }
    #top-meta { font-size: 11px; color: var(--text2); }
    .top-spacer { flex: 1; }
    #about-btn {
      display: flex; align-items: center; gap: 6px; height: 32px; padding: 0 12px;
      border: 1px solid var(--border2); border-radius: 6px; background: transparent;
      color: var(--text2); cursor: pointer; font: inherit; font-size: 12px;
      transition: color var(--tr), border-color var(--tr), background var(--tr);
    }
    #about-btn:hover { color: var(--accent); border-color: var(--accent); background: rgba(41,182,246,0.06); }

    /* ── Map ── */
    #map {
      position: fixed; top: var(--top-h); bottom: var(--bottom-h);
      left: var(--left-w); right: var(--right-w); background: #0d1a27;
      transition: left var(--tr), right var(--tr);
    }
    .leaflet-image-layer { image-rendering: pixelated; image-rendering: crisp-edges; pointer-events: none; }
    .leaflet-control-attribution { font-size: 9px; background: rgba(11,24,37,0.75) !important; color: #8ca8be !important; }
    .leaflet-control-attribution a { color: var(--accent) !important; }
    .leaflet-control-scale-line { background: rgba(11,24,37,0.75); border-color: var(--border2); color: var(--text2); }

    /* ── POI sidebar ── */
    #poi-sidebar {
      position: fixed; top: var(--top-h); left: 0; bottom: var(--bottom-h);
      width: var(--left-w); z-index: 900;
      background: var(--panel); border-right: 1px solid var(--border2);
      display: flex; flex-direction: column; overflow: hidden;
      transition: transform var(--tr);
    }
    #poi-sidebar.collapsed { transform: translateX(calc(-1 * var(--left-w))); }
    .sidebar-head {
      padding: 11px 12px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
    }
    .sidebar-head-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text2); }
    #poi-count { font-size: 11px; color: var(--accent); font-weight: 600; }
    #poi-list { flex: 1; overflow-y: auto; }
    #poi-list::-webkit-scrollbar { width: 4px; }
    #poi-list::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
    .poi-item {
      display: flex; align-items: center; gap: 9px; padding: 9px 12px;
      cursor: pointer; border-bottom: 1px solid var(--border);
      transition: background var(--tr);
    }
    .poi-item:hover { background: rgba(41,182,246,0.07); }
    .poi-item.active { background: rgba(41,182,246,0.13); }
    .poi-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; background: var(--accent2); border: 1.5px solid rgba(0,200,150,0.4); }
    .poi-name { font-size: 12px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .poi-coords { font-size: 10px; color: var(--text2); margin-top: 1px; font-variant-numeric: tabular-nums; }

    /* ── Layer panel ── */
    #layer-panel {
      position: fixed; top: var(--top-h); right: 0; bottom: var(--bottom-h);
      width: var(--right-w); z-index: 900;
      background: var(--panel); border-left: 1px solid var(--border2);
      display: flex; flex-direction: column; overflow: hidden;
    }
    .panel-head {
      padding: 11px 12px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
    }
    .panel-head-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text2); }
    #layer-panel-scroll { flex: 1; overflow-y: auto; }
    #layer-panel-scroll::-webkit-scrollbar { width: 4px; }
    #layer-panel-scroll::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
    .layer-section { border-bottom: 1px solid var(--border); }
    .layer-section-title {
      padding: 8px 12px; font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.6px; color: var(--text2); cursor: pointer; user-select: none;
      display: flex; align-items: center; justify-content: space-between;
    }
    .layer-section-title:hover { color: var(--text); }
    .layer-section-title .chev { font-size: 10px; transition: transform var(--tr); }
    .layer-section-title.collapsed .chev { transform: rotate(-90deg); }
    .layer-row {
      display: flex; align-items: center; gap: 8px; padding: 7px 12px;
      border-top: 1px solid var(--border); transition: background var(--tr);
    }
    .layer-row:hover { background: var(--panel2); }
    .layer-check { display: flex; align-items: center; gap: 7px; flex: 1; min-width: 0; cursor: pointer; }
    .layer-check input[type=checkbox] { accent-color: var(--accent); width: 13px; height: 13px; flex-shrink: 0; cursor: pointer; }
    .layer-label { font-size: 12px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .op-btn {
      flex-shrink: 0; display: grid; place-items: center; width: 24px; height: 24px;
      border: 1px solid transparent; border-radius: 5px; background: transparent;
      color: var(--text2); cursor: pointer; font-size: 12px;
      transition: color var(--tr), border-color var(--tr);
    }
    .op-btn:hover { color: var(--accent); border-color: var(--border2); }
    .op-wrap { padding: 4px 12px 8px; display: none; }
    .op-wrap.open { display: block; }
    .op-wrap input[type=range] { width: 100%; accent-color: var(--accent); }
    .op-wrap select {
      width: 100%; margin-top: 5px; background: var(--panel2); color: var(--text);
      border: 1px solid var(--border2); border-radius: 4px; padding: 3px; font: inherit; font-size: 11px;
    }

    /* ── Coherence filter ── */
    #coh-filter-section { border-top: 1px solid var(--border2); padding: 12px; flex-shrink: 0; }
    .coh-title { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text2); margin-bottom: 10px; }
    .coh-toggle-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .coh-toggle-label { font-size: 12px; color: var(--text); }
    .toggle-sw { position: relative; display: inline-block; width: 34px; height: 18px; }
    .toggle-sw input { opacity: 0; width: 0; height: 0; }
    .toggle-track {
      position: absolute; cursor: pointer; inset: 0;
      background: var(--panel2); border: 1px solid var(--border2); border-radius: 18px;
      transition: background var(--tr);
    }
    .toggle-track::after {
      content: ""; position: absolute; left: 2px; top: 2px;
      width: 12px; height: 12px; border-radius: 50%;
      background: var(--text2); transition: left var(--tr), background var(--tr);
    }
    .toggle-sw input:checked + .toggle-track { background: rgba(41,182,246,0.2); border-color: var(--accent); }
    .toggle-sw input:checked + .toggle-track::after { left: 16px; background: var(--accent); }
    .coh-slider-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--text2); margin-bottom: 4px; }
    #coh-slider { width: 100%; accent-color: var(--accent); margin-bottom: 8px; }
    #coh-threshold-display { font-size: 22px; font-weight: 700; color: var(--accent); text-align: center; line-height: 1; margin-bottom: 4px; }
    #coh-pixel-count { font-size: 11px; color: var(--text2); text-align: center; min-height: 14px; }

    /* ── Legend stack ── */
    #legend-stack {
      position: fixed; left: calc(var(--left-w) + 10px); bottom: calc(var(--bottom-h) + 10px);
      z-index: 800; display: flex; flex-direction: column; gap: 6px; pointer-events: none;
      transition: left var(--tr);
    }
    .legend-card {
      padding: 8px 10px; border: 1px solid var(--border2); border-radius: var(--radius);
      background: rgba(15,32,51,0.94); backdrop-filter: blur(8px);
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .legend-title { font-size: 11px; font-weight: 700; color: var(--text); margin-bottom: 1px; }
    .legend-date { font-size: 10px; color: var(--text2); margin-bottom: 5px; }
    .legend-scale { display: grid; grid-template-columns: auto 1fr auto; gap: 6px; align-items: center; font-size: 10px; color: var(--text2); }
    .legend-gradient { height: 10px; border-radius: 2px; border: 1px solid rgba(255,255,255,0.1); }

    /* ── Bottom bar ── */
    #bottom-bar {
      position: fixed; bottom: 0; left: 0; right: 0; height: var(--bottom-h); z-index: 1000;
      background: rgba(11,24,37,0.97); border-top: 1px solid var(--border2);
      display: flex; align-items: center; padding: 0 16px; gap: 12px;
      backdrop-filter: blur(10px);
    }
    #velocity-btn {
      flex-shrink: 0; height: 28px; padding: 0 12px; border-radius: 6px;
      border: 1px solid var(--border2); background: transparent;
      color: var(--text2); cursor: pointer; font: inherit; font-size: 11px; font-weight: 600;
      transition: all var(--tr);
    }
    #velocity-btn.active { border-color: var(--accent2); color: var(--accent2); background: rgba(0,200,150,0.08); }
    #velocity-btn:not(.active):hover { border-color: var(--text2); color: var(--text); }
    .date-edge { font-size: 10px; color: var(--text2); flex-shrink: 0; font-variant-numeric: tabular-nums; }
    #date-slider-track { flex: 1; }
    #date-slider { width: 100%; accent-color: var(--accent); }
    #current-date-label { flex-shrink: 0; font-size: 11px; font-weight: 700; color: var(--accent); min-width: 82px; text-align: right; font-variant-numeric: tabular-nums; }

    /* ── Time series panel ── */
    #time-panel {
      position: fixed; top: var(--top-h); right: var(--right-w); bottom: var(--bottom-h);
      width: 460px; z-index: 950;
      background: rgba(10,22,34,0.99); border-left: 1px solid var(--border2);
      display: flex; flex-direction: column; overflow: hidden;
      backdrop-filter: blur(14px);
      transform: translateX(110%); transition: transform 0.22s ease;
    }
    #time-panel.open { transform: translateX(0); }
    #time-panel-head {
      display: flex; align-items: center; justify-content: space-between;
      padding: 11px 14px; border-bottom: 1px solid var(--border); flex-shrink: 0;
    }
    #time-panel-head h2 { font-size: 13px; font-weight: 700; }
    #close-time-panel {
      display: grid; place-items: center; width: 28px; height: 28px;
      border: 1px solid var(--border2); border-radius: 6px; background: transparent;
      color: var(--text2); cursor: pointer; font-size: 18px; line-height: 1;
      transition: color var(--tr), border-color var(--tr);
    }
    #close-time-panel:hover { color: var(--danger); border-color: var(--danger); }
    #pixel-summary { padding: 10px 14px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .sum-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px; }
    .sum-card { padding: 7px 9px; border: 1px solid var(--border); border-radius: 6px; background: var(--panel2); }
    .sum-label { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); }
    .sum-value { font-size: 13px; font-weight: 700; color: var(--text); margin-top: 2px; font-variant-numeric: tabular-nums; }
    .sum-value.danger { color: var(--danger); }
    .sum-value.warn { color: var(--warn); }
    .sum-value.ok { color: var(--accent2); }
    .gap-warn {
      display: flex; align-items: flex-start; gap: 7px; padding: 7px 9px;
      border: 1px solid rgba(255,183,77,0.3); border-radius: 6px;
      background: rgba(255,183,77,0.06); font-size: 11px; color: var(--warn); line-height: 1.4;
    }
    #ts-chart-wrap { flex: 1; overflow: hidden; padding: 4px 2px 2px; min-height: 0; }
    #ts-chart { width: 100%; height: 100%; }

    /* ── About modal ── */
    #about-backdrop {
      position: fixed; inset: 0; z-index: 1100;
      background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); display: none;
    }
    #about-backdrop.open { display: block; }
    #about-modal {
      position: fixed; top: 50%; left: 50%; z-index: 1101;
      transform: translate(-50%, -50%);
      width: min(640px, calc(100vw - 32px)); max-height: calc(100vh - 80px);
      background: var(--panel); border: 1px solid var(--border2); border-radius: var(--radius);
      box-shadow: 0 24px 80px rgba(0,0,0,0.7);
      display: none; flex-direction: column;
    }
    #about-modal.open { display: flex; }
    #about-modal-head {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0;
    }
    #about-modal-head h2 { font-size: 15px; font-weight: 700; }
    #close-about {
      display: grid; place-items: center; width: 28px; height: 28px;
      border: 1px solid var(--border2); border-radius: 6px; background: transparent;
      color: var(--text2); cursor: pointer; font-size: 18px;
      transition: color var(--tr), border-color var(--tr);
    }
    #close-about:hover { color: var(--danger); border-color: var(--danger); }
    #about-body { flex: 1; overflow-y: auto; padding: 16px; }
    #about-body::-webkit-scrollbar { width: 4px; }
    #about-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
    .a-section { margin-bottom: 18px; }
    .a-section h3 { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px; color: var(--text2); margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
    .a-dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; }
    .a-dt { font-size: 11px; color: var(--text2); white-space: nowrap; padding: 2px 0; }
    .a-dd { font-size: 11px; color: var(--text); padding: 2px 0; word-break: break-word; }
    .a-note { font-size: 11px; color: var(--text2); line-height: 1.5; padding: 6px 8px; background: var(--panel2); border-radius: 4px; border-left: 2px solid var(--border2); margin-bottom: 6px; }
    .a-note.warn-note { border-left-color: var(--warn); color: var(--warn); }
    .scene-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 7px; }
    .scene-chip { font-size: 10px; padding: 2px 7px; background: var(--panel2); border: 1px solid var(--border); border-radius: 4px; color: var(--text2); font-variant-numeric: tabular-nums; }

    /* ── Toast ── */
    #toast {
      position: fixed; left: 50%; bottom: calc(var(--bottom-h) + 14px);
      transform: translateX(-50%); z-index: 2000;
      padding: 8px 14px; border-radius: 6px;
      background: rgba(15,32,51,0.96); border: 1px solid var(--border2);
      color: var(--text); font-size: 12px; pointer-events: none;
      box-shadow: 0 6px 24px rgba(0,0,0,0.5);
      opacity: 0; transition: opacity 0.15s ease;
    }
    #toast.visible { opacity: 1; }
    .spin { display: inline-block; width: 12px; height: 12px; border: 2px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: sp 0.7s linear infinite; }
    @keyframes sp { to { transform: rotate(360deg); } }

    @media (max-width: 900px) {
      :root { --left-w: 0px; --right-w: 0px; }
      #poi-sidebar, #layer-panel { display: none; }
      #time-panel { right: 0; width: 100%; }
    }
  </style>
</head>
<body>

  <header id="top-bar">
    <button id="sidebar-toggle" title="Toggle POI sidebar">☰</button>
    <span id="top-title">InSAR SBAS Viewer</span>
    <span id="top-meta"></span>
    <div class="top-spacer"></div>
    <button id="about-btn">ℹ About dataset</button>
  </header>

  <aside id="poi-sidebar">
    <div class="sidebar-head">
      <span class="sidebar-head-label">Points of Interest</span>
      <span id="poi-count"></span>
    </div>
    <div id="poi-list"></div>
  </aside>

  <div id="map"></div>
  <div id="legend-stack"></div>

  <aside id="layer-panel">
    <div class="panel-head">
      <span class="panel-head-label">Layers</span>
    </div>
    <div id="layer-panel-scroll">
      <div class="layer-section">
        <div class="layer-section-title" id="sec-basemaps">Basemaps <span class="chev">▾</span></div>
        <div id="base-layer-list"></div>
      </div>
      <div class="layer-section">
        <div class="layer-section-title" id="sec-data">Data Overlays <span class="chev">▾</span></div>
        <div id="data-layer-list"></div>
      </div>
    </div>
    <div id="coh-filter-section">
      <div class="coh-title">Coherence Filter</div>
      <div class="coh-toggle-row">
        <span class="coh-toggle-label">Apply threshold filter</span>
        <label class="toggle-sw">
          <input type="checkbox" id="coh-filter-toggle" />
          <span class="toggle-track"></span>
        </label>
      </div>
      <div class="coh-slider-labels"><span>0.10</span><span>0.90</span></div>
      <input type="range" id="coh-slider" min="10" max="90" step="5" value="30" />
      <div id="coh-threshold-display">0.30</div>
      <div id="coh-pixel-count"><span class="spin"></span></div>
    </div>
  </aside>

  <div id="time-panel">
    <div id="time-panel-head">
      <h2>Pixel Time Series</h2>
      <button id="close-time-panel">&#x2715;</button>
    </div>
    <div id="pixel-summary"></div>
    <div id="ts-chart-wrap"><div id="ts-chart"></div></div>
  </div>

  <div id="about-backdrop"></div>
  <div id="about-modal" role="dialog" aria-modal="true">
    <div id="about-modal-head">
      <h2>About this dataset</h2>
      <button id="close-about">&#x2715;</button>
    </div>
    <div id="about-body"></div>
  </div>

  <footer id="bottom-bar">
    <button id="velocity-btn" class="active">Velocity</button>
    <span class="date-edge" id="date-start-lbl"></span>
    <div id="date-slider-track"><input type="range" id="date-slider" min="0" step="1" value="0" /></div>
    <span class="date-edge" id="date-end-lbl"></span>
    <span id="current-date-label">—</span>
  </footer>

  <div id="toast"></div>

<script>
"use strict";

// ── Constants ──────────────────────────────────────────────────────────────────
const SEG_COLORS = ["#2166ac","#e08214","#4dac26","#9e0142","#abd9e9"];
const PRIMARY_VEL_KEY = "sbas_velocity_masked";
const DISP_KEY = "sbas_displacement_masked";
// These layers are rendered by DataCanvas (vectorized circles), not PNG imageOverlays
const CANVAS_KEYS = new Set(["sbas_velocity_masked", "sbas_velocity_raw", "sbas_displacement_masked"]);

// ── Colormap (RdBu_r — matches matplotlib) ─────────────────────────────────────
const RDBU_R = [
  [0.000, [5,   48,  97 ]],
  [0.125, [33,  102, 172]],
  [0.250, [67,  147, 195]],
  [0.375, [146, 197, 222]],
  [0.500, [247, 247, 247]],
  [0.625, [244, 165, 130]],
  [0.750, [214, 96,  77 ]],
  [0.875, [178, 24,  43 ]],
  [1.000, [103, 0,   31 ]],
];
function _lerpStops(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return `rgb(${Math.round(c0[0]+f*(c1[0]-c0[0]))},${Math.round(c0[1]+f*(c1[1]-c0[1]))},${Math.round(c0[2]+f*(c1[2]-c0[2]))})`;
    }
  }
  return `rgb(103,0,31)`;
}
function valueToColor(v, vmin, vmax) {
  if (v === null || !isFinite(v)) return null;
  // TwoSlopeNorm equivalent: [vmin,0,vmax] → [0,0.5,1]
  const t = v <= 0 ? 0.5 * (v - vmin) / (0 - vmin) : 0.5 + 0.5 * v / vmax;
  return _lerpStops(RDBU_R, t);
}

// ── DataCanvas — vectorized circle layer ───────────────────────────────────────
class DataCanvas {
  constructor(map) {
    this._map = map;
    this._el = document.createElement("canvas");
    this._el.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;";
    map.getPanes().dataPane.appendChild(this._el);
    this._ctx = this._el.getContext("2d");
    this._pts = null;    // points data from server
    this._mode = null;   // "vel_masked"|"vel_raw"|"vel_coh"|"disp"
    this._thr  = 0.30;
    this._didx = 0;
    this._opa  = 0.87;
    this._vmin = -10; this._vmax = 10;
    this._cellLat = 0.001; this._cellLon = 0.001;
    this._rafPending = false;

    // `move` fires continuously during pan so circles stay locked to geography.
    // We throttle through rAF so we don't draw faster than the display refreshes.
    const sched = () => {
      if (this._rafPending) return;
      this._rafPending = true;
      requestAnimationFrame(() => { this._rafPending = false; this._draw(); });
    };
    map.on("move zoom zoomend resize", sched);
  }

  setData(pts) {
    this._pts = pts;
    this._cellLat = pts.cellLat;
    this._cellLon = pts.cellLon;
    if (this._mode) this._recompute();
  }

  render(mode, thr, didx, opa) {
    this._mode = mode; this._thr = thr; this._didx = didx; this._opa = opa;
    this._recompute();
  }

  clear() { this._mode = null; this._clear(); }

  // Returns [vmin, vmax] based on 2–98th percentile of currently visible points
  getRange() { return [this._vmin, this._vmax]; }

  // Returns {count, total, pct} of visible vs all valid points
  getCount() {
    const vals = this._vals(), coh = this._pts?.coherence, thr = this._thr;
    if (!vals) return {count: 0, total: 0, pct: 0};
    let count = 0, total = 0;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v === null || !isFinite(v)) continue;
      total++;
      if (this._mode === "vel_coh" && coh && (coh[i] === null || coh[i] < thr)) continue;
      count++;
    }
    return {count, total, pct: total ? Math.round(count / total * 100) : 0};
  }

  _vals() {
    const d = this._pts;
    if (!d) return null;
    if (this._mode === "vel_masked") return d.vel_masked;
    if (this._mode === "vel_raw")    return d.vel_raw;
    if (this._mode === "vel_coh")    return d.vel_raw;
    if (this._mode === "disp")       return d.disp ? d.disp[this._didx] : null;
    return null;
  }

  _recompute() {
    const vals = this._vals(), coh = this._pts?.coherence, thr = this._thr;
    if (!vals) { this._clear(); return; }
    const visible = [];
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v === null || !isFinite(v)) continue;
      if (this._mode === "vel_coh" && coh && (coh[i] === null || coh[i] < thr)) continue;
      visible.push(v);
    }
    if (visible.length) {
      visible.sort((a, b) => a - b);
      const p2  = visible[Math.max(0, Math.floor(visible.length * 0.02))];
      const p98 = visible[Math.min(visible.length - 1, Math.floor(visible.length * 0.98))];
      const vr  = Math.max(Math.abs(p2), Math.abs(p98)) || 10;
      this._vmin = -vr; this._vmax = vr;
    }
    this._draw();
  }

  _clear() {
    const sz = this._map.getSize();
    this._el.width = sz.x; this._el.height = sz.y;
    this._ctx.clearRect(0, 0, sz.x, sz.y);
  }

  _draw() {
    const sz = this._map.getSize();
    this._el.width = sz.x; this._el.height = sz.y;
    const ctx = this._ctx;
    ctx.clearRect(0, 0, sz.x, sz.y);
    if (!this._mode || !this._pts) return;

    const d = this._pts, vals = this._vals();
    if (!vals) return;
    const coh = d.coherence, thr = this._thr, doCoh = this._mode === "vel_coh";

    // Pane offset: dataPane is CSS-translated during pan; subtract so canvas-local
    // coords stay locked to geography regardless of how far the user has panned.
    const offset = this._map._getMapPanePos();

    // Radius: convert one cell worth of degrees to screen pixels, take 48%
    const ctr = this._map.getCenter();
    const p0 = this._map.latLngToContainerPoint(L.latLng(ctr.lat, ctr.lng));
    const p1 = this._map.latLngToContainerPoint(L.latLng(ctr.lat + this._cellLat, ctr.lng + this._cellLon));
    const cellPxH = Math.abs(p1.y - p0.y);
    const cellPxW = Math.abs(p1.x - p0.x);
    const r = Math.max(1.5, Math.min(cellPxH, cellPxW) * 0.48);

    ctx.globalAlpha = this._opa;
    const vmin = this._vmin, vmax = this._vmax;

    for (let i = 0; i < d.lats.length; i++) {
      const v = vals[i];
      if (v === null || !isFinite(v)) continue;
      if (doCoh && (!coh || coh[i] === null || coh[i] < thr)) continue;
      const color = valueToColor(v, vmin, vmax);
      if (!color) continue;
      const pt = this._map.latLngToContainerPoint(L.latLng(d.lats[i], d.lons[i]));
      ctx.beginPath();
      ctx.arc(pt.x - offset.x, pt.y - offset.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

// ── State ──────────────────────────────────────────────────────────────────────
const S = {
  map: null, payload: null,
  dataCanvas: null, pointsData: null,
  baseLayers: {}, dataLayers: {},
  aoiLayer: null, selectionLayer: null,
  layerEnabled: {}, layerOpacity: {}, layerDateIndex: {},
  mode: "velocity",
  dateIndex: 0,
  cohThreshold: 0.30,
  cohFilterEnabled: false,
  activePoi: null,
  toastTimer: null,
};

// ── Utilities ──────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function fmt(v, unit) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v), s = a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  return unit ? `${s} ${unit}` : s;
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function showToast(msg, ms = 2000) {
  const el = document.getElementById("toast");
  el.textContent = msg; el.classList.add("visible");
  clearTimeout(S.toastTimer); S.toastTimer = setTimeout(() => el.classList.remove("visible"), ms);
}

// ── Map ────────────────────────────────────────────────────────────────────────
function initMap(payload) {
  S.map = L.map("map", { center: payload.center, zoom: payload.defaultZoom, zoomControl: true, preferCanvas: true });
  S.map.createPane("dataPane"); S.map.getPane("dataPane").style.zIndex = 420;
  S.map.createPane("aoiPane");  S.map.getPane("aoiPane").style.zIndex = 440;
  S.map.createPane("selPane");  S.map.getPane("selPane").style.zIndex = 460;
  L.control.scale({ imperial: false }).addTo(S.map);
  S.map.fitBounds(payload.bounds, { padding: [20, 20] });

  const stopProp = el => { L.DomEvent.disableClickPropagation(el); L.DomEvent.disableScrollPropagation(el); };
  ["poi-sidebar","layer-panel","time-panel","bottom-bar","top-bar"].forEach(id => {
    const el = document.getElementById(id); if (el) stopProp(el);
  });
  document.getElementById("map").addEventListener("click", onMapClick);
  S.dataCanvas = new DataCanvas(S.map);
}

function onMapClick(e) {
  if (e.target.closest("#poi-sidebar,#layer-panel,#time-panel,#top-bar,#bottom-bar,#about-modal")) return;
  const rect = document.getElementById("map").getBoundingClientRect();
  const pt = L.point(e.clientX - rect.left, e.clientY - rect.top);
  fetchPixel(S.map.containerPointToLatLng(pt));
}

// ── Layer state init ───────────────────────────────────────────────────────────
function initLayerState(payload) {
  for (const l of [...payload.baseLayers, ...payload.dataLayers]) {
    S.layerEnabled[l.key] = l.defaultEnabled;
    S.layerOpacity[l.key] = l.defaultOpacity;
    S.layerDateIndex[l.key] = payload.defaultDateIndex;
  }
  S.dateIndex = payload.defaultDateIndex;
}

// ── Layer sync ─────────────────────────────────────────────────────────────────
function syncBase(layer) {
  if (!S.baseLayers[layer.key]) {
    S.baseLayers[layer.key] = L.tileLayer(layer.url, {
      maxZoom: layer.maxZoom, maxNativeZoom: layer.maxZoom,
      opacity: S.layerOpacity[layer.key], attribution: layer.attribution,
    });
  }
  S.baseLayers[layer.key].setOpacity(S.layerOpacity[layer.key]);
  const on = S.layerEnabled[layer.key];
  if (on && !S.map.hasLayer(S.baseLayers[layer.key])) S.baseLayers[layer.key].addTo(S.map);
  else if (!on && S.map.hasLayer(S.baseLayers[layer.key])) S.map.removeLayer(S.baseLayers[layer.key]);
}

function syncData(layer) {
  if (layer.kind === "aoi") { syncAoi(layer); return; }
  if (CANVAS_KEYS.has(layer.key)) { updateCanvasMode(); return; }
  const idx = layer.temporal ? S.layerDateIndex[layer.key] : 0;
  const url = `/overlay/${encodeURIComponent(layer.key)}/${idx}.png`;
  if (!S.dataLayers[layer.key]) {
    S.dataLayers[layer.key] = L.imageOverlay(url, S.payload.bounds, { opacity: S.layerOpacity[layer.key], pane: "dataPane", interactive: false });
  } else {
    S.dataLayers[layer.key].setUrl(url);
  }
  S.dataLayers[layer.key].setOpacity(S.layerOpacity[layer.key]);
  const on = S.layerEnabled[layer.key];
  if (on && !S.map.hasLayer(S.dataLayers[layer.key])) S.dataLayers[layer.key].addTo(S.map);
  else if (!on && S.map.hasLayer(S.dataLayers[layer.key])) S.map.removeLayer(S.dataLayers[layer.key]);
  renderLegends();
}

function syncAoi(layer) {
  if (!S.payload.aoi) return;
  if (!S.aoiLayer) {
    S.aoiLayer = L.polygon(S.payload.aoi, {
      pane: "aoiPane", color: "#29b6f6", weight: 2, opacity: 0.8,
      fillColor: "#29b6f6", fillOpacity: 0.04, interactive: false,
    });
  }
  const on = S.layerEnabled[layer.key];
  if (on && !S.map.hasLayer(S.aoiLayer)) S.aoiLayer.addTo(S.map);
  else if (!on && S.map.hasLayer(S.aoiLayer)) S.map.removeLayer(S.aoiLayer);
}

// ── Canvas mode arbiter ────────────────────────────────────────────────────────
function updateCanvasMode() {
  if (!S.dataCanvas || !S.pointsData) return;
  const dc = S.dataCanvas;

  if (S.mode === "date") {
    dc.render("disp", S.cohThreshold, S.dateIndex, S.layerOpacity[DISP_KEY] ?? 0.86);
  } else if (S.cohFilterEnabled) {
    dc.render("vel_coh", S.cohThreshold, 0, 0.87);
  } else if (S.layerEnabled[PRIMARY_VEL_KEY]) {
    dc.render("vel_masked", S.cohThreshold, 0, S.layerOpacity[PRIMARY_VEL_KEY] ?? 0.87);
  } else if (S.layerEnabled["sbas_velocity_raw"]) {
    dc.render("vel_raw", S.cohThreshold, 0, S.layerOpacity["sbas_velocity_raw"] ?? 0.82);
  } else {
    dc.clear();
  }
  updateCohCount();
  renderLegends();
}

// ── Layer UI ───────────────────────────────────────────────────────────────────
function buildLayerRow(layer, group) {
  const frag = document.createDocumentFragment();

  const row = document.createElement("div");
  row.className = "layer-row"; row.dataset.key = layer.key;

  const lbl = document.createElement("label"); lbl.className = "layer-check";
  const chk = document.createElement("input"); chk.type = "checkbox"; chk.checked = layer.defaultEnabled;
  chk.addEventListener("change", () => {
    S.layerEnabled[layer.key] = chk.checked;
    group === "base" ? syncBase(layer) : syncData(layer);
  });
  const span = document.createElement("span"); span.className = "layer-label"; span.textContent = layer.label;
  lbl.appendChild(chk); lbl.appendChild(span); row.appendChild(lbl);

  const opBtn = document.createElement("button"); opBtn.className = "op-btn"; opBtn.title = "Opacity"; opBtn.textContent = "◑";
  row.appendChild(opBtn);
  frag.appendChild(row);

  const opWrap = document.createElement("div"); opWrap.className = "op-wrap";
  const opSlider = document.createElement("input"); opSlider.type = "range"; opSlider.min = 0; opSlider.max = 100; opSlider.step = 5;
  opSlider.value = Math.round(layer.defaultOpacity * 100);
  opSlider.addEventListener("input", () => {
    S.layerOpacity[layer.key] = opSlider.value / 100;
    group === "base" ? syncBase(layer) : syncData(layer);
  });
  opWrap.appendChild(opSlider);

  if (layer.temporal && S.payload.dates.length) {
    const sel = document.createElement("select");
    S.payload.dates.forEach((d, i) => {
      const opt = document.createElement("option"); opt.value = i; opt.textContent = d;
      if (i === S.payload.defaultDateIndex) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      S.layerDateIndex[layer.key] = parseInt(sel.value);
      syncData(layer);
    });
    opWrap.appendChild(sel);
  }
  frag.appendChild(opWrap);

  opBtn.addEventListener("click", () => {
    const isOpen = opWrap.classList.toggle("open");
    if (isOpen) {
      document.querySelectorAll(".op-wrap.open").forEach(w => { if (w !== opWrap) w.classList.remove("open"); });
      opWrap.classList.add("open");
    }
  });

  return frag;
}

function renderLayerLists(payload) {
  document.getElementById("top-meta").textContent = `· ${payload.dataset}`;
  const baseList = document.getElementById("base-layer-list");
  const dataList = document.getElementById("data-layer-list");
  payload.baseLayers.forEach(l => baseList.appendChild(buildLayerRow(l, "base")));
  payload.dataLayers.forEach(l => dataList.appendChild(buildLayerRow(l, "data")));

  // Section collapse toggles
  document.querySelectorAll(".layer-section-title").forEach(title => {
    title.addEventListener("click", () => {
      const body = title.nextElementSibling;
      const collapsed = title.classList.toggle("collapsed");
      body.style.display = collapsed ? "none" : "";
    });
  });
}

// ── Legends ────────────────────────────────────────────────────────────────────
function renderLegends() {
  const stack = document.getElementById("legend-stack"); stack.innerHTML = "";
  for (const layer of S.payload.dataLayers) {
    if (!S.layerEnabled[layer.key] || layer.kind === "aoi" || !layer.legend) continue;
    const dateStr = layer.temporal ? (S.payload.dates[S.layerDateIndex[layer.key]] || "") : "";
    const card = document.createElement("div"); card.className = "legend-card";
    card.innerHTML = `
      <div class="legend-title">${esc(layer.label)}</div>
      ${dateStr ? `<div class="legend-date">${esc(dateStr)}</div>` : ""}
      <div class="legend-scale">
        <span>${esc(layer.legend.low)}</span>
        <span class="legend-gradient" style="background:linear-gradient(to right,${layer.legend.colors.join(",")})"></span>
        <span>${esc(layer.legend.high)}</span>
      </div>`;
    stack.appendChild(card);
  }
  if (S.cohFilterEnabled && S.dataCanvas) {
    const rng = S.dataCanvas.getRange();
    const card = document.createElement("div"); card.className = "legend-card";
    card.innerHTML = `
      <div class="legend-title">Velocity coh ≥ ${S.cohThreshold.toFixed(2)} (mm/yr)</div>
      <div class="legend-scale">
        <span>${fmt(rng[0])}</span>
        <span class="legend-gradient" style="background:linear-gradient(to right,#08306b,#f7f7f7,#67000d)"></span>
        <span>${fmt(rng[1])}</span>
      </div>`;
    stack.appendChild(card);
  }
}

// ── Points data loader ─────────────────────────────────────────────────────────
async function loadPoints() {
  try {
    const pts = await fetch("/api/points").then(r => r.json());
    S.pointsData = pts;
    S.dataCanvas.setData(pts);
    updateCanvasMode();
  } catch (err) {
    console.error("Failed to load points data:", err);
    document.getElementById("coh-pixel-count").textContent = "load error";
  }
}

// ── Coherence filter ───────────────────────────────────────────────────────────
function updateCohCount() {
  const el = document.getElementById("coh-pixel-count");
  if (!S.dataCanvas || !S.pointsData) { el.innerHTML = '<span class="spin"></span>'; return; }
  const { count, total, pct } = S.dataCanvas.getCount();
  el.textContent = `${count.toLocaleString()} px visible (${pct}% of total)`;
}

const debouncedCanvasUpdate = debounce(updateCanvasMode, 80);

function initCohFilter(payload) {
  S.cohThreshold = payload.cohThresholdDefault;
  const slider = document.getElementById("coh-slider");
  const display = document.getElementById("coh-threshold-display");
  slider.value = Math.round(S.cohThreshold * 100);
  display.textContent = S.cohThreshold.toFixed(2);

  slider.addEventListener("input", () => {
    S.cohThreshold = parseInt(slider.value) / 100;
    display.textContent = S.cohThreshold.toFixed(2);
    debouncedCanvasUpdate();
  });
  document.getElementById("coh-filter-toggle").addEventListener("change", e => {
    S.cohFilterEnabled = e.target.checked;
    updateCanvasMode();
  });
}

// ── Date slider ────────────────────────────────────────────────────────────────
function initDateSlider(payload) {
  const slider = document.getElementById("date-slider");
  const curLbl = document.getElementById("current-date-label");
  if (!payload.dates.length) { slider.disabled = true; return; }

  slider.max = payload.dates.length - 1;
  slider.value = payload.defaultDateIndex;
  document.getElementById("date-start-lbl").textContent = payload.dates[0];
  document.getElementById("date-end-lbl").textContent = payload.dates[payload.dates.length - 1];
  curLbl.textContent = payload.dates[payload.defaultDateIndex];

  slider.addEventListener("input", () => {
    const idx = parseInt(slider.value);
    curLbl.textContent = payload.dates[idx];
    enterDateMode(idx);
  });
  document.getElementById("velocity-btn").addEventListener("click", enterVelocityMode);
}

function enterDateMode(idx) {
  S.mode = "date"; S.dateIndex = idx;
  document.getElementById("velocity-btn").classList.remove("active");
  updateCanvasMode();
}

function enterVelocityMode() {
  S.mode = "velocity";
  document.getElementById("velocity-btn").classList.add("active");
  updateCanvasMode();
}

// ── POI sidebar ────────────────────────────────────────────────────────────────
function renderPoiList(pois) {
  const list = document.getElementById("poi-list");
  document.getElementById("poi-count").textContent = pois.length || "";
  list.innerHTML = "";
  if (!pois.length) {
    list.innerHTML = `<div style="padding:14px;font-size:11px;color:var(--text2)">No POIs in this dataset.</div>`;
    return;
  }
  pois.forEach(poi => {
    const item = document.createElement("div"); item.className = "poi-item";
    item.innerHTML = `
      <div class="poi-dot"></div>
      <div>
        <div class="poi-name">${esc(poi.name)}</div>
        <div class="poi-coords">${poi.lat.toFixed(5)}, ${poi.lon.toFixed(5)}</div>
      </div>`;
    item.addEventListener("click", () => {
      document.querySelectorAll(".poi-item.active").forEach(el => el.classList.remove("active"));
      item.classList.add("active");
      S.map.setView([poi.lat, poi.lon], Math.max(S.map.getZoom(), 14), { animate: true });
      fetchPixel({ lat: poi.lat, lng: poi.lon });
    });
    list.appendChild(item);
  });
}

// ── Pixel fetch ────────────────────────────────────────────────────────────────
async function fetchPixel(latlng) {
  try {
    const r = await fetch(`/api/pixel?lat=${latlng.lat}&lon=${latlng.lng}`);
    const info = await r.json();
    if (!info.found) { showToast(`No data here (${info.reason || "no data"})`); return; }
    // Ring highlight centred on the exact pixel the data came from
    if (S.selectionLayer) S.map.removeLayer(S.selectionLayer);
    const [[lat0, lon0], [lat1, lon1]] = info.cellBounds;
    const cellLatM = Math.abs(lat1 - lat0) * 111320;
    const cellLonM = Math.abs(lon1 - lon0) * 111320 * Math.cos(info.lat * Math.PI / 180);
    const ringRadiusM = Math.min(cellLatM, cellLonM) * 0.65;
    S.selectionLayer = L.circle([info.lat, info.lon], {
      pane: "selPane", radius: ringRadiusM,
      color: "#ffd740", weight: 2.5, opacity: 1,
      fill: false, interactive: false,
    }).addTo(S.map);
    showTimePanel(info);
  } catch (err) { showToast("Pixel lookup failed"); console.error(err); }
}

// ── Time series panel ──────────────────────────────────────────────────────────
function buildSummaryHtml(info) {
  const vel = info.velocity_mm_yr;
  const velStr = vel !== null ? fmt(vel, "mm/yr") : "masked";
  const velClass = info.below_static_mask ? "danger" : (Math.abs(vel || 0) > 5 ? "warn" : "ok");
  const coh = info.coherence_median;
  const cohStr = coh !== null ? coh.toFixed(3) : "—";
  const cohClass = coh !== null && coh < 0.35 ? "danger" : coh !== null && coh < 0.5 ? "warn" : "ok";

  let warnHtml = "";
  if (info.has_gap) {
    warnHtml = `<div class="gap-warn">⚠&nbsp; Time series has a coherence gap — segments are shown separately and re-zeroed. Do not compare absolute values across segments.</div>`;
  } else if (info.valid_epoch_count < 3) {
    warnHtml = `<div class="gap-warn">⚠&nbsp; Only ${info.valid_epoch_count} valid epoch(s) at this pixel — data quality is low.</div>`;
  } else if (info.below_static_mask) {
    warnHtml = `<div class="gap-warn">⚠&nbsp; Below the static coherence mask — velocity value is unreliable.</div>`;
  }

  return `
    <div class="sum-grid">
      <div class="sum-card"><div class="sum-label">Latitude</div><div class="sum-value">${info.lat.toFixed(6)}</div></div>
      <div class="sum-card"><div class="sum-label">Longitude</div><div class="sum-value">${info.lon.toFixed(6)}</div></div>
      <div class="sum-card"><div class="sum-label">Velocity (LOS)</div><div class="sum-value ${velClass}">${esc(velStr)}</div></div>
      <div class="sum-card"><div class="sum-label">Coherence (median)</div><div class="sum-value ${cohClass}">${cohStr}</div></div>
      <div class="sum-card"><div class="sum-label">Valid epochs</div><div class="sum-value">${info.valid_epoch_count} / ${info.total_epoch_count}</div></div>
      <div class="sum-card"><div class="sum-label">Segments</div><div class="sum-value">${info.segment_count || "—"}</div></div>
    </div>${warnHtml}`;
}

function buildPlotlyData(info) {
  const { dates, series } = info;
  const { raw, segmented, segment_id, valid_time_mask, coh_per_date } = series;
  const traces = [];

  const finRaw = raw.filter(v => v !== null);
  const yMin = finRaw.length ? Math.min(...finRaw) : -1;
  const yMax = finRaw.length ? Math.max(...finRaw) : 1;
  const yRange = Math.max(yMax - yMin, 1);
  const markerY = yMin - yRange * 0.18;

  // Raw (grey dotted)
  traces.push({
    x: dates, y: raw, mode: "lines+markers", connectgaps: false,
    line: { color: "rgba(140,175,200,0.5)", width: 1.5, dash: "dot" },
    marker: { color: "rgba(140,175,200,0.6)", size: 4 },
    name: "Raw (all epochs)", yaxis: "y",
    hovertemplate: "%{x}: %{y:.2f} mm<extra>Raw</extra>",
  });

  // Segmented — one trace per unique segment
  const uniqueSegs = [...new Set(segment_id.filter(s => s > 0))].sort((a,b) => a-b);
  for (const segId of uniqueSegs) {
    const color = SEG_COLORS[(segId - 1) % SEG_COLORS.length];
    const segY = segmented.map((v, i) => segment_id[i] === segId ? v : null);
    traces.push({
      x: dates, y: segY, mode: "lines+markers", connectgaps: false,
      line: { color, width: 2.5 }, marker: { color, size: 7, symbol: "circle" },
      name: uniqueSegs.length > 1 ? `Segment ${segId}` : "Segmented",
      yaxis: "y",
      hovertemplate: `%{x}: %{y:.2f} mm<extra>Segment ${segId}</extra>`,
    });
  }

  // Dropped dates — hollow red X at bottom of main plot
  const droppedDates = dates.filter((_, i) => valid_time_mask[i] === 0 && raw[i] !== null);
  if (droppedDates.length) {
    traces.push({
      x: droppedDates, y: droppedDates.map(() => markerY),
      mode: "markers",
      marker: { symbol: "x-open", color: "#ef5350", size: 9, line: { width: 2 } },
      name: "Excluded (low coherence at epoch)", yaxis: "y",
      hovertemplate: "%{x}: excluded — low coherence at this epoch<extra></extra>",
    });
  }

  // Coherence sparkline (y2)
  if (coh_per_date.some(v => v !== null)) {
    const barColors = coh_per_date.map(v =>
      v === null ? "rgba(100,140,170,0.3)" :
      v >= S.cohThreshold ? "rgba(41,182,246,0.6)" : "rgba(239,83,80,0.5)"
    );
    traces.push({
      x: dates, y: coh_per_date, type: "bar",
      marker: { color: barColors }, name: "Coherence at epoch",
      yaxis: "y2", xaxis: "x2",
      hovertemplate: "%{x}: %{y:.3f}<extra>Coherence</extra>",
    });
    // Threshold line
    traces.push({
      x: [dates[0], dates[dates.length - 1]], y: [S.cohThreshold, S.cohThreshold],
      mode: "lines", line: { color: "#ef5350", dash: "dash", width: 1 },
      yaxis: "y2", xaxis: "x2", showlegend: false, hoverinfo: "skip",
      name: `Coh. threshold (${S.cohThreshold.toFixed(2)})`,
    });
  }

  const layout = {
    xaxis: { type: "date", showgrid: true, gridcolor: "rgba(255,255,255,0.05)", tickfont: { size: 10, color: "#6a91ae" }, showticklabels: false },
    xaxis2: { type: "date", matches: "x", showgrid: true, gridcolor: "rgba(255,255,255,0.05)", tickfont: { size: 10, color: "#6a91ae" } },
    yaxis: {
      domain: [0.35, 1.0],
      title: { text: "LOS displacement (mm)", font: { size: 10, color: "#6a91ae" } },
      zeroline: true, zerolinecolor: "rgba(255,255,255,0.15)", zerolinewidth: 1,
      gridcolor: "rgba(255,255,255,0.05)", tickfont: { size: 10, color: "#6a91ae" },
      range: [markerY - yRange * 0.05, yMax + yRange * 0.1],
    },
    yaxis2: {
      domain: [0, 0.28],
      title: { text: "Coherence", font: { size: 10, color: "#6a91ae" } },
      range: [0, 1], gridcolor: "rgba(255,255,255,0.05)", tickfont: { size: 10, color: "#6a91ae" },
    },
    margin: { l: 50, r: 10, t: 8, b: 28 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(15,32,51,0.45)",
    showlegend: true,
    legend: { x: 0, y: 1.03, xanchor: "left", yanchor: "bottom", orientation: "h", font: { size: 9, color: "#6a91ae" }, bgcolor: "rgba(0,0,0,0)" },
    font: { family: "Inter,Segoe UI,system-ui,sans-serif" },
  };

  return { traces, layout };
}

function showTimePanel(info) {
  document.getElementById("pixel-summary").innerHTML = buildSummaryHtml(info);

  const chartDiv = document.getElementById("ts-chart");
  chartDiv.innerHTML = "";
  const wrapH = document.getElementById("ts-chart-wrap").clientHeight - 8;
  const { traces, layout } = buildPlotlyData(info);
  layout.height = Math.max(300, wrapH);

  Plotly.newPlot(chartDiv, traces, layout, { responsive: true, displayModeBar: false, scrollZoom: false });
  document.getElementById("time-panel").classList.add("open");
}

document.getElementById("close-time-panel").addEventListener("click", () => {
  document.getElementById("time-panel").classList.remove("open");
});

// ── About modal ────────────────────────────────────────────────────────────────
async function loadAbout() {
  const body = document.getElementById("about-body");
  body.innerHTML = `<div style="text-align:center;padding:36px"><span class="spin"></span></div>`;
  openAbout();
  try {
    const data = await fetch("/api/metadata").then(r => r.json());
    body.innerHTML = buildAboutHtml(data);
  } catch { body.innerHTML = `<p style="color:var(--danger);padding:16px">Failed to load metadata.</p>`; }
}

function buildAboutHtml(data) {
  const m = data.metadata || {}, p = data.parameters || {};
  const scenes = m.scenes || p.scenes || {}, tw = m.time_window || p.time_window || {};
  const coh = m.coherence_masking || p.coherence_masking || {};
  const proc = m.processing || p.processing || {}, units = m.units || {}, notes = m.notes || {};
  const pairs = m.pairs || p.pairs || {};

  const dl = rows => `<div class="a-dl">${rows.map(([k,v]) =>
    `<span class="a-dt">${esc(k)}</span><span class="a-dd">${esc(String(v ?? "—"))}</span>`
  ).join("")}</div>`;

  const chips = (scenes.dates || []).map(d => `<span class="scene-chip">${esc(d)}</span>`).join("");
  const note = (text, warn=false) => text ? `<div class="a-note ${warn?"warn-note":""}">${esc(text)}</div>` : "";

  return `
    <div class="a-section"><h3>Dataset</h3>${dl([
      ["File", data.dataset],
      ["Project", `${m.project||p.project||"—"} / ${m.orbit||p.orbit||"—"}`],
      ["Processing date", (m.processing_date_utc||p.processing_date||"").slice(0,10)],
      ["PyGMTSAR", m.pygmtsar_version||"—"],
    ])}</div>
    <div class="a-section"><h3>Time Window</h3>${dl([
      ["Start", tw.start||"—"], ["End", tw.end||"—"],
      ["Duration", tw.days?`${tw.days} days`:"—"],
      ["Scenes", scenes.count?`${scenes.count} acquisitions`:"—"],
      ["Pairs used", pairs.best_used!=null?`${pairs.best_used} of ${pairs.initial}`:"—"],
    ])}${chips?`<div class="scene-chips">${chips}</div>`:""}</div>
    <div class="a-section"><h3>Processing</h3>${dl([
      ["Coherence threshold", coh.coh_threshold!=null?`${coh.coh_threshold} (${coh.coh_mask_method||"median"})`:"—"],
      ["Dynamic coh. mask", coh.dynamic_coh_mask_actually_used?"Enabled":"Disabled"],
      ["Segmented output", coh.segmented_actually_created?"Created":"Not created"],
      ["Geocode resolution", proc.geocode_res_m?`${proc.geocode_res_m} m`:"—"],
      ["SBAS wavelength", proc.sbas_wavelength_m?`${proc.sbas_wavelength_m} m`:"—"],
    ])}</div>
    <div class="a-section"><h3>Units</h3>${dl(Object.entries(units).map(([k,v])=>[k,v]))}</div>
    <div class="a-section"><h3>Technical Notes</h3>
      ${note(notes.segmented_warning, true)}
      ${note(notes.displacement_reference)}
      ${note(notes.dynamic_coherence_approximation)}
      ${note(notes.mask_interpretation)}
    </div>`;
}

function openAbout() {
  document.getElementById("about-backdrop").classList.add("open");
  document.getElementById("about-modal").classList.add("open");
}
function closeAbout() {
  document.getElementById("about-backdrop").classList.remove("open");
  document.getElementById("about-modal").classList.remove("open");
}

document.getElementById("about-btn").addEventListener("click", loadAbout);
document.getElementById("close-about").addEventListener("click", closeAbout);
document.getElementById("about-backdrop").addEventListener("click", closeAbout);
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { closeAbout(); document.getElementById("time-panel").classList.remove("open"); }
});

// ── Sidebar toggle ─────────────────────────────────────────────────────────────
document.getElementById("sidebar-toggle").addEventListener("click", () => {
  const sidebar = document.getElementById("poi-sidebar");
  const mapEl = document.getElementById("map");
  const legends = document.getElementById("legend-stack");
  const collapsed = sidebar.classList.toggle("collapsed");
  mapEl.style.left = collapsed ? "0" : "var(--left-w)";
  legends.style.left = collapsed ? "10px" : "calc(var(--left-w) + 10px)";
  setTimeout(() => S.map && S.map.invalidateSize(), 200);
});

// ── Boot ───────────────────────────────────────────────────────────────────────
async function boot() {
  try {
    const payload = await fetch("/api/viewer").then(r => r.json());
    S.payload = payload;

    initLayerState(payload);
    initMap(payload);
    renderLayerLists(payload);
    payload.baseLayers.forEach(l => syncBase(l));
    payload.dataLayers.forEach(l => syncData(l));
    renderLegends();
    initCohFilter(payload);
    initDateSlider(payload);
    renderPoiList(payload.pois);
    loadPoints();   // background: pixel data for canvas rendering
  } catch (err) {
    console.error("Boot failed:", err);
    document.body.innerHTML = `<div style="padding:40px;color:#ef5350;font-family:monospace">
      <h2>Viewer failed to start</h2><pre>${esc(String(err))}</pre>
      <p>Check the terminal for more details.</p>
    </div>`;
  }
}

boot();
</script>
</body>
</html>
"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    project_paths = resolve_project_paths(args.project_dir)
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else default_dataset_path(project_paths)
    )

    VIEWER_DATA = load_viewer_data(dataset_path, project_paths)

    print(f"Project : {project_paths.root_dir}")
    print(f"Product : {project_paths.product_dir}")
    print(f"Dataset : {dataset_path}")
    print(f"Dates   : {len(VIEWER_DATA.dates)} scenes  ({', '.join(d.strftime('%Y-%m-%d') for d in VIEWER_DATA.dates)})")
    print(f"Grid    : {len(VIEWER_DATA.latitudes)}×{len(VIEWER_DATA.longitudes)}  ({int(VIEWER_DATA.spatial_mask.sum())} valid pixels)")
    print(f"POIs    : {len(VIEWER_DATA.pois)}")
    print(f"Open    : http://{args.host}:{args.port}")

    flask_app = create_app()
    flask_app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)
