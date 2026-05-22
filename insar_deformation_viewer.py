"""
Fast Leaflet viewer for SBAS maps and pixel deformation time series.

Run:
    python insar_deformation_viewer.py Data\\project_D_results_only

Then open:
    http://127.0.0.1:8050

The project folder can be either the outer export bundle or the inner
outputs/<project>_<orbit> product folder.

Dependency notes:
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

import numpy as np
import pandas as pd
import xarray as xr
from flask import Flask, Response, jsonify, request
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image

from insar_project import ProjectPaths, add_project_dir_argument, find_netcdf_files, resolve_project_paths


DEFAULT_ZOOM = 13
PRIMARY_DISPLACEMENT_LAYER = "sbas_displacement_masked"


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
    kind: str
    units: str
    colormap: str
    default_enabled: bool
    default_opacity: float
    temporal: bool = False
    symmetric: bool = False


@dataclass(frozen=True)
class ViewerData:
    dataset_path: Path
    parameters_path: Path
    title: str
    dates: pd.DatetimeIndex
    latitudes: np.ndarray
    longitudes: np.ndarray
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    spatial_mask: np.ndarray
    layer_specs: tuple[DataLayerSpec, ...]
    layer_ranges: dict[str, tuple[float, float]]
    static_values: dict[str, np.ndarray]
    temporal_values: dict[str, np.ndarray]
    displacement_labels: dict[str, str]
    aoi_lons: np.ndarray | None
    aoi_lats: np.ndarray | None
    center_lat: float
    center_lon: float


BASEMAP_SPECS: tuple[TileLayerSpec, ...] = (
    TileLayerSpec(
        key="esri_satellite",
        label="Esri Satellite",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Tiles (c) Esri",
        default_enabled=True,
        default_opacity=1.0,
        max_zoom=18,
    ),
    TileLayerSpec(
        key="openstreetmap",
        label="OpenStreetMap",
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="(c) OpenStreetMap contributors",
        default_enabled=False,
        default_opacity=0.85,
        max_zoom=19,
    ),
    TileLayerSpec(
        key="carto_light",
        label="Carto Light",
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attribution="(c) OpenStreetMap contributors (c) CARTO",
        default_enabled=False,
        default_opacity=0.8,
        max_zoom=20,
    ),
)

REQUESTED_DATA_LAYERS: tuple[DataLayerSpec, ...] = (
    DataLayerSpec(
        key="sbas_displacement_masked",
        label="SBAS displacement (masked) [mm]",
        variable="sbas_displacement_masked",
        kind="scalar",
        units="mm",
        colormap="RdBu_r",
        default_enabled=True,
        default_opacity=0.86,
        temporal=True,
        symmetric=True,
    ),
    DataLayerSpec(
        key="sbas_velocity_raw",
        label="SBAS velocity (raw) [mm/year]",
        variable="sbas_velocity_raw",
        kind="scalar",
        units="mm/year",
        colormap="RdBu_r",
        default_enabled=False,
        default_opacity=0.82,
        symmetric=True,
    ),
    DataLayerSpec(
        key="sbas_velocity_masked",
        label="SBAS velocity (masked) [mm/year]",
        variable="sbas_velocity_masked",
        kind="scalar",
        units="mm/year",
        colormap="RdBu_r",
        default_enabled=False,
        default_opacity=0.86,
        symmetric=True,
    ),
    DataLayerSpec(
        key="coherence_median",
        label="Coherence (median across pairs)",
        variable="coherence_median",
        kind="scalar",
        units="coherence",
        colormap="viridis",
        default_enabled=False,
        default_opacity=0.76,
    ),
    DataLayerSpec(
        key="valid_pixel_mask",
        label="valid_pixel_mask (coh >= 0.3)",
        variable="valid_pixel_mask",
        kind="mask",
        units="valid",
        colormap="mask",
        default_enabled=False,
        default_opacity=0.62,
    ),
    DataLayerSpec(
        key="aoi_original",
        label="AOI (original)",
        variable=None,
        kind="aoi",
        units="",
        colormap="",
        default_enabled=True,
        default_opacity=1.0,
    ),
)

DISPLACEMENT_VARIABLES = (
    ("sbas_displacement_masked", "SBAS masked displacement"),
    ("sbas_displacement_raw", "SBAS raw displacement"),
    ("sbas_displacement_segmented_same_pixel", "SBAS segmented displacement"),
    ("displacement_sbas", "SBAS displacement"),
)

SERIES_COLORS = {
    "sbas_displacement_masked": "#126a65",
    "sbas_displacement_raw": "#2f5f9f",
    "sbas_displacement_segmented_same_pixel": "#8b4ec7",
    "displacement_sbas": "#126a65",
}

VIEWER_DATA: ViewerData | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_dir_argument(parser)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="NetCDF dataset to view. Defaults to results_tight.nc, then AOI, then results_wide.nc.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viewer host.")
    parser.add_argument("--port", type=int, default=8050, help="Viewer port.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    return parser.parse_args()


def default_dataset_path(project_paths: ProjectPaths) -> Path:
    candidates = (
        project_paths.product_dir / "results_tight.nc",
        project_paths.aoi_output_dir / "results_aoi_masked.nc",
        project_paths.product_dir / "results_wide.nc",
    )
    for path in candidates:
        if path.exists():
            return path

    nc_files = find_netcdf_files(project_paths)
    if nc_files:
        return nc_files[0]

    raise FileNotFoundError(
        f"No NetCDF dataset found under product folder: {project_paths.product_dir}"
    )


def load_parameters(parameters_path: Path) -> dict:
    if not parameters_path.exists():
        return {}
    with parameters_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_polygon_wkt(wkt: str | None) -> list[tuple[float, float]]:
    if not wkt:
        return []

    match = re.match(r"^\s*POLYGON\s*\(\((.+)\)\)\s*$", wkt, flags=re.IGNORECASE)
    if not match:
        return []

    coordinates = []
    for point in match.group(1).split(","):
        parts = point.strip().split()
        if len(parts) < 2:
            return []
        coordinates.append((float(parts[0]), float(parts[1])))

    if coordinates and coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])

    return coordinates


def make_polygon_mask(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    coordinates: list[tuple[float, float]],
) -> np.ndarray | None:
    if len(coordinates) < 4:
        return None

    try:
        from matplotlib.path import Path as MatplotlibPath
    except ImportError:
        return None

    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    mask = MatplotlibPath(coordinates).contains_points(points, radius=1e-12)
    return mask.reshape((len(lat_values), len(lon_values)))


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    if values.size == 1:
        delta = 0.0001
        return np.array([values[0] - delta / 2, values[0] + delta / 2])

    midpoints = (values[:-1] + values[1:]) / 2
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def build_spatial_mask(dataset, latitudes: np.ndarray, longitudes: np.ndarray, aoi_coordinates) -> np.ndarray:
    if "aoi_mask" in dataset:
        return dataset["aoi_mask"].transpose("lat", "lon").values.astype(bool)

    if aoi_coordinates:
        polygon_mask = make_polygon_mask(longitudes, latitudes, aoi_coordinates)
        if polygon_mask is not None:
            return polygon_mask

    candidate_masks = []
    for spec in REQUESTED_DATA_LAYERS:
        if spec.variable and spec.variable in dataset:
            values = dataset[spec.variable]
            if "date" in values.dims:
                array = values.transpose("date", "lat", "lon").values
                candidate_masks.append(np.isfinite(array).any(axis=0))
            else:
                array = values.transpose("lat", "lon").values
                candidate_masks.append(np.isfinite(array))

    if not candidate_masks:
        raise ValueError("No spatial data layers were found in the dataset.")

    return np.logical_or.reduce(candidate_masks)


def robust_range(values: np.ndarray, mask: np.ndarray, symmetric: bool) -> tuple[float, float]:
    if values.ndim == 3:
        finite = values[:, mask]
    else:
        finite = values[mask]
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)

    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), 98))
        if not math.isfinite(limit) or limit == 0:
            limit = 1.0
        return -limit, limit

    low, high = np.nanpercentile(finite, [2, 98])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or low == high:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def load_viewer_data(dataset_path: Path, parameters_path: Path) -> ViewerData:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")

    parameters = load_parameters(parameters_path)
    aoi_coordinates = parse_polygon_wkt((parameters.get("aoi") or {}).get("raw_wkt"))
    aoi_lons = np.array([lon for lon, _lat in aoi_coordinates]) if aoi_coordinates else None
    aoi_lats = np.array([lat for _lon, lat in aoi_coordinates]) if aoi_coordinates else None

    with xr.open_dataset(dataset_path) as dataset:
        dataset = dataset.sortby("lat").sortby("lon").load()

    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError("Dataset must contain lat and lon coordinates.")

    latitudes = dataset["lat"].values.astype("float64")
    longitudes = dataset["lon"].values.astype("float64")
    spatial_mask = build_spatial_mask(dataset, latitudes, longitudes, aoi_coordinates)

    if not np.any(spatial_mask):
        raise ValueError("No pixels are available inside the selected spatial mask.")

    layer_specs = tuple(
        spec
        for spec in REQUESTED_DATA_LAYERS
        if spec.kind == "aoi" or (spec.variable is not None and spec.variable in dataset.data_vars)
    )

    static_values: dict[str, np.ndarray] = {}
    temporal_values: dict[str, np.ndarray] = {}
    layer_ranges: dict[str, tuple[float, float]] = {}

    for spec in layer_specs:
        if spec.variable is None:
            continue
        if spec.temporal:
            values = dataset[spec.variable].transpose("date", "lat", "lon").values.astype("float64")
            temporal_values[spec.key] = values
        else:
            values = dataset[spec.variable].transpose("lat", "lon").values.astype("float64")
            static_values[spec.key] = values

        layer_ranges[spec.key] = (0.0, 1.0) if spec.kind == "mask" else robust_range(values, spatial_mask, spec.symmetric)

    if "date" in dataset.coords:
        dates = pd.DatetimeIndex(pd.to_datetime(dataset["date"].values))
    else:
        dates = pd.DatetimeIndex([])

    displacement_labels: dict[str, str] = {}
    for variable, label in DISPLACEMENT_VARIABLES:
        if variable not in dataset.data_vars or "date" not in dataset[variable].dims:
            continue
        if variable not in temporal_values:
            temporal_values[variable] = dataset[variable].transpose("date", "lat", "lon").values.astype("float64")
        displacement_labels[variable] = label

    mask_rows, mask_cols = np.where(spatial_mask)
    pixel_lats = latitudes[mask_rows]
    pixel_lons = longitudes[mask_cols]

    title = str(dataset.attrs.get("title") or dataset_path.stem)
    return ViewerData(
        dataset_path=dataset_path,
        parameters_path=parameters_path,
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
        displacement_labels=displacement_labels,
        aoi_lons=aoi_lons,
        aoi_lats=aoi_lats,
        center_lat=float(np.nanmean(pixel_lats)),
        center_lon=float(np.nanmean(pixel_lons)),
    )


def require_viewer_data() -> ViewerData:
    if VIEWER_DATA is None:
        raise RuntimeError("Viewer data has not been loaded.")
    return VIEWER_DATA


def spec_by_key(key: str) -> DataLayerSpec:
    data = require_viewer_data()
    for spec in data.layer_specs:
        if spec.key == key:
            return spec
    raise KeyError(key)


def clamp_date_index(date_index: int | None) -> int:
    data = require_viewer_data()
    if not len(data.dates):
        return 0
    if date_index is None:
        return len(data.dates) - 1
    return max(0, min(len(data.dates) - 1, int(date_index)))


def values_for_layer(key: str, date_index: int | None = None) -> np.ndarray:
    data = require_viewer_data()
    spec = spec_by_key(key)
    if spec.temporal:
        return data.temporal_values[key][clamp_date_index(date_index)]
    return data.static_values[key]


def format_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def rgba_to_uint8(colors: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rgba = np.clip(colors * 255, 0, 255).astype("uint8")
    rgba[..., 3] = np.clip(alpha * 255, 0, 255).astype("uint8")
    return rgba


@lru_cache(maxsize=128)
def overlay_png_bytes(key: str, date_index: int) -> bytes:
    data = require_viewer_data()
    spec = spec_by_key(key)
    if spec.kind == "aoi":
        raise ValueError("AOI is rendered as vector geometry, not an image overlay.")

    values = values_for_layer(key, date_index)
    visible = data.spatial_mask.copy()

    if spec.kind == "mask":
        mask_values = np.where(np.isfinite(values), values > 0.5, False)
        rgba = np.zeros(values.shape + (4,), dtype="uint8")
        rgba[data.spatial_mask & ~mask_values] = np.array([154, 167, 173, 185], dtype="uint8")
        rgba[data.spatial_mask & mask_values] = np.array([33, 166, 122, 225], dtype="uint8")
    else:
        visible &= np.isfinite(values)
        low, high = data.layer_ranges[key]
        if spec.symmetric:
            norm = TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
        else:
            norm = Normalize(vmin=low, vmax=high)

        cmap = colormaps.get_cmap(spec.colormap)
        normalized = np.zeros(values.shape, dtype="float64")
        normalized[visible] = norm(values[visible])
        colors = cmap(normalized)
        alpha = np.where(visible, 1.0, 0.0)
        rgba = rgba_to_uint8(colors, alpha)

    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def build_legend(spec: DataLayerSpec) -> dict | None:
    if spec.kind == "aoi":
        return None

    data = require_viewer_data()
    low, high = data.layer_ranges.get(spec.key, (0.0, 1.0))
    if spec.kind == "mask":
        return {
            "low": "invalid",
            "high": "valid",
            "colors": ["#9aa7ad", "#21a67a"],
        }

    if spec.colormap == "viridis":
        colors = ["#440154", "#31688e", "#35b779", "#fde725"]
    else:
        colors = ["#08306b", "#f7f7f7", "#67000d"]

    return {
        "low": format_number(low),
        "high": format_number(high),
        "colors": colors,
    }


def build_viewer_payload() -> dict:
    data = require_viewer_data()
    default_date_index = clamp_date_index(None)

    return {
        "title": data.title,
        "dataset": str(data.dataset_path),
        "center": [data.center_lat, data.center_lon],
        "bounds": [
            [float(data.lat_edges[0]), float(data.lon_edges[0])],
            [float(data.lat_edges[-1]), float(data.lon_edges[-1])],
        ],
        "defaultZoom": DEFAULT_ZOOM,
        "dates": [date.strftime("%Y-%m-%d") for date in data.dates],
        "defaultDateIndex": default_date_index,
        "baseLayers": [
            {
                "key": spec.key,
                "label": spec.label,
                "url": spec.url,
                "attribution": spec.attribution,
                "defaultEnabled": spec.default_enabled,
                "defaultOpacity": spec.default_opacity,
                "maxZoom": spec.max_zoom,
            }
            for spec in BASEMAP_SPECS
        ],
        "dataLayers": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "units": spec.units,
                "defaultEnabled": spec.default_enabled,
                "defaultOpacity": spec.default_opacity,
                "temporal": spec.temporal,
                "legend": build_legend(spec),
            }
            for spec in data.layer_specs
        ],
        "aoi": (
            [[float(lat), float(lon)] for lon, lat in zip(data.aoi_lons, data.aoi_lats)]
            if data.aoi_lons is not None and data.aoi_lats is not None
            else None
        ),
        "seriesColors": SERIES_COLORS,
    }


def cell_index_from_latlon(lat: float, lon: float) -> tuple[int, int] | None:
    data = require_viewer_data()
    if lat < data.lat_edges[0] or lat > data.lat_edges[-1]:
        return None
    if lon < data.lon_edges[0] or lon > data.lon_edges[-1]:
        return None

    row = int(np.searchsorted(data.lat_edges, lat, side="right") - 1)
    col = int(np.searchsorted(data.lon_edges, lon, side="right") - 1)
    if row < 0 or row >= len(data.latitudes) or col < 0 or col >= len(data.longitudes):
        return None
    return row, col


def pixel_metrics(row: int, col: int, date_index: int) -> list[dict]:
    metrics = []
    for spec in require_viewer_data().layer_specs:
        if spec.kind == "aoi":
            continue
        value = values_for_layer(spec.key, date_index)[row, col]
        metrics.append(
            {
                "label": spec.label,
                "value": finite_or_none(value),
                "units": spec.units,
            }
        )
    return metrics


def pixel_time_series(row: int, col: int) -> list[dict]:
    data = require_viewer_data()
    series = []
    for variable, label in data.displacement_labels.items():
        values = data.temporal_values[variable][:, row, col]
        if not np.isfinite(values).any():
            continue
        series.append(
            {
                "key": variable,
                "label": label,
                "color": SERIES_COLORS.get(variable, "#126a65"),
                "values": [finite_or_none(value) for value in values],
            }
        )
    return series


def pixel_payload(lat: float, lon: float, date_index: int) -> dict:
    data = require_viewer_data()
    index = cell_index_from_latlon(lat, lon)
    if index is None:
        return {"found": False, "reason": "outside grid"}

    row, col = index
    if not data.spatial_mask[row, col]:
        return {"found": False, "reason": "outside AOI"}

    series = pixel_time_series(row, col)
    if not series:
        return {"found": False, "reason": "no finite displacement series"}

    return {
        "found": True,
        "row": row,
        "col": col,
        "lat": float(data.latitudes[row]),
        "lon": float(data.longitudes[col]),
        "cellBounds": [
            [float(data.lat_edges[row]), float(data.lon_edges[col])],
            [float(data.lat_edges[row + 1]), float(data.lon_edges[col + 1])],
        ],
        "dates": [date.strftime("%Y-%m-%d") for date in data.dates],
        "metrics": pixel_metrics(row, col, date_index),
        "series": series,
    }


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
        date_index = clamp_date_index(int(request.args.get("date", clamp_date_index(None))))
        return jsonify(pixel_payload(lat, lon, date_index))

    @app.get("/overlay/<key>/<int:date_index>.png")
    def overlay_api(key: str, date_index: int) -> Response:
        date_index = clamp_date_index(date_index)
        content = overlay_png_bytes(key, date_index)
        return Response(
            content,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return app


APP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>InSAR SBAS Viewer</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    * { box-sizing: border-box; }
    html, body, #map {
      height: 100%;
      width: 100%;
      margin: 0;
      padding: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: #172126;
    }
    #map {
      background: #dfe6e8;
    }
    .leaflet-image-layer {
      image-rendering: pixelated;
      image-rendering: crisp-edges;
      pointer-events: none;
    }
    .leaflet-data-pane,
    .leaflet-aoi-pane,
    .leaflet-selection-pane {
      pointer-events: none;
    }
    .leaflet-control-attribution {
      font-size: 10px;
    }
    .floating-control {
      position: fixed;
      top: 14px;
      right: 14px;
      z-index: 1000;
      width: min(340px, calc(100vw - 28px));
      max-height: calc(100vh - 28px);
      overflow: auto;
      border: 1px solid rgba(23, 33, 38, 0.18);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.94);
      box-shadow: 0 12px 34px rgba(10, 18, 22, 0.18);
      backdrop-filter: blur(8px);
    }
    .control-header {
      padding: 12px 13px 10px;
      border-bottom: 1px solid #dfe7ea;
    }
    .control-title {
      margin: 0;
      font-size: 16px;
      font-weight: 750;
      letter-spacing: 0;
    }
    .control-subtitle {
      margin-top: 2px;
      color: #607078;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .layer-section {
      border-bottom: 1px solid #dfe7ea;
    }
    .layer-section summary {
      cursor: pointer;
      padding: 10px 13px;
      color: #1f2b31;
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
      user-select: none;
    }
    .layer-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 32px;
      gap: 8px;
      align-items: center;
      min-height: 42px;
      padding: 7px 10px 7px 13px;
      border-top: 1px solid #e7eef1;
      background: rgba(255, 255, 255, 0.78);
    }
    .layer-check {
      display: flex;
      align-items: center;
      min-width: 0;
      gap: 8px;
      font-size: 13px;
      font-weight: 600;
      line-height: 1.25;
    }
    .layer-check input {
      flex: 0 0 auto;
      width: 14px;
      height: 14px;
      accent-color: #126a65;
    }
    .layer-check span {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .icon-button {
      display: inline-grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: #42525a;
      cursor: pointer;
    }
    .icon-button:hover {
      border-color: #bfd0d6;
      background: #f1f6f7;
      color: #126a65;
    }
    .icon-button svg {
      width: 16px;
      height: 16px;
      stroke-width: 2.2;
    }
    .settings-panel,
    .time-panel {
      position: fixed;
      z-index: 1001;
      border: 1px solid rgba(23, 33, 38, 0.2);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 16px 42px rgba(10, 18, 22, 0.2);
      backdrop-filter: blur(8px);
    }
    .settings-panel {
      top: 78px;
      right: 370px;
      width: min(310px, calc(100vw - 28px));
      padding: 13px;
    }
    .time-panel {
      right: 14px;
      bottom: 24px;
      width: min(390px, calc(100vw - 28px));
      max-height: calc(100vh - 120px);
      overflow: auto;
    }
    .panel-hidden {
      display: none;
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
      padding-bottom: 8px;
      border-bottom: 1px solid #e1e9ec;
    }
    .panel-title {
      margin: 0;
      font-size: 15px;
      font-weight: 750;
      line-height: 1.25;
    }
    .panel-body {
      padding-top: 12px;
    }
    .setting-group {
      margin-bottom: 12px;
    }
    .setting-label {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 7px;
      color: #435058;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .setting-slider {
      width: 100%;
      accent-color: #126a65;
    }
    .setting-select {
      width: 100%;
      height: 34px;
      border: 1px solid #bdcbd0;
      border-radius: 6px;
      background: #ffffff;
      color: #172126;
      font: inherit;
      font-size: 13px;
    }
    .legend-stack {
      position: fixed;
      left: 14px;
      bottom: 30px;
      z-index: 999;
      display: grid;
      gap: 8px;
      width: min(280px, calc(100vw - 28px));
      pointer-events: none;
    }
    .legend-card {
      padding: 9px 10px;
      border: 1px solid rgba(23, 33, 38, 0.17);
      border-radius: 7px;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 8px 24px rgba(10, 18, 22, 0.13);
    }
    .legend-title {
      font-size: 12px;
      font-weight: 750;
      line-height: 1.25;
    }
    .legend-date {
      margin-top: 1px;
      color: #607078;
      font-size: 11px;
    }
    .legend-scale {
      display: grid;
      grid-template-columns: auto minmax(70px, 1fr) auto;
      gap: 7px;
      align-items: center;
      margin-top: 6px;
      color: #435058;
      font-size: 11px;
    }
    .legend-gradient {
      height: 12px;
      border: 1px solid rgba(23, 33, 38, 0.18);
      border-radius: 2px;
    }
    .time-panel .panel-header {
      padding: 12px 12px 8px;
    }
    .time-panel .panel-body {
      padding: 10px 12px 12px;
    }
    .pixel-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
      margin-bottom: 8px;
    }
    .metric {
      border: 1px solid #dce5e9;
      border-radius: 6px;
      background: #fbfcfc;
      padding: 7px 8px;
    }
    .metric-label {
      color: #607078;
      font-size: 10px;
      font-weight: 750;
      text-transform: uppercase;
      line-height: 1.2;
    }
    .metric-value {
      margin-top: 3px;
      color: #172126;
      font-size: 13px;
      font-weight: 720;
      overflow-wrap: anywhere;
    }
    .chart-wrap {
      width: 100%;
      overflow: hidden;
    }
    .chart-wrap svg {
      display: block;
      width: 100%;
      height: auto;
    }
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 10px;
      margin-top: 8px;
      font-size: 11px;
      color: #42525a;
    }
    .legend-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      margin-right: 4px;
      border-radius: 50%;
      vertical-align: middle;
    }
    .toast {
      position: fixed;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%);
      z-index: 1002;
      padding: 8px 11px;
      border-radius: 6px;
      background: rgba(23, 33, 38, 0.9);
      color: #ffffff;
      font-size: 12px;
      pointer-events: none;
    }
    @media (max-width: 900px) {
      .settings-panel {
        top: auto;
        right: 14px;
        left: 14px;
        bottom: 18px;
        width: auto;
      }
      .time-panel {
        left: 14px;
        right: 14px;
        width: auto;
      }
      .legend-stack {
        bottom: 20px;
      }
    }
  </style>
</head>
<body>
  <div id="map"></div>

  <div id="layer-control" class="floating-control">
    <div class="control-header">
      <h1 class="control-title">SBAS Layer Viewer</h1>
      <div id="dataset-label" class="control-subtitle"></div>
    </div>
    <details class="layer-section" open>
      <summary>Ground Maps</summary>
      <div id="base-layer-list"></div>
    </details>
    <details class="layer-section" open>
      <summary>Data Overlays</summary>
      <div id="data-layer-list"></div>
    </details>
  </div>

  <div id="settings-panel" class="settings-panel panel-hidden"></div>
  <div id="time-panel" class="time-panel panel-hidden"></div>
  <div id="legend-stack" class="legend-stack"></div>
  <div id="toast" class="toast panel-hidden"></div>

  <script>
    const state = {
      map: null,
      payload: null,
      baseLayers: {},
      dataLayers: {},
      aoiLayer: null,
      selectedLayer: null,
      layerState: {},
      settingsTarget: null,
      toastTimer: null,
    };
    window.insarViewer = state;

    const stopPropagation = (element) => {
      L.DomEvent.disableClickPropagation(element);
      L.DomEvent.disableScrollPropagation(element);
    };

    document.querySelectorAll(".floating-control, .settings-panel, .time-panel").forEach(stopPropagation);

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;",
      }[char]));
    }

    function formatMetric(value, units) {
      if (value === null || value === undefined || Number.isNaN(value)) return "no data";
      const abs = Math.abs(value);
      const number = abs >= 100 ? value.toFixed(0) : abs >= 10 ? value.toFixed(1) : value.toFixed(3);
      return `${number}${units ? ` ${units}` : ""}`;
    }

    function layerImageUrl(layer, dateIndex) {
      const index = layer.temporal ? dateIndex : 0;
      return `/overlay/${encodeURIComponent(layer.key)}/${index}.png`;
    }

    function currentDateIndex(layerKey) {
      return state.layerState[layerKey]?.dateIndex ?? state.payload.defaultDateIndex;
    }

    function initMap(payload) {
      state.map = L.map("map", {
        center: payload.center,
        zoom: payload.defaultZoom,
        zoomControl: true,
        preferCanvas: true,
      });
      state.map.createPane("dataPane");
      state.map.getPane("dataPane").style.zIndex = 420;
      state.map.createPane("aoiPane");
      state.map.getPane("aoiPane").style.zIndex = 440;
      state.map.createPane("selectionPane");
      state.map.getPane("selectionPane").style.zIndex = 460;
      L.control.scale().addTo(state.map);
      state.map.fitBounds(payload.bounds, { padding: [20, 20] });
      document.getElementById("map").addEventListener("click", onMapContainerClick);
      document.addEventListener("click", onDocumentMapClick, true);
    }

    function initLayerState(payload) {
      [...payload.baseLayers, ...payload.dataLayers].forEach((layer) => {
        state.layerState[layer.key] = {
          enabled: layer.defaultEnabled,
          opacity: layer.defaultOpacity,
          dateIndex: payload.defaultDateIndex,
        };
      });
    }

    function addBaseLayer(layer) {
      if (!state.baseLayers[layer.key]) {
        state.baseLayers[layer.key] = L.tileLayer(layer.url, {
          maxZoom: layer.maxZoom,
          maxNativeZoom: layer.maxZoom,
          opacity: state.layerState[layer.key].opacity,
          attribution: layer.attribution,
        });
      }
      if (state.layerState[layer.key].enabled && !state.map.hasLayer(state.baseLayers[layer.key])) {
        state.baseLayers[layer.key].addTo(state.map);
      }
    }

    function updateBaseLayer(layer) {
      addBaseLayer(layer);
      const object = state.baseLayers[layer.key];
      object.setOpacity(state.layerState[layer.key].opacity);
      if (state.layerState[layer.key].enabled) {
        if (!state.map.hasLayer(object)) object.addTo(state.map);
      } else if (state.map.hasLayer(object)) {
        state.map.removeLayer(object);
      }
    }

    function updateDataLayer(layer) {
      if (layer.kind === "aoi") {
        updateAoiLayer(layer);
        return;
      }

      const layerState = state.layerState[layer.key];
      if (!state.dataLayers[layer.key]) {
        state.dataLayers[layer.key] = L.imageOverlay(
          layerImageUrl(layer, layerState.dateIndex),
          state.payload.bounds,
          {
            opacity: layerState.opacity,
            pane: "dataPane",
            interactive: false,
          },
        );
      }

      const object = state.dataLayers[layer.key];
      object.setOpacity(layerState.opacity);
      object.setUrl(layerImageUrl(layer, layerState.dateIndex));

      if (layerState.enabled) {
        if (!state.map.hasLayer(object)) object.addTo(state.map);
      } else if (state.map.hasLayer(object)) {
        state.map.removeLayer(object);
      }
      renderLegends();
    }

    function updateAoiLayer(layer) {
      const layerState = state.layerState[layer.key];
      if (!state.payload.aoi) return;

      if (!state.aoiLayer) {
        state.aoiLayer = L.polygon(state.payload.aoi, {
          pane: "aoiPane",
          color: "#ffcc33",
          weight: 3,
          opacity: layerState.opacity,
          fillColor: "#ffcc33",
          fillOpacity: 0.08 * layerState.opacity,
          interactive: false,
        });
      }

      state.aoiLayer.setStyle({
        opacity: layerState.opacity,
        fillOpacity: 0.08 * layerState.opacity,
      });

      if (layerState.enabled) {
        if (!state.map.hasLayer(state.aoiLayer)) state.aoiLayer.addTo(state.map);
      } else if (state.map.hasLayer(state.aoiLayer)) {
        state.map.removeLayer(state.aoiLayer);
      }
    }

    function createLayerRow(layer, group) {
      const row = document.createElement("div");
      row.className = "layer-row";
      row.innerHTML = `
        <label class="layer-check">
          <input type="checkbox" ${layer.defaultEnabled ? "checked" : ""} />
          <span>${escapeHtml(layer.label)}</span>
        </label>
        <button class="icon-button" type="button" title="Layer settings" aria-label="Layer settings for ${escapeHtml(layer.label)}">
          <i data-lucide="settings"></i>
        </button>
      `;

      const checkbox = row.querySelector("input");
      checkbox.addEventListener("change", () => {
        state.layerState[layer.key].enabled = checkbox.checked;
        group === "base" ? updateBaseLayer(layer) : updateDataLayer(layer);
      });

      row.querySelector("button").addEventListener("click", () => openSettings(layer, group));
      return row;
    }

    function renderLayerLists(payload) {
      document.getElementById("dataset-label").textContent = payload.dataset.split(/[\\/]/).pop();
      const baseList = document.getElementById("base-layer-list");
      const dataList = document.getElementById("data-layer-list");
      payload.baseLayers.forEach((layer) => baseList.appendChild(createLayerRow(layer, "base")));
      payload.dataLayers.forEach((layer) => dataList.appendChild(createLayerRow(layer, "data")));
      lucide.createIcons();
    }

    function openSettings(layer, group) {
      state.settingsTarget = { key: layer.key, group };
      const panel = document.getElementById("settings-panel");
      const layerState = state.layerState[layer.key];
      const dates = state.payload.dates;
      const dateControl = layer.temporal ? `
        <div class="setting-group">
          <label class="setting-label" for="setting-date">
            <span>Date</span><span>${escapeHtml(dates[layerState.dateIndex] || "")}</span>
          </label>
          <select id="setting-date" class="setting-select">
            ${dates.map((date, index) => `<option value="${index}" ${index === layerState.dateIndex ? "selected" : ""}>${escapeHtml(date)}</option>`).join("")}
          </select>
        </div>
      ` : "";

      panel.innerHTML = `
        <div class="panel-header">
          <h2 class="panel-title">${escapeHtml(layer.label)}</h2>
          <button class="icon-button" type="button" id="close-settings" aria-label="Close settings">
            <i data-lucide="x"></i>
          </button>
        </div>
        <div class="panel-body">
          <div class="setting-group">
            <label class="setting-label" for="setting-opacity">
              <span>Opacity</span><span id="opacity-value">${Math.round(layerState.opacity * 100)}%</span>
            </label>
            <input id="setting-opacity" class="setting-slider" type="range" min="0" max="100" step="5" value="${Math.round(layerState.opacity * 100)}" />
          </div>
          ${dateControl}
        </div>
      `;
      panel.classList.remove("panel-hidden");
      stopPropagation(panel);
      lucide.createIcons();

      panel.querySelector("#close-settings").addEventListener("click", () => {
        panel.classList.add("panel-hidden");
      });

      panel.querySelector("#setting-opacity").addEventListener("input", (event) => {
        const opacity = Number(event.target.value) / 100;
        state.layerState[layer.key].opacity = opacity;
        panel.querySelector("#opacity-value").textContent = `${event.target.value}%`;
        group === "base" ? updateBaseLayer(layer) : updateDataLayer(layer);
      });

      const dateSelect = panel.querySelector("#setting-date");
      if (dateSelect) {
        dateSelect.addEventListener("change", (event) => {
          state.layerState[layer.key].dateIndex = Number(event.target.value);
          updateDataLayer(layer);
          openSettings(layer, group);
        });
      }
    }

    function renderLegends() {
      const stack = document.getElementById("legend-stack");
      stack.innerHTML = "";
      state.payload.dataLayers.forEach((layer) => {
        const layerState = state.layerState[layer.key];
        if (!layerState.enabled || layer.kind === "aoi" || !layer.legend) return;
        const gradient = `linear-gradient(to right, ${layer.legend.colors.join(",")})`;
        const card = document.createElement("div");
        card.className = "legend-card";
        card.innerHTML = `
          <div class="legend-title">${escapeHtml(layer.label)}</div>
          ${layer.temporal ? `<div class="legend-date">${escapeHtml(state.payload.dates[layerState.dateIndex] || "")}</div>` : ""}
          <div class="legend-scale">
            <span>${escapeHtml(layer.legend.low)}</span>
            <span class="legend-gradient" style="background:${gradient}"></span>
            <span>${escapeHtml(layer.legend.high)}</span>
          </div>
        `;
        stack.appendChild(card);
      });
    }

    function svgPoint(x, y) {
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }

    function buildTimeSeriesSvg(info) {
      const width = 350;
      const height = 190;
      const margin = { left: 42, right: 12, top: 12, bottom: 34 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const values = info.series.flatMap((series) => series.values.filter((value) => value !== null));
      if (!values.length) return "<div class='control-subtitle'>No finite displacement values for this pixel.</div>";

      let minY = Math.min(...values);
      let maxY = Math.max(...values);
      if (minY === maxY) {
        minY -= 1;
        maxY += 1;
      }
      const yPadding = (maxY - minY) * 0.12;
      minY -= yPadding;
      maxY += yPadding;

      const xFor = (index) => margin.left + (info.dates.length <= 1 ? 0 : (index / (info.dates.length - 1)) * plotWidth);
      const yFor = (value) => margin.top + ((maxY - value) / (maxY - minY)) * plotHeight;
      const zeroY = yFor(Math.max(minY, Math.min(maxY, 0)));

      const lines = info.series.map((series) => {
        const points = series.values
          .map((value, index) => value === null ? null : [xFor(index), yFor(value)])
          .filter(Boolean);
        if (!points.length) return "";
        const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${svgPoint(point[0], point[1])}`).join(" ");
        const circles = points.map((point) => `<circle cx="${point[0].toFixed(1)}" cy="${point[1].toFixed(1)}" r="2.8" fill="${series.color}"></circle>`).join("");
        return `<path d="${path}" fill="none" stroke="${series.color}" stroke-width="2.4"></path>${circles}`;
      }).join("");

      const firstDate = info.dates[0] || "";
      const lastDate = info.dates[info.dates.length - 1] || "";
      const ticks = [minY, 0, maxY]
        .filter((value, index, array) => array.indexOf(value) === index)
        .map((value) => {
          const y = yFor(value);
          return `
            <line x1="${margin.left}" x2="${width - margin.right}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#e2eaed"></line>
            <text x="${margin.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-size="10" fill="#607078">${formatMetric(value, "")}</text>
          `;
        }).join("");

      return `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Pixel displacement time series">
          <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
          ${ticks}
          <line x1="${margin.left}" x2="${width - margin.right}" y1="${zeroY.toFixed(1)}" y2="${zeroY.toFixed(1)}" stroke="#9aa7ad" stroke-dasharray="4 4"></line>
          <line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}" stroke="#9aa7ad"></line>
          <line x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}" stroke="#9aa7ad"></line>
          ${lines}
          <text x="${margin.left}" y="${height - 10}" text-anchor="start" font-size="10" fill="#607078">${escapeHtml(firstDate)}</text>
          <text x="${width - margin.right}" y="${height - 10}" text-anchor="end" font-size="10" fill="#607078">${escapeHtml(lastDate)}</text>
          <text x="12" y="${margin.top + plotHeight / 2}" transform="rotate(-90 12 ${margin.top + plotHeight / 2})" text-anchor="middle" font-size="10" fill="#607078">mm</text>
        </svg>
      `;
    }

    function showTimePanel(info) {
      const panel = document.getElementById("time-panel");
      const chartLegend = info.series.map((series) => `
        <span><span class="legend-dot" style="background:${series.color}"></span>${escapeHtml(series.label)}</span>
      `).join("");
      const metrics = [
        { label: "Longitude", value: info.lon.toFixed(6) },
        { label: "Latitude", value: info.lat.toFixed(6) },
        ...info.metrics.filter((metric) => metric.value !== null).slice(0, 4).map((metric) => ({
          label: metric.label,
          value: formatMetric(metric.value, metric.units),
        })),
      ];

      panel.innerHTML = `
        <div class="panel-header">
          <h2 class="panel-title">Pixel deformation</h2>
          <button class="icon-button" type="button" id="close-time" aria-label="Close time series">
            <i data-lucide="x"></i>
          </button>
        </div>
        <div class="panel-body">
          <div class="pixel-meta">
            ${metrics.map((metric) => `
              <div class="metric">
                <div class="metric-label">${escapeHtml(metric.label)}</div>
                <div class="metric-value">${escapeHtml(metric.value)}</div>
              </div>
            `).join("")}
          </div>
          <div class="chart-wrap">${buildTimeSeriesSvg(info)}</div>
          <div class="chart-legend">${chartLegend}</div>
        </div>
      `;
      panel.classList.remove("panel-hidden");
      stopPropagation(panel);
      panel.querySelector("#close-time").addEventListener("click", () => panel.classList.add("panel-hidden"));
      lucide.createIcons();
    }

    function updateSelectedPixel(info) {
      if (state.selectedLayer) {
        state.map.removeLayer(state.selectedLayer);
      }
      state.selectedLayer = L.rectangle(info.cellBounds, {
        pane: "selectionPane",
        color: "#ffcc33",
        weight: 2,
        opacity: 1,
        fill: false,
        interactive: false,
      }).addTo(state.map);
    }

    function showToast(message) {
      const toast = document.getElementById("toast");
      toast.textContent = message;
      toast.classList.remove("panel-hidden");
      clearTimeout(state.toastTimer);
      state.toastTimer = setTimeout(() => toast.classList.add("panel-hidden"), 1800);
    }

    function onMapContainerClick(event) {
      if (event.target.closest(".leaflet-control, .floating-control, .settings-panel, .time-panel")) return;
      event.stopPropagation();
      routeMapClick(event);
    }

    function onDocumentMapClick(event) {
      if (event.target.closest(".leaflet-control, .floating-control, .settings-panel, .time-panel")) return;
      if (!document.getElementById("map").contains(event.target)) return;
      routeMapClick(event);
    }

    function routeMapClick(event) {
      document.body.dataset.lastMapClick = `${event.clientX},${event.clientY}`;
      const rect = document.getElementById("map").getBoundingClientRect();
      const point = L.point(event.clientX - rect.left, event.clientY - rect.top);
      handleMapLatLng(state.map.containerPointToLatLng(point));
    }

    async function handleMapLatLng(latlng) {
      const dateIndex = currentDateIndex(PRIMARY_DISPLACEMENT_LAYER);
      const response = await fetch(`/api/pixel?lat=${latlng.lat}&lon=${latlng.lng}&date=${dateIndex}`);
      if (!response.ok) {
        showToast("Pixel lookup failed");
        return;
      }
      const info = await response.json();
      if (!info.found) {
        showToast("No deformation pixel here");
        return;
      }
      updateSelectedPixel(info);
      showTimePanel(info);
    }

    function applyInitialLayers(payload) {
      payload.baseLayers.forEach(updateBaseLayer);
      payload.dataLayers.forEach(updateDataLayer);
      renderLegends();
    }

    async function boot() {
      const response = await fetch("/api/viewer");
      const payload = await response.json();
      state.payload = payload;
      initLayerState(payload);
      initMap(payload);
      renderLayerLists(payload);
      applyInitialLayers(payload);
    }

    boot().catch((error) => {
      console.error(error);
      showToast("Viewer failed to start");
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    args = parse_args()
    project_paths = resolve_project_paths(args.project_dir)
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else default_dataset_path(project_paths)
    )
    VIEWER_DATA = load_viewer_data(dataset_path, project_paths.parameters_path)

    print(f"Project folder: {project_paths.root_dir}")
    print(f"Product folder: {project_paths.product_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Open: http://{args.host}:{args.port}")

    flask_app = create_app()
    flask_app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)
