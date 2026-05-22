"""
Interactive SBAS map and time-series viewer.

Run:
    python insar_deformation_viewer.py Data\\project_D_results_only

Then open:
    http://127.0.0.1:8050

The project folder can be either the outer export bundle or the inner
outputs/<project>_<orbit> product folder.

Dependency notes:
    pip install dash plotly xarray netcdf4 pandas matplotlib numpy
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html

from insar_project import ProjectPaths, add_project_dir_argument, find_netcdf_files, resolve_project_paths


APP_ACCENT = "#126a65"
APP_BG = "#edf1f2"
PANEL_BG = "#ffffff"
TEXT_MAIN = "#172126"
TEXT_MUTED = "#5d6970"

DEFAULT_ZOOM = 14


@dataclass(frozen=True)
class TileLayerSpec:
    key: str
    label: str
    url: str
    attribution: str
    default_enabled: bool
    default_opacity: float


@dataclass(frozen=True)
class DataLayerSpec:
    key: str
    label: str
    variable: str | None
    kind: str
    units: str
    colorscale: str | list
    default_enabled: bool
    default_opacity: float


@dataclass(frozen=True)
class ViewerData:
    dataset_path: Path
    parameters_path: Path
    title: str
    dates: pd.DatetimeIndex
    pixel_lats: np.ndarray
    pixel_lons: np.ndarray
    pixel_rows: np.ndarray
    pixel_cols: np.ndarray
    layer_values: dict[str, np.ndarray]
    layer_specs: tuple[DataLayerSpec, ...]
    displacement_series: dict[str, np.ndarray]
    displacement_labels: dict[str, str]
    aoi_lons: np.ndarray | None
    aoi_lats: np.ndarray | None
    center_lat: float
    center_lon: float
    default_selected_ids: list[int]


BASEMAP_SPECS: tuple[TileLayerSpec, ...] = (
    TileLayerSpec(
        key="esri_satellite",
        label="Esri Satellite",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Tiles (c) Esri",
        default_enabled=True,
        default_opacity=1.0,
    ),
    TileLayerSpec(
        key="openstreetmap",
        label="OpenStreetMap",
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="(c) OpenStreetMap contributors",
        default_enabled=False,
        default_opacity=0.75,
    ),
    TileLayerSpec(
        key="carto_light",
        label="Carto Light",
        url="https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attribution="(c) OpenStreetMap contributors (c) CARTO",
        default_enabled=False,
        default_opacity=0.7,
    ),
)

REQUESTED_DATA_LAYERS: tuple[DataLayerSpec, ...] = (
    DataLayerSpec(
        key="sbas_velocity_raw",
        label="SBAS velocity (raw) [mm/year]",
        variable="sbas_velocity_raw",
        kind="scalar",
        units="mm/year",
        colorscale="RdBu_r",
        default_enabled=False,
        default_opacity=0.82,
    ),
    DataLayerSpec(
        key="sbas_velocity_masked",
        label="SBAS velocity (masked) [mm/year]",
        variable="sbas_velocity_masked",
        kind="scalar",
        units="mm/year",
        colorscale="RdBu_r",
        default_enabled=True,
        default_opacity=0.88,
    ),
    DataLayerSpec(
        key="coherence_median",
        label="Coherence (median across pairs)",
        variable="coherence_median",
        kind="scalar",
        units="coherence",
        colorscale="Viridis",
        default_enabled=False,
        default_opacity=0.76,
    ),
    DataLayerSpec(
        key="valid_pixel_mask",
        label="valid_pixel_mask (coh >= 0.3)",
        variable="valid_pixel_mask",
        kind="mask",
        units="valid",
        colorscale=[
            [0.0, "#9aa7ad"],
            [0.499, "#9aa7ad"],
            [0.5, "#21a67a"],
            [1.0, "#21a67a"],
        ],
        default_enabled=False,
        default_opacity=0.62,
    ),
    DataLayerSpec(
        key="aoi_original",
        label="AOI (original)",
        variable=None,
        kind="aoi",
        units="",
        colorscale="",
        default_enabled=True,
        default_opacity=1.0,
    ),
)

DISPLACEMENT_VARIABLES = (
    ("sbas_displacement_raw", "SBAS raw displacement"),
    ("sbas_displacement_masked", "SBAS masked displacement"),
    ("sbas_displacement_segmented_same_pixel", "SBAS segmented displacement"),
    ("displacement_sbas", "SBAS displacement"),
)

VIEWER_DATA: ViewerData | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_dir_argument(parser)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="NetCDF dataset to view. Defaults to results_tight.nc, then results_wide.nc.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port.")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
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


def finite_mean_by_column(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    means = np.full(values.shape[1], np.nan, dtype="float64")
    valid_columns = counts > 0
    means[valid_columns] = sums[valid_columns] / counts[valid_columns]
    return means


def nearest_pixel_id(
    pixel_lons: np.ndarray,
    pixel_lats: np.ndarray,
    lon: float,
    lat: float,
    candidate_ids: np.ndarray | None = None,
) -> int:
    if candidate_ids is None or len(candidate_ids) == 0:
        distances = (pixel_lons - lon) ** 2 + (pixel_lats - lat) ** 2
        return int(np.nanargmin(distances))

    distances = (pixel_lons[candidate_ids] - lon) ** 2 + (pixel_lats[candidate_ids] - lat) ** 2
    return int(candidate_ids[int(np.nanargmin(distances))])


def finite_color_range(values: np.ndarray, symmetric: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)

    if symmetric:
        max_abs = float(np.nanpercentile(np.abs(finite), 98))
        if max_abs == 0:
            max_abs = 1.0
        return -max_abs, max_abs

    low, high = np.nanpercentile(finite, [2, 98])
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def build_spatial_mask(dataset, latitudes: np.ndarray, longitudes: np.ndarray, aoi_coordinates) -> np.ndarray:
    if "aoi_mask" in dataset:
        return dataset["aoi_mask"].values.astype(bool)

    if aoi_coordinates:
        polygon_mask = make_polygon_mask(longitudes, latitudes, aoi_coordinates)
        if polygon_mask is not None:
            return polygon_mask

    candidate_masks = []
    for spec in REQUESTED_DATA_LAYERS:
        if spec.variable and spec.variable in dataset:
            values = dataset[spec.variable].transpose("lat", "lon").values
            candidate_masks.append(np.isfinite(values))

    if not candidate_masks:
        raise ValueError("No spatial data layers were found in the dataset.")

    return np.logical_or.reduce(candidate_masks)


def load_viewer_data(dataset_path: Path, parameters_path: Path) -> ViewerData:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")

    parameters = load_parameters(parameters_path)
    aoi_coordinates = parse_polygon_wkt((parameters.get("aoi") or {}).get("raw_wkt"))
    aoi_lons = np.array([lon for lon, _lat in aoi_coordinates]) if aoi_coordinates else None
    aoi_lats = np.array([lat for _lon, lat in aoi_coordinates]) if aoi_coordinates else None

    with xr.open_dataset(dataset_path) as dataset:
        dataset = dataset.load()

    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError("Dataset must contain lat and lon coordinates.")

    latitudes = dataset["lat"].values
    longitudes = dataset["lon"].values
    spatial_mask = build_spatial_mask(dataset, latitudes, longitudes, aoi_coordinates)
    pixel_rows, pixel_cols = np.where(spatial_mask)
    if len(pixel_rows) == 0:
        raise ValueError("No pixels are available inside the selected spatial mask.")

    layer_specs = tuple(
        spec
        for spec in REQUESTED_DATA_LAYERS
        if spec.kind == "aoi" or (spec.variable is not None and spec.variable in dataset.data_vars)
    )

    layer_values: dict[str, np.ndarray] = {}
    for spec in layer_specs:
        if spec.variable is None:
            continue
        values = dataset[spec.variable].transpose("lat", "lon").values
        layer_values[spec.key] = values[pixel_rows, pixel_cols]

    if "date" in dataset.coords:
        dates = pd.DatetimeIndex(pd.to_datetime(dataset["date"].values))
    else:
        dates = pd.DatetimeIndex([])

    displacement_series: dict[str, np.ndarray] = {}
    displacement_labels: dict[str, str] = {}
    for variable, label in DISPLACEMENT_VARIABLES:
        if variable not in dataset.data_vars or "date" not in dataset[variable].dims:
            continue
        values = dataset[variable].transpose("date", "lat", "lon").values
        displacement_series[variable] = values[:, pixel_rows, pixel_cols].T
        displacement_labels[variable] = label

    pixel_lats = latitudes[pixel_rows]
    pixel_lons = longitudes[pixel_cols]

    default_candidates = np.arange(len(pixel_lons))
    masked_velocity = layer_values.get("sbas_velocity_masked")
    if masked_velocity is not None and np.isfinite(masked_velocity).any():
        default_candidates = np.flatnonzero(np.isfinite(masked_velocity))

    default_selected_ids = [int(default_candidates[0])] if len(default_candidates) else []
    pois = parameters.get("pois") or []
    if pois:
        lon = pois[0].get("lon")
        lat = pois[0].get("lat")
        if lon is not None and lat is not None:
            default_selected_ids = [
                nearest_pixel_id(
                    pixel_lons,
                    pixel_lats,
                    float(lon),
                    float(lat),
                    default_candidates,
                )
            ]

    title = str(dataset.attrs.get("title") or dataset_path.stem)
    return ViewerData(
        dataset_path=dataset_path,
        parameters_path=parameters_path,
        title=title,
        dates=dates,
        pixel_lats=pixel_lats,
        pixel_lons=pixel_lons,
        pixel_rows=pixel_rows,
        pixel_cols=pixel_cols,
        layer_values=layer_values,
        layer_specs=layer_specs,
        displacement_series=displacement_series,
        displacement_labels=displacement_labels,
        aoi_lons=aoi_lons,
        aoi_lats=aoi_lats,
        center_lat=float(np.nanmean(pixel_lats)),
        center_lon=float(np.nanmean(pixel_lons)),
        default_selected_ids=default_selected_ids,
    )


def opacity_to_float(value) -> float:
    if value is None:
        return 1.0
    return max(0.0, min(1.0, float(value) / 100.0))


def enabled_dict(keys: list[str], checklist_values: list[list[str]], opacity_values: list[float]) -> dict:
    states = {}
    for index, key in enumerate(keys):
        selected_values = checklist_values[index] if index < len(checklist_values) else []
        opacity = opacity_values[index] if index < len(opacity_values) else 100
        states[key] = {
            "enabled": key in (selected_values or []),
            "opacity": opacity_to_float(opacity),
        }
    return states


def rgba(hex_color: str, opacity: float) -> str:
    hex_value = hex_color.lstrip("#")
    red = int(hex_value[0:2], 16)
    green = int(hex_value[2:4], 16)
    blue = int(hex_value[4:6], 16)
    return f"rgba({red},{green},{blue},{opacity:.3f})"


def mapbox_tile_layers(base_states: dict) -> list[dict]:
    layers = []
    for spec in BASEMAP_SPECS:
        state = base_states.get(spec.key, {})
        if not state.get("enabled"):
            continue
        layers.append(
            {
                "sourcetype": "raster",
                "source": [spec.url],
                "below": "traces",
                "opacity": state.get("opacity", spec.default_opacity),
            }
        )
    return layers


def selected_ids_from_points(points: list[dict]) -> list[int]:
    ids = []
    for point in points:
        customdata = point.get("customdata")
        if isinstance(customdata, (list, tuple)):
            customdata = customdata[0]
        if customdata is None:
            continue
        try:
            ids.append(int(customdata))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids))


def clean_selected_ids(selected_ids: list[int]) -> list[int]:
    total = len(VIEWER_DATA.pixel_lons)
    return sorted({int(pixel_id) for pixel_id in selected_ids if 0 <= int(pixel_id) < total})


def add_aoi_trace(figure: go.Figure, opacity: float) -> None:
    if VIEWER_DATA.aoi_lons is None or VIEWER_DATA.aoi_lats is None:
        return

    figure.add_trace(
        go.Scattermapbox(
            lon=VIEWER_DATA.aoi_lons,
            lat=VIEWER_DATA.aoi_lats,
            mode="lines",
            line={"color": rgba("#ffcc33", opacity), "width": 3},
            fill="toself",
            fillcolor=rgba("#ffcc33", opacity * 0.08),
            hoverinfo="skip",
            showlegend=False,
            name="AOI (original)",
        )
    )


def add_data_layer_trace(
    figure: go.Figure,
    spec: DataLayerSpec,
    values: np.ndarray,
    opacity: float,
    colorbar_index: int,
) -> int:
    if spec.kind == "mask":
        finite = np.isfinite(values)
        marker_size = 6
        cmin, cmax = 0, 1
    else:
        finite = np.isfinite(values)
        marker_size = 6
        cmin, cmax = finite_color_range(values[finite], symmetric=spec.key.startswith("sbas_velocity"))

    if not finite.any():
        return colorbar_index

    pixel_ids = np.flatnonzero(finite)
    colorbar_y = max(0.18, 0.86 - (colorbar_index * 0.24))
    customdata = np.column_stack([pixel_ids, values[finite]])

    figure.add_trace(
        go.Scattermapbox(
            lon=VIEWER_DATA.pixel_lons[finite],
            lat=VIEWER_DATA.pixel_lats[finite],
            mode="markers",
            customdata=customdata,
            marker={
                "size": marker_size,
                "color": values[finite],
                "colorscale": spec.colorscale,
                "cmin": cmin,
                "cmax": cmax,
                "opacity": opacity,
                "colorbar": {
                    "title": spec.units,
                    "thickness": 12,
                    "len": 0.22,
                    "y": colorbar_y,
                    "x": 1.01,
                },
            },
            hovertemplate=(
                f"<b>{spec.label}</b><br>"
                "lon=%{lon:.6f}<br>"
                "lat=%{lat:.6f}<br>"
                f"value=%{{customdata[1]:.3f}} {spec.units}"
                "<extra></extra>"
            ),
            showlegend=False,
            name=spec.label,
        )
    )
    return colorbar_index + 1


def build_map_figure(base_states: dict, data_states: dict, selected_ids: list[int]) -> go.Figure:
    figure = go.Figure()
    colorbar_index = 0

    for spec in VIEWER_DATA.layer_specs:
        state = data_states.get(spec.key, {})
        if not state.get("enabled"):
            continue

        opacity = state.get("opacity", spec.default_opacity)
        if spec.kind == "aoi":
            add_aoi_trace(figure, opacity)
            continue

        values = VIEWER_DATA.layer_values.get(spec.key)
        if values is None:
            continue
        colorbar_index = add_data_layer_trace(figure, spec, values, opacity, colorbar_index)

    selected_ids_clean = clean_selected_ids(selected_ids)
    if selected_ids_clean:
        selected_customdata = np.array(selected_ids_clean, dtype=int)
        figure.add_trace(
            go.Scattermapbox(
                lon=VIEWER_DATA.pixel_lons[selected_ids_clean],
                lat=VIEWER_DATA.pixel_lats[selected_ids_clean],
                mode="markers",
                customdata=selected_customdata,
                marker={
                    "size": 14,
                    "color": "#111820",
                    "opacity": 0.85,
                },
                hoverinfo="skip",
                showlegend=False,
                name="Selected pixel halo",
            )
        )
        figure.add_trace(
            go.Scattermapbox(
                lon=VIEWER_DATA.pixel_lons[selected_ids_clean],
                lat=VIEWER_DATA.pixel_lats[selected_ids_clean],
                mode="markers",
                customdata=selected_customdata,
                marker={"size": 7, "color": "#ffcc33", "opacity": 1.0},
                hovertemplate=(
                    "<b>Selected pixel</b><br>"
                    "lon=%{lon:.6f}<br>"
                    "lat=%{lat:.6f}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name="Selected pixels",
            )
        )

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        clickmode="event+select",
        uirevision="insar-map",
        mapbox={
            "style": "white-bg",
            "center": {"lat": VIEWER_DATA.center_lat, "lon": VIEWER_DATA.center_lon},
            "zoom": DEFAULT_ZOOM,
            "layers": mapbox_tile_layers(base_states),
        },
        font={"family": "Segoe UI, Arial, sans-serif", "color": TEXT_MAIN},
    )
    return figure


def build_timeseries_figure(selected_ids: list[int]) -> go.Figure:
    selected_ids_clean = clean_selected_ids(selected_ids)
    figure = go.Figure()

    if not selected_ids_clean or not VIEWER_DATA.displacement_series:
        figure.update_layout(
            annotations=[
                {
                    "text": "Click a data pixel to inspect its displacement history",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14, "color": TEXT_MUTED},
                }
            ]
        )
    else:
        colors = {
            "sbas_displacement_raw": "#38598c",
            "sbas_displacement_masked": "#126a65",
            "sbas_displacement_segmented_same_pixel": "#9b5de5",
            "displacement_sbas": "#126a65",
        }
        for variable, series in VIEWER_DATA.displacement_series.items():
            selected_series = series[selected_ids_clean, :]
            if not np.isfinite(selected_series).any():
                continue
            mean_series = finite_mean_by_column(selected_series)
            figure.add_trace(
                go.Scatter(
                    x=VIEWER_DATA.dates,
                    y=mean_series,
                    mode="lines+markers",
                    line={"color": colors.get(variable, APP_ACCENT), "width": 3},
                    marker={"size": 7, "color": colors.get(variable, APP_ACCENT)},
                    hovertemplate="date=%{x|%Y-%m-%d}<br>displacement=%{y:.2f} mm<extra></extra>",
                    name=VIEWER_DATA.displacement_labels.get(variable, variable),
                )
            )

    figure.update_layout(
        margin={"l": 58, "r": 18, "t": 28, "b": 46},
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridcolor": "#e6ecef",
            "zeroline": False,
        },
        yaxis={
            "title": "LOS displacement (mm)",
            "showgrid": True,
            "gridcolor": "#e6ecef",
            "zeroline": True,
            "zerolinecolor": "#9aa7ad",
        },
        legend={"orientation": "h", "y": 1.08, "x": 0},
        font={"family": "Segoe UI, Arial, sans-serif", "color": TEXT_MAIN},
    )
    return figure


def selected_pixel_cards(selected_ids: list[int]) -> list:
    selected_ids_clean = clean_selected_ids(selected_ids)
    if not selected_ids_clean:
        return [
            html.Div("No pixel selected", className="empty-state"),
        ]

    pixel_id = selected_ids_clean[-1]
    cards = [
        metric_card("Selected pixels", f"{len(selected_ids_clean):,}"),
        metric_card("Longitude", f"{VIEWER_DATA.pixel_lons[pixel_id]:.6f}"),
        metric_card("Latitude", f"{VIEWER_DATA.pixel_lats[pixel_id]:.6f}"),
    ]

    for spec in VIEWER_DATA.layer_specs:
        if spec.kind == "aoi" or spec.key not in VIEWER_DATA.layer_values:
            continue
        value = VIEWER_DATA.layer_values[spec.key][pixel_id]
        text = "no data" if not np.isfinite(value) else f"{value:.3f} {spec.units}".strip()
        cards.append(metric_card(spec.label, text))

    return cards


def metric_card(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ],
        className="metric",
    )


def layer_row(spec, group: str) -> html.Div:
    default_value = [spec.key] if spec.default_enabled else []
    default_opacity = int(round(spec.default_opacity * 100))
    return html.Div(
        [
            html.Div(
                [
                    dcc.Checklist(
                        id={"type": f"{group}-toggle", "key": spec.key},
                        options=[{"label": spec.label, "value": spec.key}],
                        value=default_value,
                        className="layer-toggle",
                    ),
                    html.Div(f"{default_opacity}%", className="opacity-readout"),
                ],
                className="layer-row-top",
            ),
            dcc.Slider(
                id={"type": f"{group}-opacity", "key": spec.key},
                min=0,
                max=100,
                step=5,
                value=default_opacity,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                className="opacity-slider",
            ),
        ],
        className="layer-row",
    )


def make_app() -> Dash:
    if VIEWER_DATA is None:
        raise RuntimeError("Viewer data has not been loaded.")

    app = Dash(__name__)
    app.title = "InSAR SBAS Layer Viewer"

    app.layout = html.Div(
        [
            dcc.Store(id="selection-store", data=VIEWER_DATA.default_selected_ids),
            html.Aside(
                [
                    html.Div(
                        [
                            html.H1("SBAS Layer Viewer"),
                            html.Div(VIEWER_DATA.dataset_path.name, className="subtle-text"),
                        ],
                        className="sidebar-header",
                    ),
                    html.Details(
                        [
                            html.Summary("Ground Maps"),
                            html.Div([layer_row(spec, "base") for spec in BASEMAP_SPECS]),
                        ],
                        open=True,
                        className="layer-menu",
                    ),
                    html.Details(
                        [
                            html.Summary("Data Overlays"),
                            html.Div([layer_row(spec, "data") for spec in VIEWER_DATA.layer_specs]),
                        ],
                        open=True,
                        className="layer-menu",
                    ),
                    html.Button("Clear selection", id="clear-selection", n_clicks=0, className="secondary-button"),
                ],
                className="sidebar",
            ),
            html.Main(
                [
                    html.Section(
                        dcc.Graph(
                            id="layer-map",
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "toImageButtonOptions": {"filename": "insar_layer_map"},
                            },
                            className="map-graph",
                        ),
                        className="map-panel",
                    ),
                    html.Aside(
                        [
                            html.Div(
                                [
                                    html.H2("Pixel Time Series"),
                                    html.Div(
                                        f"{len(VIEWER_DATA.displacement_series)} displacement arrays loaded",
                                        className="subtle-text",
                                    ),
                                ],
                                className="inspector-header",
                            ),
                            dcc.Graph(
                                id="timeseries-plot",
                                config={"displaylogo": False},
                                className="timeseries-graph",
                            ),
                            html.Div(id="pixel-summary", className="metrics-grid"),
                        ],
                        className="inspector",
                    ),
                ],
                className="workspace",
            ),
        ],
        className="app-shell",
    )

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                background: #edf1f2;
                color: #172126;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            .app-shell {
                height: 100vh;
                display: grid;
                grid-template-columns: 334px minmax(0, 1fr);
                overflow: hidden;
            }
            .sidebar {
                min-width: 0;
                padding: 18px 16px;
                border-right: 1px solid #d7e0e4;
                background: #ffffff;
                overflow: auto;
            }
            .sidebar-header {
                margin-bottom: 18px;
            }
            .sidebar-header h1 {
                margin: 0;
                font-size: 22px;
                font-weight: 700;
                letter-spacing: 0;
            }
            .subtle-text {
                margin-top: 4px;
                color: #5d6970;
                font-size: 13px;
                line-height: 1.35;
            }
            .layer-menu {
                margin-bottom: 14px;
                border: 1px solid #d9e3e7;
                border-radius: 8px;
                background: #fbfcfc;
                overflow: hidden;
            }
            .layer-menu summary {
                padding: 12px 13px;
                cursor: pointer;
                color: #1f2b31;
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
                user-select: none;
            }
            .layer-row {
                padding: 10px 12px 12px;
                border-top: 1px solid #e4ecef;
                background: #ffffff;
            }
            .layer-row-top {
                display: flex;
                justify-content: space-between;
                gap: 8px;
                align-items: start;
            }
            .layer-toggle label {
                color: #172126;
                font-size: 13px;
                font-weight: 620;
                line-height: 1.3;
            }
            .layer-toggle input {
                margin-right: 8px;
            }
            .opacity-readout {
                color: #6a767d;
                font-size: 12px;
                white-space: nowrap;
                padding-top: 1px;
            }
            .opacity-slider {
                margin: 5px 0 0;
            }
            .secondary-button {
                width: 100%;
                height: 38px;
                border: 1px solid #b8c6cb;
                border-radius: 7px;
                background: #ffffff;
                color: #172126;
                font-weight: 700;
                cursor: pointer;
            }
            .secondary-button:hover {
                border-color: #126a65;
                color: #126a65;
            }
            .workspace {
                min-width: 0;
                min-height: 0;
                display: grid;
                grid-template-columns: minmax(520px, 1fr) 430px;
                gap: 12px;
                padding: 12px;
            }
            .map-panel,
            .inspector {
                min-width: 0;
                min-height: 0;
                border: 1px solid #d7e0e4;
                border-radius: 8px;
                background: #ffffff;
                overflow: hidden;
            }
            .map-graph {
                height: 100%;
                min-height: calc(100vh - 24px);
            }
            .inspector {
                display: grid;
                grid-template-rows: auto minmax(280px, 42vh) minmax(0, 1fr);
                overflow: auto;
            }
            .inspector-header {
                padding: 15px 16px 8px;
                border-bottom: 1px solid #e2eaed;
            }
            .inspector-header h2 {
                margin: 0;
                font-size: 18px;
                font-weight: 700;
            }
            .timeseries-graph {
                min-height: 300px;
                border-bottom: 1px solid #e2eaed;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 9px;
                padding: 12px;
            }
            .metric {
                min-width: 0;
                border: 1px solid #dce5e9;
                border-radius: 7px;
                background: #fbfcfc;
                padding: 9px 10px;
            }
            .metric-label {
                color: #5d6970;
                font-size: 11px;
                font-weight: 700;
                line-height: 1.25;
                text-transform: uppercase;
            }
            .metric-value {
                margin-top: 4px;
                color: #172126;
                font-size: 15px;
                font-weight: 720;
                overflow-wrap: anywhere;
            }
            .empty-state {
                grid-column: 1 / -1;
                color: #5d6970;
                font-size: 14px;
                padding: 12px 4px;
            }
            @media (max-width: 1180px) {
                .app-shell {
                    grid-template-columns: 300px minmax(0, 1fr);
                }
                .workspace {
                    grid-template-columns: 1fr;
                    grid-template-rows: minmax(560px, 1fr) minmax(420px, auto);
                    overflow: auto;
                }
                .map-graph {
                    min-height: 560px;
                }
            }
            @media (max-width: 760px) {
                .app-shell {
                    display: block;
                    height: auto;
                    overflow: auto;
                }
                .sidebar {
                    border-right: 0;
                    border-bottom: 1px solid #d7e0e4;
                }
                .workspace {
                    padding: 8px;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    @app.callback(
        Output("selection-store", "data"),
        Input("layer-map", "clickData"),
        Input("layer-map", "selectedData"),
        Input("clear-selection", "n_clicks"),
        State("selection-store", "data"),
        prevent_initial_call=True,
    )
    def update_selection(click_data, selected_data, _clear_clicks, current_selection):
        if ctx.triggered_id == "clear-selection":
            return []

        triggered = ctx.triggered[0]["prop_id"].split(".")[-1] if ctx.triggered else None
        if triggered == "clickData" and click_data and click_data.get("points"):
            ids = selected_ids_from_points(click_data["points"])
            return ids[:1] if ids else (current_selection or [])

        if triggered == "selectedData" and selected_data and selected_data.get("points"):
            ids = selected_ids_from_points(selected_data["points"])
            return ids if ids else (current_selection or [])

        return current_selection or []

    @app.callback(
        Output("layer-map", "figure"),
        Output("timeseries-plot", "figure"),
        Output("pixel-summary", "children"),
        Input({"type": "base-toggle", "key": ALL}, "value"),
        Input({"type": "base-opacity", "key": ALL}, "value"),
        Input({"type": "data-toggle", "key": ALL}, "value"),
        Input({"type": "data-opacity", "key": ALL}, "value"),
        Input("selection-store", "data"),
    )
    def update_view(
        base_toggle_values,
        base_opacity_values,
        data_toggle_values,
        data_opacity_values,
        selected_ids,
    ):
        base_keys = [spec.key for spec in BASEMAP_SPECS]
        data_keys = [spec.key for spec in VIEWER_DATA.layer_specs]
        base_states = enabled_dict(base_keys, base_toggle_values, base_opacity_values)
        data_states = enabled_dict(data_keys, data_toggle_values, data_opacity_values)
        selected_ids = selected_ids or []
        return (
            build_map_figure(base_states, data_states, selected_ids),
            build_timeseries_figure(selected_ids),
            selected_pixel_cards(selected_ids),
        )

    return app


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

    dash_app = make_app()
    dash_app.run(host=args.host, port=args.port, debug=args.debug)
