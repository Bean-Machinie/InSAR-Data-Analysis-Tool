"""
Interactive AOI deformation viewer.

Run:
    python insar_deformation_viewer.py

Then open:
    http://127.0.0.1:8050

Dependency notes:
    pip install dash plotly xarray netcdf4 rioxarray pandas matplotlib numpy
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from dash import Dash, Input, Output, State, ctx, dcc, html


PROJECT_DIR = Path(r"E:\Scripts\InSAR-Data-Analysis-Tool\Data\project_dam_D")
AOI_DATASET_PATH = PROJECT_DIR / "aoi_only" / "results_aoi_masked.nc"
PARAMETERS_PATH = PROJECT_DIR / "parameters.json"

DISPLACEMENT_VARIABLES = (
    "displacement_sbas",
    "displacement_ps",
)

VARIABLE_LABELS = {
    "displacement_sbas": "SBAS displacement",
    "displacement_ps": "PS displacement",
}

QUALITY_VARIABLES = {
    "displacement_sbas": "rmse_sbas",
    "displacement_ps": "rmse_ps",
}

DEFAULT_RMSE_THRESHOLD = 0.50

APP_ACCENT = "#126a65"
APP_BG = "#f5f7f8"
PANEL_BG = "#ffffff"
TEXT_MAIN = "#172126"
TEXT_MUTED = "#5d6970"


@dataclass(frozen=True)
class ViewerData:
    dataset_path: Path
    variables: tuple[str, ...]
    dates: pd.DatetimeIndex
    latitudes: np.ndarray
    longitudes: np.ndarray
    pixel_rows: np.ndarray
    pixel_cols: np.ndarray
    pixel_lats: np.ndarray
    pixel_lons: np.ndarray
    series_by_variable: dict[str, np.ndarray]
    rmse_by_variable: dict[str, np.ndarray]
    rmse_ranges: dict[str, tuple[float, float]]
    default_selected_ids: list[int]


def load_parameters() -> dict:
    if not PARAMETERS_PATH.exists():
        return {}
    with PARAMETERS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def nearest_pixel_id(pixel_lons: np.ndarray, pixel_lats: np.ndarray, lon: float, lat: float) -> int:
    distances = (pixel_lons - lon) ** 2 + (pixel_lats - lat) ** 2
    return int(np.nanargmin(distances))


def load_viewer_data(dataset_path: Path) -> ViewerData:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"AOI dataset was not found: {dataset_path}. "
            "Run clip_insar_to_aoi.py first."
        )

    with xr.open_dataset(dataset_path) as dataset:
        dataset = dataset.load()

    variables = tuple(name for name in DISPLACEMENT_VARIABLES if name in dataset.data_vars)
    if not variables:
        raise ValueError("No displacement variables found in the AOI dataset.")

    if "date" not in dataset.coords or "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError("Dataset must contain date, lat, and lon coordinates.")

    latitudes = dataset["lat"].values
    longitudes = dataset["lon"].values
    dates = pd.DatetimeIndex(pd.to_datetime(dataset["date"].values))

    if "aoi_mask" in dataset:
        mask = dataset["aoi_mask"].values.astype(bool)
    else:
        first_variable = dataset[variables[0]].transpose("date", "lat", "lon").values
        mask = np.isfinite(first_variable).any(axis=0)

    pixel_rows, pixel_cols = np.where(mask)
    if len(pixel_rows) == 0:
        raise ValueError("AOI mask contains no valid pixels.")

    pixel_lats = latitudes[pixel_rows]
    pixel_lons = longitudes[pixel_cols]

    series_by_variable: dict[str, np.ndarray] = {}
    rmse_by_variable: dict[str, np.ndarray] = {}
    rmse_ranges: dict[str, tuple[float, float]] = {}
    for variable in variables:
        values = dataset[variable].transpose("date", "lat", "lon").values
        series_by_variable[variable] = values[:, pixel_rows, pixel_cols].T

        rmse_variable = QUALITY_VARIABLES.get(variable)
        if rmse_variable in dataset.data_vars:
            rmse_values = dataset[rmse_variable].transpose("lat", "lon").values[
                pixel_rows,
                pixel_cols,
            ]
            rmse_by_variable[variable] = rmse_values
            finite_rmse = rmse_values[np.isfinite(rmse_values)]
            if finite_rmse.size:
                rmse_ranges[variable] = (
                    float(np.nanmin(finite_rmse)),
                    float(np.nanmax(finite_rmse)),
                )

    default_selected_ids = [0]
    parameters = load_parameters()
    pois = parameters.get("pois") or []
    if pois:
        lon = pois[0].get("lon")
        lat = pois[0].get("lat")
        if lon is not None and lat is not None:
            default_selected_ids = [nearest_pixel_id(pixel_lons, pixel_lats, float(lon), float(lat))]

    return ViewerData(
        dataset_path=dataset_path,
        variables=variables,
        dates=dates,
        latitudes=latitudes,
        longitudes=longitudes,
        pixel_rows=pixel_rows,
        pixel_cols=pixel_cols,
        pixel_lats=pixel_lats,
        pixel_lons=pixel_lons,
        series_by_variable=series_by_variable,
        rmse_by_variable=rmse_by_variable,
        rmse_ranges=rmse_ranges,
        default_selected_ids=default_selected_ids,
    )


VIEWER_DATA = load_viewer_data(AOI_DATASET_PATH)


def finite_color_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs == 0:
        max_abs = 1.0

    return -max_abs, max_abs


def quality_pixel_ids(
    variable: str,
    apply_quality_mask: bool,
    rmse_threshold: float | None,
) -> np.ndarray:
    total_pixels = len(VIEWER_DATA.pixel_lons)
    if not apply_quality_mask:
        return np.arange(total_pixels)

    rmse_values = VIEWER_DATA.rmse_by_variable.get(variable)
    if rmse_values is None:
        return np.arange(total_pixels)

    threshold = DEFAULT_RMSE_THRESHOLD if rmse_threshold is None else float(rmse_threshold)
    return np.flatnonzero(np.isfinite(rmse_values) & (rmse_values <= threshold))


def clean_selected_ids(selected_ids: list[int], allowed_ids: np.ndarray) -> list[int]:
    allowed = set(int(pixel_id) for pixel_id in allowed_ids)
    return sorted(
        {
            int(pixel_id)
            for pixel_id in selected_ids
            if int(pixel_id) in allowed
        }
    )


def build_map_figure(
    variable: str,
    date_index: int,
    selected_ids: list[int],
    apply_quality_mask: bool,
    rmse_threshold: float | None,
) -> go.Figure:
    series = VIEWER_DATA.series_by_variable[variable]
    visible_ids = quality_pixel_ids(variable, apply_quality_mask, rmse_threshold)
    values = series[visible_ids, date_index]
    color_min, color_max = finite_color_range(values)
    selected_ids_clean = clean_selected_ids(selected_ids, visible_ids)

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=VIEWER_DATA.pixel_lons[visible_ids],
            y=VIEWER_DATA.pixel_lats[visible_ids],
            mode="markers",
            customdata=visible_ids.tolist(),
            marker={
                "size": 7,
                "color": values,
                "colorscale": "RdBu_r",
                "cmin": color_min,
                "cmax": color_max,
                "colorbar": {
                    "title": "mm",
                    "thickness": 14,
                    "len": 0.82,
                },
                "line": {"width": 0},
            },
            hovertemplate=(
                "lon=%{x:.6f}<br>"
                "lat=%{y:.6f}<br>"
                "deformation=%{marker.color:.2f} mm"
                "<extra></extra>"
            ),
            name="AOI pixels",
        )
    )

    if selected_ids_clean:
        figure.add_trace(
            go.Scattergl(
                x=VIEWER_DATA.pixel_lons[selected_ids_clean],
                y=VIEWER_DATA.pixel_lats[selected_ids_clean],
                mode="markers",
                marker={
                    "size": 11,
                    "color": "rgba(0, 0, 0, 0)",
                    "line": {"color": "#101820", "width": 2},
                },
                hoverinfo="skip",
                showlegend=False,
                name="Selected pixels",
            )
        )

    figure.update_layout(
        margin={"l": 38, "r": 20, "t": 34, "b": 38},
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        dragmode="lasso",
        clickmode="event+select",
        uirevision=f"{variable}-{date_index}-{apply_quality_mask}-{rmse_threshold}",
        title={
            "text": f"{VARIABLE_LABELS.get(variable, variable)} on {VIEWER_DATA.dates[date_index].date()}",
            "font": {"size": 15, "color": TEXT_MAIN},
        },
        xaxis={
            "title": "Longitude",
            "showgrid": True,
            "gridcolor": "#e6ecef",
            "zeroline": False,
        },
        yaxis={
            "title": "Latitude",
            "showgrid": True,
            "gridcolor": "#e6ecef",
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        font={"family": "Segoe UI, Arial, sans-serif", "color": TEXT_MAIN},
    )

    return figure


def series_lines_for_pixels(series: np.ndarray, selected_ids: list[int]) -> tuple[list, list]:
    dates = list(VIEWER_DATA.dates)
    x_values: list = []
    y_values: list = []

    for pixel_id in selected_ids:
        if pixel_id < 0 or pixel_id >= series.shape[0]:
            continue
        x_values.extend(dates)
        x_values.append(None)
        y_values.extend(series[pixel_id].tolist())
        y_values.append(None)

    return x_values, y_values


def build_timeseries_figure(
    variable: str,
    selected_ids: list[int],
    apply_quality_mask: bool,
    rmse_threshold: float | None,
) -> go.Figure:
    series = VIEWER_DATA.series_by_variable[variable]
    visible_ids = quality_pixel_ids(variable, apply_quality_mask, rmse_threshold)
    selected_ids_clean = clean_selected_ids(selected_ids, visible_ids)

    figure = go.Figure()

    if not selected_ids_clean:
        figure.update_layout(
            annotations=[
                {
                    "text": "No pixels selected",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 15, "color": TEXT_MUTED},
                }
            ]
        )
    else:
        x_lines, y_lines = series_lines_for_pixels(series, selected_ids_clean)
        selected_series = series[selected_ids_clean, :]
        mean_series = np.nanmean(selected_series, axis=0)

        figure.add_trace(
            go.Scattergl(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line={"color": "rgba(80, 88, 94, 0.22)", "width": 1},
                hoverinfo="skip",
                name="Selected pixels",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=VIEWER_DATA.dates,
                y=mean_series,
                mode="lines+markers",
                line={"color": APP_ACCENT, "width": 4},
                marker={"size": 7, "color": APP_ACCENT},
                hovertemplate="date=%{x|%Y-%m-%d}<br>mean=%{y:.2f} mm<extra></extra>",
                name="Mean",
            )
        )

    figure.update_layout(
        margin={"l": 58, "r": 22, "t": 34, "b": 48},
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        title={
            "text": f"{VARIABLE_LABELS.get(variable, variable)} time series",
            "font": {"size": 15, "color": TEXT_MAIN},
        },
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridcolor": "#e6ecef",
            "zeroline": False,
        },
        yaxis={
            "title": "Deformation (mm)",
            "showgrid": True,
            "gridcolor": "#e6ecef",
            "zeroline": True,
            "zerolinecolor": "#9aa7ad",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        font={"family": "Segoe UI, Arial, sans-serif", "color": TEXT_MAIN},
    )

    return figure


def build_selection_summary(
    variable: str,
    selected_ids: list[int],
    apply_quality_mask: bool,
    rmse_threshold: float | None,
) -> list:
    series = VIEWER_DATA.series_by_variable[variable]
    visible_ids = quality_pixel_ids(variable, apply_quality_mask, rmse_threshold)
    selected_ids_clean = clean_selected_ids(selected_ids, visible_ids)
    total_pixels = len(VIEWER_DATA.pixel_lons)
    rmse_range = VIEWER_DATA.rmse_ranges.get(variable)
    if apply_quality_mask and rmse_range is not None:
        threshold = DEFAULT_RMSE_THRESHOLD if rmse_threshold is None else float(rmse_threshold)
        quality_text = f"{len(visible_ids):,} / {total_pixels:,}"
        threshold_text = f"RMSE <= {threshold:.2f} mm"
    elif rmse_range is not None:
        quality_text = f"{total_pixels:,} / {total_pixels:,}"
        threshold_text = "RMSE mask off"
    else:
        quality_text = f"{total_pixels:,} / {total_pixels:,}"
        threshold_text = "No RMSE layer"

    if not selected_ids_clean:
        return [
            html.Div(
                [
                    html.Div("Selected pixels", className="summary-label"),
                    html.Div("0", className="summary-value"),
                ],
                className="summary-item",
            ),
            html.Div(
                [
                    html.Div("Quality pixels", className="summary-label"),
                    html.Div(quality_text, className="summary-value"),
                ],
                className="summary-item",
            ),
            html.Div(
                [
                    html.Div("Quality rule", className="summary-label"),
                    html.Div(threshold_text, className="summary-value"),
                ],
                className="summary-item",
            ),
        ]

    selected_series = series[selected_ids_clean, :]
    mean_series = np.nanmean(selected_series, axis=0)
    final_mean = float(mean_series[-1])
    min_mean = float(np.nanmin(mean_series))
    max_mean = float(np.nanmax(mean_series))

    return [
        html.Div(
            [
                html.Div("Selected pixels", className="summary-label"),
                html.Div(f"{len(selected_ids_clean):,}", className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Final mean", className="summary-label"),
                html.Div(f"{final_mean:.2f} mm", className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Mean range", className="summary-label"),
                html.Div(f"{min_mean:.2f} to {max_mean:.2f} mm", className="summary-value"),
            ],
            className="summary-item",
        ),
        html.Div(
            [
                html.Div("Quality pixels", className="summary-label"),
                html.Div(quality_text, className="summary-value"),
            ],
            className="summary-item",
        ),
    ]


def extract_ids_from_points(points: list[dict]) -> list[int]:
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


def make_app() -> Dash:
    app = Dash(__name__)
    app.title = "InSAR AOI Deformation Viewer"

    variable_options = [
        {"label": VARIABLE_LABELS.get(variable, variable), "value": variable}
        for variable in VIEWER_DATA.variables
    ]
    date_options = [
        {"label": date.strftime("%Y-%m-%d"), "value": index}
        for index, date in enumerate(VIEWER_DATA.dates)
    ]

    app.layout = html.Div(
        [
            dcc.Store(id="selection-store", data=VIEWER_DATA.default_selected_ids),
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("AOI Deformation Explorer"),
                            html.Div(
                                f"Dataset: {VIEWER_DATA.dataset_path.name}",
                                className="subtle-text",
                            ),
                        ],
                        className="title-block",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Displacement source"),
                                    dcc.Dropdown(
                                        id="variable-dropdown",
                                        options=variable_options,
                                        value=VIEWER_DATA.variables[0],
                                        clearable=False,
                                    ),
                                ],
                                className="control-group",
                            ),
                            html.Div(
                                [
                                    html.Label("Map date"),
                                    dcc.Dropdown(
                                        id="date-dropdown",
                                        options=date_options,
                                        value=len(VIEWER_DATA.dates) - 1,
                                        clearable=False,
                                    ),
                                ],
                                className="date-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Quality mask"),
                                    dcc.Checklist(
                                        id="quality-mask-toggle",
                                        options=[
                                            {
                                                "label": "Use RMSE mask",
                                                "value": "rmse",
                                            }
                                        ],
                                        value=["rmse"],
                                        className="quality-checklist",
                                    ),
                                ],
                                className="quality-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Max RMSE"),
                                    dcc.Input(
                                        id="rmse-threshold",
                                        type="number",
                                        value=DEFAULT_RMSE_THRESHOLD,
                                        min=0,
                                        step=0.01,
                                        className="number-input",
                                    ),
                                ],
                                className="rmse-control",
                            ),
                            html.Button("Clear selection", id="clear-selection", n_clicks=0),
                        ],
                        className="controls",
                    ),
                ],
                className="topbar",
            ),
            html.Div(id="selection-summary", className="summary-strip"),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id="deformation-map",
                            config={
                                "displaylogo": False,
                                "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                                "toImageButtonOptions": {"filename": "aoi_deformation_map"},
                            },
                            className="graph",
                        ),
                        className="panel map-panel",
                    ),
                    html.Div(
                        dcc.Graph(
                            id="timeseries-plot",
                            config={
                                "displaylogo": False,
                                "toImageButtonOptions": {"filename": "aoi_deformation_timeseries"},
                            },
                            className="graph",
                        ),
                        className="panel series-panel",
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
                background: #f5f7f8;
                color: #172126;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            .app-shell {
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            .topbar {
                display: grid;
                grid-template-columns: minmax(260px, 420px) 1fr;
                gap: 22px;
                align-items: center;
                padding: 18px 22px 12px;
                border-bottom: 1px solid #dce4e7;
                background: #ffffff;
            }
            .title-block h1 {
                margin: 0;
                font-size: 24px;
                font-weight: 650;
                letter-spacing: 0;
            }
            .subtle-text {
                margin-top: 4px;
                color: #5d6970;
                font-size: 13px;
            }
            .controls {
                display: grid;
                grid-template-columns: minmax(185px, 220px) minmax(150px, 180px) minmax(150px, 170px) minmax(110px, 130px) minmax(140px, 170px);
                gap: 16px;
                align-items: end;
            }
            .control-group label,
            .date-control label,
            .quality-control label,
            .rmse-control label {
                display: block;
                margin-bottom: 7px;
                color: #435058;
                font-size: 12px;
                font-weight: 650;
                text-transform: uppercase;
            }
            .date-control {
                min-width: 0;
            }
            .quality-control,
            .rmse-control {
                min-width: 0;
            }
            .quality-checklist label {
                margin: 0;
                color: #172126;
                font-size: 13px;
                font-weight: 600;
                text-transform: none;
                white-space: nowrap;
            }
            .quality-checklist input {
                margin-right: 7px;
            }
            .number-input {
                width: 100%;
                height: 38px;
                padding: 0 10px;
                border: 1px solid #b8c6cb;
                border-radius: 6px;
                color: #172126;
                font: inherit;
            }
            #clear-selection {
                height: 38px;
                padding: 0 16px;
                border: 1px solid #b8c6cb;
                border-radius: 6px;
                background: #ffffff;
                color: #172126;
                font-weight: 650;
                cursor: pointer;
            }
            #clear-selection:hover {
                border-color: #126a65;
                color: #126a65;
            }
            .summary-strip {
                display: flex;
                gap: 10px;
                padding: 10px 22px;
                border-bottom: 1px solid #dce4e7;
                background: #eef3f4;
                min-height: 64px;
            }
            .summary-item {
                min-width: 150px;
                padding: 7px 11px;
                border: 1px solid #d5e0e4;
                border-radius: 6px;
                background: #ffffff;
            }
            .summary-label {
                color: #5d6970;
                font-size: 12px;
                font-weight: 650;
                text-transform: uppercase;
            }
            .summary-value {
                margin-top: 3px;
                color: #172126;
                font-size: 18px;
                font-weight: 700;
            }
            .workspace {
                display: grid;
                grid-template-columns: minmax(420px, 1.02fr) minmax(440px, 0.98fr);
                gap: 14px;
                flex: 1;
                min-height: 0;
                padding: 14px;
            }
            .panel {
                min-width: 0;
                min-height: calc(100vh - 190px);
                border: 1px solid #dce4e7;
                border-radius: 8px;
                background: #ffffff;
                overflow: hidden;
            }
            .graph {
                height: 100%;
                min-height: calc(100vh - 192px);
            }
            .Select-control {
                border-color: #b8c6cb;
                border-radius: 6px;
            }
            @media (max-width: 1050px) {
                .topbar,
                .controls,
                .workspace {
                    grid-template-columns: 1fr;
                }
                .panel,
                .graph {
                    min-height: 520px;
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
        Input("deformation-map", "clickData"),
        Input("deformation-map", "selectedData"),
        Input("clear-selection", "n_clicks"),
        State("selection-store", "data"),
        prevent_initial_call=True,
    )
    def update_selection(click_data, selected_data, _clear_clicks, current_selection):
        triggered = ctx.triggered[0]["prop_id"].split(".")[-1] if ctx.triggered else None

        if ctx.triggered_id == "clear-selection":
            return []

        if triggered == "clickData" and click_data and click_data.get("points"):
            ids = extract_ids_from_points(click_data["points"])
            return ids[:1] if ids else (current_selection or [])

        if triggered == "selectedData" and selected_data and selected_data.get("points"):
            ids = extract_ids_from_points(selected_data["points"])
            return ids if ids else (current_selection or [])

        return current_selection or []

    @app.callback(
        Output("deformation-map", "figure"),
        Output("timeseries-plot", "figure"),
        Output("selection-summary", "children"),
        Input("variable-dropdown", "value"),
        Input("date-dropdown", "value"),
        Input("quality-mask-toggle", "value"),
        Input("rmse-threshold", "value"),
        Input("selection-store", "data"),
    )
    def update_figures(variable, date_index, quality_mask_values, rmse_threshold, selected_ids):
        selected_ids = selected_ids or []
        date_index = int(date_index)
        apply_quality_mask = "rmse" in (quality_mask_values or [])
        map_figure = build_map_figure(
            variable,
            date_index,
            selected_ids,
            apply_quality_mask,
            rmse_threshold,
        )
        timeseries_figure = build_timeseries_figure(
            variable,
            selected_ids,
            apply_quality_mask,
            rmse_threshold,
        )
        selection_summary = build_selection_summary(
            variable,
            selected_ids,
            apply_quality_mask,
            rmse_threshold,
        )
        return map_figure, timeseries_figure, selection_summary

    return app


if __name__ == "__main__":
    dash_app = make_app()
    dash_app.run(host="127.0.0.1", port=8050, debug=False)
