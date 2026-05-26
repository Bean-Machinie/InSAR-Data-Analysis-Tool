"""Variable registry — maps NetCDF variable names to display metadata."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class VariableSpec(BaseModel):
    key: str
    display_name: str
    category: Literal["overlay_canvas", "overlay_png", "timeseries_only", "metadata"]
    dimensions: Literal["spatial", "temporal"]
    units: str
    default_colormap: str
    value_range_hint: tuple[float, float] | None = None
    symmetric: bool = False
    legacy_aliases: list[str] = []


# Canvas-rendered variables (dense point cloud via DataCanvas)
_CANVAS: list[VariableSpec] = [
    VariableSpec(
        key="sbas_velocity_masked",
        display_name="Velocity — masked (mm/yr)",
        category="overlay_canvas",
        dimensions="spatial",
        units="mm/yr",
        default_colormap="RdBu_r",
        value_range_hint=(-20.0, 20.0),
        symmetric=True,
        legacy_aliases=["velocity_sbas"],
    ),
    VariableSpec(
        key="sbas_velocity_raw",
        display_name="Velocity — raw (mm/yr)",
        category="overlay_canvas",
        dimensions="spatial",
        units="mm/yr",
        default_colormap="RdBu_r",
        value_range_hint=(-20.0, 20.0),
        symmetric=True,
    ),
    VariableSpec(
        key="sbas_displacement_masked",
        display_name="Displacement — masked (mm)",
        category="overlay_canvas",
        dimensions="temporal",
        units="mm",
        default_colormap="RdBu_r",
        value_range_hint=(-50.0, 50.0),
        symmetric=True,
        legacy_aliases=["displacement_sbas"],
    ),
]

# Server-rendered PNG overlays
_PNG: list[VariableSpec] = [
    VariableSpec(
        key="coherence_median",
        display_name="Coherence (median)",
        category="overlay_png",
        dimensions="spatial",
        units="",
        default_colormap="viridis",
        value_range_hint=(0.0, 1.0),
    ),
    VariableSpec(
        key="coherence_mean",
        display_name="Coherence (mean)",
        category="overlay_png",
        dimensions="spatial",
        units="",
        default_colormap="viridis",
        value_range_hint=(0.0, 1.0),
    ),
    VariableSpec(
        key="valid_pixel_mask",
        display_name="Valid pixel mask",
        category="overlay_png",
        dimensions="spatial",
        units="",
        default_colormap="mask",
        value_range_hint=(0.0, 1.0),
    ),
    VariableSpec(
        key="dem",
        display_name="DEM (elevation, m)",
        category="overlay_png",
        dimensions="spatial",
        units="m",
        default_colormap="terrain",
    ),
    VariableSpec(
        key="sbas_rmse_masked",
        display_name="RMSE — masked (mm)",
        category="overlay_png",
        dimensions="spatial",
        units="mm",
        default_colormap="hot_r",
        legacy_aliases=["rmse_sbas"],
    ),
    VariableSpec(
        key="sbas_rmse_raw",
        display_name="RMSE — raw (mm)",
        category="overlay_png",
        dimensions="spatial",
        units="mm",
        default_colormap="hot_r",
    ),
    VariableSpec(
        key="psf",
        display_name="PSF",
        category="overlay_png",
        dimensions="spatial",
        units="",
        default_colormap="viridis",
    ),
]

# Time-series only (pixel queries, not rendered as map layers)
_TIMESERIES: list[VariableSpec] = [
    VariableSpec(
        key="sbas_displacement_raw",
        display_name="Displacement — raw (mm)",
        category="timeseries_only",
        dimensions="temporal",
        units="mm",
        default_colormap="RdBu_r",
        symmetric=True,
    ),
    VariableSpec(
        key="sbas_displacement_segmented_same_pixel",
        display_name="Displacement — segmented (mm)",
        category="timeseries_only",
        dimensions="temporal",
        units="mm",
        default_colormap="RdBu_r",
        symmetric=True,
    ),
    VariableSpec(
        key="sbas_segment_id",
        display_name="Segment ID",
        category="timeseries_only",
        dimensions="temporal",
        units="",
        default_colormap="tab10",
    ),
    VariableSpec(
        key="sbas_valid_time_mask",
        display_name="Valid time mask",
        category="timeseries_only",
        dimensions="temporal",
        units="",
        default_colormap="binary",
    ),
    VariableSpec(
        key="coherence_per_date",
        display_name="Coherence per date",
        category="timeseries_only",
        dimensions="temporal",
        units="",
        default_colormap="viridis",
    ),
]

ALL_SPECS: list[VariableSpec] = _CANVAS + _PNG + _TIMESERIES

_BY_KEY: dict[str, VariableSpec] = {s.key: s for s in ALL_SPECS}

# Build alias → canonical key map
_ALIAS_MAP: dict[str, str] = {}
for _spec in ALL_SPECS:
    for _alias in _spec.legacy_aliases:
        _ALIAS_MAP[_alias] = _spec.key


def resolve_key(name: str) -> str:
    """Return canonical key for a variable name, resolving legacy aliases."""
    if name in _BY_KEY:
        return name
    return _ALIAS_MAP.get(name, name)


def get_spec(key: str) -> VariableSpec | None:
    canonical = resolve_key(key)
    return _BY_KEY.get(canonical)


def canvas_keys() -> list[str]:
    return [s.key for s in _CANVAS]


def png_keys() -> list[str]:
    return [s.key for s in _PNG]
