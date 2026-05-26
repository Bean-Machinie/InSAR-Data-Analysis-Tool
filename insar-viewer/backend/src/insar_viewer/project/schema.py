"""Pydantic v2 models for parameters.json and manifest.json."""
from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AoiParams(BaseModel):
    raw_wkt: str
    buffer_deg: float = 0.05
    tight_pad_deg: float = 0.005


class TimeWindow(BaseModel):
    start: str
    end: str
    days: int | None = None


class ScenesParams(BaseModel):
    count: int
    dates: list[str]


class PairsParams(BaseModel):
    initial: int | None = None
    best_used: int | None = None


class ProcessingParams(BaseModel):
    model_config = {"extra": "allow"}

    geocode_res_m: float | None = None
    sbas_days: int | None = None
    sbas_wavelength_m: float | None = None
    ps_wavelength_m: float | None = None


class PoiEntry(BaseModel):
    name: str
    lon: float
    lat: float


class Parameters(BaseModel):
    """Parsed parameters.json."""

    model_config = {"extra": "allow"}

    project: str
    orbit: str
    aoi: AoiParams
    time_window: TimeWindow
    scenes: ScenesParams
    pygmtsar_version: str | None = None
    processing_date: str | None = None
    pairs: PairsParams | None = None
    processing: ProcessingParams | None = None
    pois: list[PoiEntry] = Field(default_factory=list)


# ── Manifest ──────────────────────────────────────────────────────────────────

class VariableShape(BaseModel):
    model_config = {"extra": "allow"}

    lat: int | None = None
    lon: int | None = None
    date: int | None = None


class VariableMeta(BaseModel):
    model_config = {"extra": "allow"}

    units: str = ""
    description: str = ""
    shape: VariableShape | None = None


class ManifestExtents(BaseModel):
    wide_bounds_lonlat: list[float] | None = None
    tight_bounds_lonlat: list[float] | None = None


class Manifest(BaseModel):
    """Parsed manifest.json."""

    model_config = {"extra": "allow"}

    project: str
    orbit: str
    created: str | None = None
    crs: str = "EPSG:4326"
    variables: dict[str, VariableMeta] = Field(default_factory=dict)
    extents: ManifestExtents | None = None
    files: dict[str, Any] = Field(default_factory=dict)
