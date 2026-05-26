"""NetCDF dataset loading with eager in-memory load."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ValidationError

from .discovery import find_product_dir
from .registry import ALL_SPECS, canvas_keys, png_keys, resolve_key
from .schema import Manifest, Parameters

logger = logging.getLogger(__name__)

PREFERRED_NC = ("results_tight.nc", "results_aoi_masked.nc", "results_wide.nc")

# Variables fetched only for pixel-level time-series queries (not rendered)
PIXEL_SERIES_VARS: dict[str, str] = {
    "raw": "sbas_displacement_raw",
    "segmented": "sbas_displacement_segmented_same_pixel",
    "segment_id": "sbas_segment_id",
    "valid_time_mask": "sbas_valid_time_mask",
    "coh_per_date": "coherence_per_date",
    # Legacy aliases
    "raw_legacy": "displacement_sbas",
}


def _default_nc(product_dir: Path) -> Path:
    for name in PREFERRED_NC:
        p = product_dir / name
        if p.exists():
            return p
    ncs = sorted(product_dir.glob("*.nc"))
    if ncs:
        return ncs[0]
    raise FileNotFoundError(f"No NetCDF file found under {product_dir}")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _parse_wkt_coords(wkt: str | None) -> tuple[list[float], list[float]] | None:
    """Parse POLYGON WKT → (lons, lats). Returns None if unparseable."""
    if not wkt:
        return None
    m = re.match(r"^\s*POLYGON\s*\(\((.+)\)\)\s*$", wkt, re.IGNORECASE)
    if not m:
        return None
    lons, lats = [], []
    for pt in m.group(1).split(","):
        parts = pt.strip().split()
        if len(parts) < 2:
            return None
        lons.append(float(parts[0]))
        lats.append(float(parts[1]))
    if lons and (lons[0] != lons[-1] or lats[0] != lats[-1]):
        lons.append(lons[0])
        lats.append(lats[0])
    return lons, lats


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        d = 0.0001
        return np.array([values[0] - d / 2, values[0] + d / 2])
    midpoints = (values[:-1] + values[1:]) / 2
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def _robust_range(
    arr: np.ndarray, mask: np.ndarray, symmetric: bool
) -> tuple[float, float]:
    flat = arr[:, mask] if arr.ndim == 3 else arr[mask]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        limit = float(np.nanpercentile(np.abs(flat), 98))
        if not np.isfinite(limit) or limit == 0:
            limit = 1.0
        return -limit, limit
    lo = float(np.nanpercentile(flat, 2))
    hi = float(np.nanpercentile(flat, 98))
    if lo == hi:
        lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


class LoadedProject:
    """All data for a single loaded project."""

    def __init__(
        self,
        root_dir: Path,
        product_dir: Path,
        nc_path: Path,
        parameters: Parameters,
        manifest: Manifest | None,
        dataset: xr.Dataset,
    ) -> None:
        self.root_dir = root_dir
        self.product_dir = product_dir
        self.nc_path = nc_path
        self.parameters = parameters
        self.manifest = manifest

        if "lat" not in dataset.coords or "lon" not in dataset.coords:
            raise ValueError("Dataset must have lat/lon coordinates.")

        ds = dataset.sortby("lat").sortby("lon")

        self.latitudes: np.ndarray = ds["lat"].values.astype("float64")
        self.longitudes: np.ndarray = ds["lon"].values.astype("float64")
        self.lat_edges = _coordinate_edges(self.latitudes)
        self.lon_edges = _coordinate_edges(self.longitudes)

        if "date" in ds.coords:
            self.dates: pd.DatetimeIndex = pd.DatetimeIndex(
                pd.to_datetime(ds["date"].values)
            )
        else:
            self.dates = pd.DatetimeIndex([])

        # Build spatial mask — broadest extent of processed pixels
        self.spatial_mask: np.ndarray = self._build_mask(ds)

        # Load canvas + PNG layer arrays
        self.static_values: dict[str, np.ndarray] = {}
        self.temporal_values: dict[str, np.ndarray] = {}
        self.layer_ranges: dict[str, tuple[float, float]] = {}
        self.available_keys: list[str] = []

        all_render_keys = canvas_keys() + png_keys()
        for spec in ALL_SPECS:
            if spec.key not in all_render_keys and spec.key not in (
                list(PIXEL_SERIES_VARS.values())
            ):
                continue
            varname = self._find_varname(ds, spec.key, spec.legacy_aliases)
            if varname is None:
                continue
            var = ds[varname]
            if spec.dimensions == "temporal":
                arr = var.transpose("date", "lat", "lon").values.astype("float64")
                self.temporal_values[spec.key] = arr
            else:
                arr = var.transpose("lat", "lon").values.astype("float64")
                self.static_values[spec.key] = arr
            if spec.key in all_render_keys:
                self.available_keys.append(spec.key)
                if spec.category == "metadata":
                    self.layer_ranges[spec.key] = (0.0, 1.0)
                else:
                    self.layer_ranges[spec.key] = _robust_range(
                        arr, self.spatial_mask, spec.symmetric
                    )

        # Pixel-series arrays keyed by role name
        self.pixel_series: dict[str, np.ndarray] = {}
        for role, varname in PIXEL_SERIES_VARS.items():
            if varname in ds.data_vars:
                var = ds[varname]
                if "date" not in var.dims:
                    continue
                arr = var.transpose("date", "lat", "lon").values
                arr = arr.astype("float64") if arr.dtype.kind == "f" else arr.astype("int32")
                # De-duplicate by role (raw wins over raw_legacy)
                base_role = role.replace("_legacy", "")
                if base_role not in self.pixel_series:
                    self.pixel_series[base_role] = arr

        mask_rows, mask_cols = np.where(self.spatial_mask)
        self.center_lat = float(np.nanmean(self.latitudes[mask_rows]))
        self.center_lon = float(np.nanmean(self.longitudes[mask_cols]))

        # AOI polygon from parameters
        wkt = (parameters.aoi.raw_wkt if parameters.aoi else None)
        coords = _parse_wkt_coords(wkt)
        self.aoi_lons: list[float] | None = coords[0] if coords else None
        self.aoi_lats: list[float] | None = coords[1] if coords else None

    @staticmethod
    def _find_varname(
        ds: xr.Dataset, key: str, aliases: list[str]
    ) -> str | None:
        for name in [key] + aliases:
            if name in ds.data_vars:
                return name
        return None

    def _build_mask(self, ds: xr.Dataset) -> np.ndarray:
        for varname in ("sbas_velocity_raw", "sbas_velocity_masked", "velocity_sbas"):
            if varname in ds.data_vars:
                arr = ds[varname].transpose("lat", "lon").values
                mask = np.isfinite(arr)
                if np.any(mask):
                    return mask
        # Fallback: union of all finite pixels
        candidates: list[np.ndarray] = []
        for vname in ds.data_vars:
            var = ds[vname]
            if "lat" not in var.dims or "lon" not in var.dims:
                continue
            if "date" in var.dims:
                arr = var.transpose("date", "lat", "lon").values
                candidates.append(np.isfinite(arr).any(axis=0))
            else:
                arr = var.transpose("lat", "lon").values
                candidates.append(np.isfinite(arr))
        if not candidates:
            raise ValueError("No spatial data in dataset.")
        return np.logical_or.reduce(candidates)


def load_project(root_dir: Path) -> LoadedProject:
    """Discover product folder, load dataset, return LoadedProject."""
    root_dir = root_dir.expanduser().resolve()
    product_dir = find_product_dir(root_dir)

    params_path = product_dir / "parameters.json"
    manifest_path = product_dir / "manifest.json"

    raw_params = _load_json(params_path)
    if not raw_params:
        raise ValueError(
            f"parameters.json not found or empty at {params_path}. "
            "Is this a valid InSAR product folder?"
        )
    try:
        parameters = Parameters.model_validate(raw_params)
    except ValidationError as exc:
        raise ValueError(f"parameters.json is invalid: {exc}") from exc

    manifest: Manifest | None = None
    raw_manifest = _load_json(manifest_path)
    if raw_manifest:
        try:
            manifest = Manifest.model_validate(raw_manifest)
        except ValidationError:
            logger.warning("manifest.json could not be parsed — ignoring.")

    nc_path = _default_nc(product_dir)
    logger.info("Loading dataset: %s", nc_path)

    with xr.open_dataset(nc_path) as raw_ds:
        dataset = raw_ds.load()

    return LoadedProject(
        root_dir=root_dir,
        product_dir=product_dir,
        nc_path=nc_path,
        parameters=parameters,
        manifest=manifest,
        dataset=dataset,
    )
