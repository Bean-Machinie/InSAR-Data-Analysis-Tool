"""
Create AOI-only InSAR outputs from the exported project folder.

This script reads the AOI polygon from parameters.json, masks NetCDF variables
and GeoTIFFs to that polygon, and creates simple AOI-only quicklook plots.

Dependency notes:
    pip install xarray netcdf4 rioxarray rasterio pandas matplotlib numpy
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape


PROJECT_DIR = Path(r"E:\Scripts\InSAR-Data-Analysis-Tool\Data\project_dam_D")
PARAMETERS_PATH = PROJECT_DIR / "parameters.json"
GEOTIFF_DIR = PROJECT_DIR / "geotiffs"
OUTPUT_DIR = PROJECT_DIR / "aoi_only"
OUTPUT_GEOTIFF_DIR = OUTPUT_DIR / "geotiffs"

PREFERRED_NETCDF_INPUTS = (
    PROJECT_DIR / "results_wide.nc",
    PROJECT_DIR / "results_tight.nc",
)

MAP_VARIABLES = (
    "velocity_sbas",
    "velocity_ps",
    "rmse_sbas",
    "rmse_ps",
    "dem",
    "psf",
)

DISPLACEMENT_VARIABLES = (
    "displacement_sbas",
    "displacement_ps",
)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_parameters(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Could not find parameters file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_polygon_wkt(wkt: str) -> list[tuple[float, float]]:
    """Parse a simple POLYGON WKT into lon/lat coordinate pairs."""
    match = re.match(r"^\s*POLYGON\s*\(\((.+)\)\)\s*$", wkt, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Only simple POLYGON WKT is supported. Got: {wkt[:80]}")

    coordinates = []
    for point in match.group(1).split(","):
        parts = point.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid WKT coordinate: {point}")
        lon, lat = float(parts[0]), float(parts[1])
        coordinates.append((lon, lat))

    if len(coordinates) < 4:
        raise ValueError("AOI polygon must contain at least 4 coordinates.")

    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])

    return coordinates


def polygon_bounds(coordinates: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [lon for lon, _lat in coordinates]
    lats = [lat for _lon, lat in coordinates]
    return min(lons), min(lats), max(lons), max(lats)


def polygon_to_geojson(coordinates: list[tuple[float, float]]) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[list(point) for point in coordinates]],
    }


def find_netcdf_input() -> Path | None:
    for path in PREFERRED_NETCDF_INPUTS:
        if path.exists():
            return path

    nc_files = sorted(PROJECT_DIR.glob("*.nc"))
    return nc_files[0] if nc_files else None


def coord_slice(values, lower: float, upper: float) -> slice:
    if len(values) == 0:
        return slice(lower, upper)
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def make_aoi_mask(dataset, coordinates: list[tuple[float, float]]):
    import numpy as np
    import xarray as xr
    from matplotlib.path import Path as MatplotlibPath

    lon_values = dataset["lon"].values
    lat_values = dataset["lat"].values
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    polygon_path = MatplotlibPath(coordinates)
    inside = polygon_path.contains_points(points, radius=1e-12)
    inside = inside.reshape((len(lat_values), len(lon_values)))

    return xr.DataArray(
        inside,
        dims=("lat", "lon"),
        coords={"lat": dataset["lat"], "lon": dataset["lon"]},
        name="aoi_mask",
    )


def finite_min_max(data_array) -> tuple[float | None, float | None]:
    import numpy as np

    values = data_array.values
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        return None, None

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None, None

    return float(finite_values.min()), float(finite_values.max())


def print_variable_summary(dataset) -> None:
    print("\nAOI-only variable min/max:")
    for name in MAP_VARIABLES + DISPLACEMENT_VARIABLES:
        if name not in dataset:
            continue
        min_value, max_value = finite_min_max(dataset[name])
        if min_value is None or max_value is None:
            print(f"  {name}: no finite AOI pixels")
        else:
            print(f"  {name}: min={min_value:.6g}, max={max_value:.6g}")


def clip_netcdf_to_aoi(coordinates: list[tuple[float, float]]) -> Path | None:
    import xarray as xr

    source_path = find_netcdf_input()
    if source_path is None:
        print("No NetCDF file found. Skipping NetCDF AOI clipping.")
        return None

    output_path = OUTPUT_DIR / "results_aoi_masked.nc"
    lon_min, lat_min, lon_max, lat_max = polygon_bounds(coordinates)

    print_section("NetCDF AOI clipping")
    print(f"Input:  {source_path}")
    print(f"Output: {output_path}")
    print(f"AOI bounds: lon {lon_min:.8f} to {lon_max:.8f}, lat {lat_min:.8f} to {lat_max:.8f}")

    try:
        with xr.open_dataset(source_path) as dataset:
            subset = dataset.sel(
                lon=coord_slice(dataset["lon"].values, lon_min, lon_max),
                lat=coord_slice(dataset["lat"].values, lat_min, lat_max),
            )

            if subset.sizes.get("lat", 0) == 0 or subset.sizes.get("lon", 0) == 0:
                print("AOI bounds do not overlap the NetCDF coordinates.")
                return None

            aoi_mask = make_aoi_mask(subset, coordinates)
            masked = subset.copy()

            for name, data_array in subset.data_vars.items():
                if {"lat", "lon"}.issubset(data_array.dims):
                    masked[name] = data_array.where(aoi_mask)

            masked["aoi_mask"] = aoi_mask.astype("uint8")
            masked["aoi_mask"].attrs.update(
                {
                    "description": "1 inside AOI polygon, 0 outside AOI polygon",
                    "flag_values": "0, 1",
                    "flag_meanings": "outside_aoi inside_aoi",
                }
            )
            masked.attrs["aoi_source"] = str(PARAMETERS_PATH)
            masked.attrs["aoi_wkt"] = load_parameters(PARAMETERS_PATH)["aoi"]["raw_wkt"]
            masked.attrs["aoi_clip_note"] = (
                "Variables with lat/lon dimensions are masked to the AOI polygon."
            )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            masked.to_netcdf(output_path)

            print(f"Subset dimensions: {dict(masked.sizes)}")
            inside_count = int(masked["aoi_mask"].sum().item())
            total_count = int(masked["aoi_mask"].size)
            print(f"AOI pixels: {inside_count} of {total_count}")
            print_variable_summary(masked)

        return output_path
    except Exception as exc:
        print(f"Could not clip NetCDF to AOI: {exc}")
        return None


def raster_min_max(data, nodata) -> tuple[float | None, float | None]:
    import numpy as np

    values = data.astype("float64", copy=False)
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None, None

    return float(finite_values.min()), float(finite_values.max())


def choose_raster_nodata(source) -> float | int | None:
    import numpy as np

    if source.nodata is not None:
        return source.nodata

    dtype = np.dtype(source.dtypes[0])
    if np.issubdtype(dtype, np.floating):
        return np.nan

    return 0


def clip_geotiff(path: Path, geometry: dict) -> Path | None:
    import rasterio
    from rasterio.mask import mask

    output_path = OUTPUT_GEOTIFF_DIR / path.name

    try:
        with rasterio.open(path) as source:
            nodata = choose_raster_nodata(source)
            data, transform = mask(
                source,
                [geometry],
                crop=True,
                filled=True,
                nodata=nodata,
            )

            profile = source.profile.copy()
            profile.update(
                {
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                    "nodata": nodata,
                }
            )

            OUTPUT_GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **profile) as destination:
                destination.write(data)

            min_value, max_value = raster_min_max(data[0], nodata)
            if min_value is None or max_value is None:
                print(f"  {path.name}: wrote {output_path.name}, no finite AOI pixels")
            else:
                print(
                    f"  {path.name}: wrote {output_path.name}, "
                    f"min={min_value:.6g}, max={max_value:.6g}"
                )

        return output_path
    except Exception as exc:
        print(f"  {path.name}: could not clip GeoTIFF: {exc}")
        return None


def clip_geotiffs_to_aoi(coordinates: list[tuple[float, float]]) -> list[Path]:
    print_section("GeoTIFF AOI clipping")

    geotiff_files = sorted(
        path for path in GEOTIFF_DIR.iterdir() if path.suffix.lower() in {".tif", ".tiff"}
    ) if GEOTIFF_DIR.exists() else []

    if not geotiff_files:
        print("No GeoTIFF files found. Skipping GeoTIFF AOI clipping.")
        return []

    geometry = polygon_to_geojson(coordinates)
    written_paths = []
    print(f"Clipping {len(geotiff_files)} GeoTIFF files to: {OUTPUT_GEOTIFF_DIR}")
    for path in geotiff_files:
        output_path = clip_geotiff(path, geometry)
        if output_path is not None:
            written_paths.append(output_path)

    return written_paths


def save_aoi_geojson(coordinates: list[tuple[float, float]]) -> Path:
    output_path = OUTPUT_DIR / "aoi_polygon.geojson"
    feature_collection = {
        "type": "FeatureCollection",
        "name": "aoi_polygon",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "project_dam_D_AOI",
                    "source": str(PARAMETERS_PATH),
                    "crs": "EPSG:4326",
                },
                "geometry": polygon_to_geojson(coordinates),
            }
        ],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(feature_collection, file, indent=2)

    print(f"Saved QGIS AOI polygon: {output_path}")
    return output_path


def raster_file_min_max(path: Path) -> tuple[float | None, float | None]:
    import rasterio

    with rasterio.open(path) as source:
        data = source.read(1, masked=False)
        return raster_min_max(data, source.nodata)


def save_qgis_velocity_style(raster_path: Path, label: str) -> Path | None:
    try:
        min_value, max_value = raster_file_min_max(raster_path)
    except Exception as exc:
        print(f"Could not read raster for QGIS style {raster_path.name}: {exc}")
        return None

    if min_value is None or max_value is None:
        print(f"Could not create QGIS style for {raster_path.name}: no finite values.")
        return None

    max_abs = max(abs(min_value), abs(max_value))
    if max_abs == 0:
        max_abs = 1.0

    output_path = raster_path.with_suffix(".qml")
    label = escape(label)

    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer opacity="1" alphaBand="-1" band="1" type="singlebandpseudocolor" classificationMin="{-max_abs:.6f}" classificationMax="{max_abs:.6f}">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0">
          <item value="{-max_abs:.6f}" label="{-max_abs:.2f} mm/year" color="#2166ac" alpha="255"/>
          <item value="0.000000" label="0.00 mm/year" color="#f7f7f7" alpha="255"/>
          <item value="{max_abs:.6f}" label="{max_abs:.2f} mm/year" color="#b2182b" alpha="255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeOn="0" grayscaleMode="0" saturation="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <legend type="default-vector"/>
  <blendMode>0</blendMode>
  <layerOpacity>1</layerOpacity>
  <customproperties>
    <Option type="Map">
      <Option name="insar_units" type="QString" value="mm/year"/>
      <Option name="insar_style_note" type="QString" value="{label}: negative is blue, positive is red, zero is white"/>
    </Option>
  </customproperties>
</qgis>
"""

    output_path.write_text(qml, encoding="utf-8")
    print(f"Saved QGIS style: {output_path}")
    return output_path


def save_qgis_support_files(
    coordinates: list[tuple[float, float]],
    geotiff_paths: list[Path],
) -> None:
    print_section("QGIS support files")
    save_aoi_geojson(coordinates)

    by_name = {path.name.lower(): path for path in geotiff_paths}
    for filename, label in (
        ("velocity_sbas.tif", "AOI SBAS velocity"),
        ("velocity_ps.tif", "AOI PS velocity"),
    ):
        raster_path = by_name.get(filename)
        if raster_path is None:
            print(f"QGIS style skipped: {filename} was not written.")
            continue
        save_qgis_velocity_style(raster_path, label)


def plot_aoi_velocity(netcdf_path: Path | None) -> Path | None:
    if netcdf_path is None:
        print("No AOI NetCDF available for velocity quicklook.")
        return None

    import matplotlib.pyplot as plt
    import xarray as xr

    output_path = OUTPUT_DIR / "aoi_velocity_sbas.png"

    try:
        with xr.open_dataset(netcdf_path) as dataset:
            if "velocity_sbas" not in dataset:
                print("velocity_sbas not found. Skipping AOI velocity quicklook.")
                return None

            velocity = dataset["velocity_sbas"]
            min_value, max_value = finite_min_max(velocity)
            if min_value is None or max_value is None:
                print("velocity_sbas has no finite AOI pixels. Skipping quicklook.")
                return None

            max_abs = max(abs(min_value), abs(max_value))

            plt.figure(figsize=(8, 6))
            velocity.plot(
                cmap="RdBu_r",
                vmin=-max_abs,
                vmax=max_abs,
                cbar_kwargs={"label": "SBAS velocity (mm/year)"},
            )
            plt.title("AOI-only SBAS velocity")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

        print(f"Saved AOI velocity quicklook: {output_path}")
        return output_path
    except Exception as exc:
        print(f"Could not create AOI velocity quicklook: {exc}")
        return None


def save_mean_displacement_timeseries(netcdf_path: Path | None) -> Path | None:
    if netcdf_path is None:
        return None

    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr

    output_csv = OUTPUT_DIR / "aoi_mean_displacement_timeseries.csv"
    output_png = OUTPUT_DIR / "aoi_mean_displacement_timeseries.png"

    try:
        with xr.open_dataset(netcdf_path) as dataset:
            rows = {}
            if "date" not in dataset.coords:
                print("No date coordinate found. Skipping AOI mean displacement time series.")
                return None

            dates = pd.to_datetime(dataset["date"].values)
            rows["date"] = dates

            for name in DISPLACEMENT_VARIABLES:
                if name in dataset:
                    rows[f"{name}_mean_mm"] = dataset[name].mean(
                        dim=("lat", "lon"),
                        skipna=True,
                    ).values

        table = pd.DataFrame(rows)
        if len(table.columns) <= 1:
            print("No displacement variables found. Skipping AOI mean displacement time series.")
            return None

        table.to_csv(output_csv, index=False)

        plt.figure(figsize=(8, 4))
        for column in table.columns:
            if column == "date":
                continue
            plt.plot(table["date"], table[column], marker="o", label=column)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Date")
        plt.ylabel("Mean displacement (mm)")
        plt.title("AOI mean displacement")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        plt.close()

        print(f"Saved AOI mean displacement CSV: {output_csv}")
        print(f"Saved AOI mean displacement plot: {output_png}")
        return output_csv
    except Exception as exc:
        print(f"Could not create AOI mean displacement time series: {exc}")
        return None


def main() -> None:
    print_section("AOI-only InSAR clipping")
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Output folder:  {OUTPUT_DIR}")

    try:
        parameters = load_parameters(PARAMETERS_PATH)
        raw_wkt = parameters["aoi"]["raw_wkt"]
        coordinates = parse_polygon_wkt(raw_wkt)
    except Exception as exc:
        print(f"Could not load AOI polygon: {exc}")
        return

    print(f"AOI polygon vertices: {len(coordinates)}")
    print(f"AOI WKT: {raw_wkt}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aoi_netcdf_path = clip_netcdf_to_aoi(coordinates)
    aoi_geotiff_paths = clip_geotiffs_to_aoi(coordinates)
    save_qgis_support_files(coordinates, aoi_geotiff_paths)
    plot_aoi_velocity(aoi_netcdf_path)
    save_mean_displacement_timeseries(aoi_netcdf_path)


if __name__ == "__main__":
    main()
