"""
Simple inspection script for exported InSAR results.

Dependency notes:
    pip install xarray netcdf4 rioxarray rasterio pandas matplotlib numpy
"""

from pathlib import Path


PROJECT_DIR = Path(r"E:\Scripts\InSAR-Data-Analysis-Tool\Data\project_dam_D")
GEOTIFF_DIR = PROJECT_DIR / "geotiffs"
TIMESERIES_DIR = PROJECT_DIR / "timeseries"
OUTPUT_DIR = Path(__file__).resolve().parent

IMPORTANT_NETCDF_VARIABLES = (
    "velocity_sbas",
    "velocity_ps",
    "displacement_sbas",
    "displacement_ps",
    "rmse_sbas",
    "rmse_ps",
)

PREFERRED_NETCDF_NAMES = (
    "results_tight.nc",
    "results_AOI_clipped.nc",
)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "unknown size"

    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def list_files(folder: Path, label: str) -> list[Path]:
    print_section(label)

    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        return []

    if not folder.is_dir():
        print(f"Path exists but is not a folder: {folder}")
        return []

    files = sorted(path for path in folder.iterdir() if path.is_file())
    folders = sorted(path for path in folder.iterdir() if path.is_dir())

    if folders:
        print("Subfolders:")
        for path in folders:
            print(f"  [dir]  {path.name}")

    if files:
        print("Files:")
        for path in files:
            print(f"  {path.name} ({format_size(path)})")
    else:
        print("No files found.")

    return files


def prioritize_netcdf_files(files: list[Path]) -> list[Path]:
    nc_files = [path for path in files if path.suffix.lower() == ".nc"]
    by_name = {path.name.lower(): path for path in nc_files}

    ordered: list[Path] = []
    for name in PREFERRED_NETCDF_NAMES:
        path = by_name.get(name.lower())
        if path is not None:
            ordered.append(path)

    ordered_names = {path.name.lower() for path in ordered}
    ordered.extend(
        sorted(path for path in nc_files if path.name.lower() not in ordered_names)
    )
    return ordered


def print_xarray_crs(dataset) -> None:
    crs_values = []

    try:
        import rioxarray  # noqa: F401

        rio_crs = getattr(getattr(dataset, "rio", None), "crs", None)
        if rio_crs is not None:
            crs_values.append(("rio.crs", str(rio_crs)))
    except ImportError:
        pass
    except Exception as exc:
        print(f"CRS check through rioxarray failed: {exc}")

    for attr_name in ("crs", "crs_wkt", "spatial_ref", "projection"):
        value = dataset.attrs.get(attr_name)
        if value:
            crs_values.append((f"dataset.attrs[{attr_name!r}]", str(value)))

    spatial_ref = dataset.coords.get("spatial_ref")
    if spatial_ref is not None:
        for attr_name in ("crs_wkt", "spatial_ref", "grid_mapping_name"):
            value = spatial_ref.attrs.get(attr_name)
            if value:
                crs_values.append((f"spatial_ref.attrs[{attr_name!r}]", str(value)))

    if crs_values:
        print("CRS:")
        seen = set()
        for source, value in crs_values:
            if (source, value) in seen:
                continue
            seen.add((source, value))
            short_value = value if len(value) <= 500 else value[:500] + "..."
            print(f"  {source}: {short_value}")
    else:
        print("CRS: not found in dataset-level metadata.")


def xarray_min_max(data_array):
    try:
        import numpy as np

        values = data_array.values
        if values.size == 0:
            return None, None

        if not np.issubdtype(values.dtype, np.number):
            return None, None

        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None, None

        return float(finite_values.min()), float(finite_values.max())
    except Exception as exc:
        print(f"  Could not compute min/max for {data_array.name}: {exc}")
        return None, None


def inspect_netcdf(path: Path) -> bool:
    print_section(f"NetCDF: {path.name}")

    try:
        import xarray as xr
    except ImportError:
        print("xarray is not installed. Install dependencies listed at the top.")
        return False

    try:
        with xr.open_dataset(path) as dataset:
            print("Dataset summary:")
            print(dataset)

            print("\nDimensions:")
            if dataset.sizes:
                for name, size in dataset.sizes.items():
                    print(f"  {name}: {size}")
            else:
                print("  No dimensions found.")

            print("\nCoordinates:")
            if dataset.coords:
                for name, coord in dataset.coords.items():
                    print(f"  {name}: dims={coord.dims}, shape={coord.shape}, dtype={coord.dtype}")
            else:
                print("  No coordinates found.")

            print("\nData variables:")
            if dataset.data_vars:
                for name, var in dataset.data_vars.items():
                    print(f"  {name}: dims={var.dims}, shape={var.shape}, dtype={var.dtype}")
            else:
                print("  No data variables found.")

            print()
            print_xarray_crs(dataset)

            print("\nImportant variable min/max:")
            any_important = False
            for name in IMPORTANT_NETCDF_VARIABLES:
                if name not in dataset:
                    continue
                any_important = True
                min_value, max_value = xarray_min_max(dataset[name])
                if min_value is None or max_value is None:
                    print(f"  {name}: min/max unavailable")
                else:
                    print(f"  {name}: min={min_value:.6g}, max={max_value:.6g}")

            if not any_important:
                print("  None of the expected important variables were found.")

            return "velocity_sbas" in dataset.data_vars
    except Exception as exc:
        print(f"Could not inspect NetCDF file {path}: {exc}")
        return False


def inspect_netcdf_files(netcdf_files: list[Path]) -> Path | None:
    if not netcdf_files:
        print_section("NetCDF")
        print("No NetCDF files found in the project root.")
        return None

    velocity_sbas_path = None
    for path in netcdf_files:
        has_velocity_sbas = inspect_netcdf(path)
        if has_velocity_sbas and velocity_sbas_path is None:
            velocity_sbas_path = path

    return velocity_sbas_path


def raster_min_max(data, nodata):
    try:
        import numpy as np

        values = data.astype("float64", copy=False)
        if nodata is not None:
            values = np.where(values == nodata, np.nan, values)

        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None, None

        return float(finite_values.min()), float(finite_values.max())
    except Exception as exc:
        print(f"  Could not compute raster min/max: {exc}")
        return None, None


def inspect_geotiff(path: Path) -> None:
    try:
        import rasterio
    except ImportError:
        print("rasterio is not installed. Install dependencies listed at the top.")
        return

    try:
        with rasterio.open(path) as src:
            data = src.read(1, masked=False)
            min_value, max_value = raster_min_max(data, src.nodata)

            print(f"\n{path.name}")
            print(f"  Width/height: {src.width} x {src.height}")
            print(f"  CRS: {src.crs}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Nodata: {src.nodata}")
            if min_value is None or max_value is None:
                print("  Min/max: unavailable")
            else:
                print(f"  Min/max: {min_value:.6g} / {max_value:.6g}")
    except Exception as exc:
        print(f"\n{path.name}")
        print(f"  Could not inspect GeoTIFF: {exc}")


def inspect_geotiff_files(geotiff_files: list[Path], max_files: int = 5) -> None:
    print_section("GeoTIFF inspection")

    if not geotiff_files:
        print("No GeoTIFF files found in geotiffs/.")
        return

    print(f"Inspecting first {min(len(geotiff_files), max_files)} of {len(geotiff_files)} GeoTIFF files.")
    for path in geotiff_files[:max_files]:
        inspect_geotiff(path)


def inspect_first_csv(csv_files: list[Path]) -> Path | None:
    print_section("CSV time series inspection")

    if not csv_files:
        print("No CSV files found in timeseries/.")
        return None

    try:
        import pandas as pd
    except ImportError:
        print("pandas is not installed. Install dependencies listed at the top.")
        return None

    csv_path = csv_files[0]
    try:
        data = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Could not read CSV file {csv_path}: {exc}")
        return None

    print(f"File: {csv_path.name}")
    print(f"Rows: {len(data)}")
    print("Columns:")
    for column in data.columns:
        print(f"  {column}")

    print("\nFirst 5 rows:")
    if data.empty:
        print("  CSV is empty.")
    else:
        print(data.head(5).to_string(index=False))

    return csv_path


def reduce_data_array_for_plot(data_array):
    plot_data = data_array.squeeze(drop=True)

    while plot_data.ndim > 2:
        dim = plot_data.dims[0]
        plot_data = plot_data.isel({dim: 0}).squeeze(drop=True)

    return plot_data


def plot_velocity_sbas(netcdf_path: Path) -> bool:
    output_path = OUTPUT_DIR / "quicklook_velocity_sbas.png"

    try:
        import matplotlib.pyplot as plt
        import xarray as xr
    except ImportError:
        print("matplotlib or xarray is not installed. Skipping NetCDF quicklook.")
        return False

    try:
        with xr.open_dataset(netcdf_path) as dataset:
            if "velocity_sbas" not in dataset:
                return False

            plot_data = reduce_data_array_for_plot(dataset["velocity_sbas"])

            plt.figure(figsize=(8, 6))
            if plot_data.ndim == 2:
                plot_data.plot(cmap="viridis")
            else:
                plot_data.plot()
            plt.title(f"velocity_sbas from {netcdf_path.name}")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

        print(f"Saved quicklook plot: {output_path}")
        return True
    except Exception as exc:
        print(f"Could not create velocity_sbas quicklook: {exc}")
        return False


def plot_first_geotiff(geotiff_path: Path) -> bool:
    output_path = OUTPUT_DIR / "quicklook_first_geotiff.png"

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
    except ImportError:
        print("matplotlib, numpy, or rasterio is not installed. Skipping GeoTIFF quicklook.")
        return False

    try:
        with rasterio.open(geotiff_path) as src:
            data = src.read(1, masked=False).astype("float64", copy=False)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)

        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap="viridis")
        plt.colorbar(label=geotiff_path.name)
        plt.title(geotiff_path.name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved quicklook plot: {output_path}")
        return True
    except Exception as exc:
        print(f"Could not create GeoTIFF quicklook: {exc}")
        return False


def plot_first_timeseries(csv_path: Path) -> bool:
    output_path = OUTPUT_DIR / "quicklook_timeseries.png"

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib or pandas is not installed. Skipping CSV quicklook.")
        return False

    try:
        data = pd.read_csv(csv_path)
        numeric_columns = list(data.select_dtypes(include="number").columns)
        if not numeric_columns:
            print("CSV quicklook skipped: no numeric columns found.")
            return False

        y_column = numeric_columns[0]
        x_values = data.index
        x_label = "row"

        for column in data.columns:
            if column == y_column:
                continue
            parsed = pd.to_datetime(data[column], errors="coerce")
            if parsed.notna().sum() >= max(1, len(parsed) // 2):
                x_values = parsed
                x_label = column
                break

        plt.figure(figsize=(8, 4))
        plt.plot(x_values, data[y_column], marker="o")
        plt.xlabel(x_label)
        plt.ylabel(y_column)
        plt.title(f"{y_column} from {csv_path.name}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved quicklook plot: {output_path}")
        return True
    except Exception as exc:
        print(f"Could not create CSV quicklook: {exc}")
        return False


def create_quicklook(
    velocity_sbas_netcdf: Path | None,
    geotiff_files: list[Path],
    csv_path: Path | None,
) -> None:
    print_section("Quicklook plot")

    if velocity_sbas_netcdf is not None and plot_velocity_sbas(velocity_sbas_netcdf):
        return

    if geotiff_files and plot_first_geotiff(geotiff_files[0]):
        return

    if csv_path is not None and plot_first_timeseries(csv_path):
        return

    print("No quicklook plot could be created from the available data.")


def main() -> None:
    print_section("InSAR output inspection")
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Quicklook output folder: {OUTPUT_DIR}")

    root_files = list_files(PROJECT_DIR, "Root folder contents")
    geotiff_files = [
        path
        for path in list_files(GEOTIFF_DIR, "geotiffs/ contents")
        if path.suffix.lower() in {".tif", ".tiff"}
    ]
    csv_files = [
        path
        for path in list_files(TIMESERIES_DIR, "timeseries/ contents")
        if path.suffix.lower() == ".csv"
    ]

    netcdf_files = prioritize_netcdf_files(root_files)
    velocity_sbas_netcdf = inspect_netcdf_files(netcdf_files)
    inspect_geotiff_files(geotiff_files)
    csv_path = inspect_first_csv(csv_files)
    create_quicklook(velocity_sbas_netcdf, geotiff_files, csv_path)


if __name__ == "__main__":
    main()
