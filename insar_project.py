"""
Shared project discovery helpers for exported InSAR products.

The current product layout is an outer bundle folder that contains quicklook
images plus machine-readable outputs under outputs/<project>_<orbit>/.
These helpers also accept the machine-readable product folder directly.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


NEW_STANDARD_PROJECT = Path("Data") / "project_D_results_only"
LEGACY_PROJECT = Path("Data") / "old" / "project_dam_D"
PROJECT_ENV_VAR = "INSAR_PROJECT_DIR"

PREFERRED_NETCDF_NAMES = (
    "results_tight.nc",
    "results_aoi_masked.nc",
    "results_AOI_clipped.nc",
    "results_wide.nc",
)

SBAS_DISPLACEMENT_VARIABLES = (
    "sbas_displacement_masked",
    "sbas_displacement_segmented_same_pixel",
    "sbas_displacement_raw",
)

LEGACY_DISPLACEMENT_VARIABLES = (
    "displacement_sbas",
)

DISPLACEMENT_VARIABLES = SBAS_DISPLACEMENT_VARIABLES + LEGACY_DISPLACEMENT_VARIABLES

IMPORTANT_NETCDF_VARIABLES = (
    "sbas_velocity_masked",
    "sbas_velocity_raw",
    "sbas_rmse_masked",
    "sbas_rmse_raw",
    "sbas_displacement_masked",
    "sbas_displacement_segmented_same_pixel",
    "sbas_displacement_raw",
    "sbas_valid_time_mask",
    "sbas_segment_id",
    "coherence_median",
    "coherence_mean",
    "valid_pixel_mask",
    "coherence_per_date",
    "coherence_stack",
    "dem",
    "velocity_sbas",
    "displacement_sbas",
    "rmse_sbas",
    "psf",
)

MAP_VARIABLES = (
    "sbas_velocity_masked",
    "sbas_velocity_raw",
    "sbas_rmse_masked",
    "sbas_rmse_raw",
    "coherence_median",
    "coherence_mean",
    "valid_pixel_mask",
    "dem",
    "velocity_sbas",
    "rmse_sbas",
    "psf",
)

VELOCITY_VARIABLE_CANDIDATES = (
    "sbas_velocity_masked",
    "sbas_velocity_raw",
    "velocity_sbas",
)

QUALITY_VARIABLES = {
    "sbas_displacement_masked": "sbas_rmse_masked",
    "sbas_displacement_segmented_same_pixel": "sbas_rmse_masked",
    "sbas_displacement_raw": "sbas_rmse_raw",
    "displacement_sbas": "rmse_sbas",
}

VARIABLE_LABELS = {
    "sbas_displacement_masked": "SBAS masked displacement",
    "sbas_displacement_segmented_same_pixel": "SBAS segmented displacement",
    "sbas_displacement_raw": "SBAS raw displacement",
    "displacement_sbas": "SBAS displacement",
    "sbas_velocity_masked": "SBAS masked velocity",
    "sbas_velocity_raw": "SBAS raw velocity",
    "velocity_sbas": "SBAS velocity",
    "sbas_rmse_masked": "SBAS masked RMSE",
    "sbas_rmse_raw": "SBAS raw RMSE",
    "rmse_sbas": "SBAS RMSE",
    "coherence_median": "Median coherence",
    "coherence_mean": "Mean coherence",
    "valid_pixel_mask": "Static valid pixel mask",
    "sbas_valid_time_mask": "Dynamic valid time mask",
    "sbas_segment_id": "SBAS segment ID",
    "dem": "DEM",
    "psf": "PSF",
}


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path
    product_dir: Path
    geotiff_dir: Path
    timeseries_dir: Path
    parameters_path: Path
    manifest_path: Path
    metadata_path: Path

    @property
    def quicklook_dir(self) -> Path:
        return self.root_dir

    @property
    def aoi_output_dir(self) -> Path:
        return self.product_dir / "aoi_only"


def add_project_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "project_dir",
        nargs="?",
        help=(
            "Project bundle or product folder. Defaults to INSAR_PROJECT_DIR, "
            "then Data/project_D_results_only when available."
        ),
    )


def default_project_dir() -> Path:
    env_value = os.environ.get(PROJECT_ENV_VAR)
    if env_value:
        return Path(env_value)

    for candidate in (NEW_STANDARD_PROJECT, LEGACY_PROJECT):
        if candidate.exists():
            return candidate

    data_dir = Path("Data")
    if data_dir.exists():
        product = find_first_product_dir(data_dir)
        if product is not None:
            return product

    return Path.cwd()


def resolve_project_paths(project_dir: str | Path | None = None) -> ProjectPaths:
    root_dir = Path(project_dir) if project_dir else default_project_dir()
    root_dir = root_dir.expanduser().resolve()
    product_dir = find_product_dir(root_dir)

    return ProjectPaths(
        root_dir=root_dir,
        product_dir=product_dir,
        geotiff_dir=product_dir / "geotiffs",
        timeseries_dir=product_dir / "timeseries",
        parameters_path=product_dir / "parameters.json",
        manifest_path=product_dir / "manifest.json",
        metadata_path=product_dir / "sbas_results_metadata.json",
    )


def find_first_product_dir(base_dir: Path) -> Path | None:
    candidates = [path.parent for path in base_dir.rglob("results_tight.nc")]
    candidates.extend(path.parent for path in base_dir.rglob("results_wide.nc"))
    candidates = sorted(set(candidates), key=lambda path: len(path.parts))
    return candidates[0] if candidates else None


def find_product_dir(root_dir: Path) -> Path:
    candidates = [root_dir]
    outputs_dir = root_dir / "outputs"
    if outputs_dir.exists():
        candidates.extend(path for path in outputs_dir.iterdir() if path.is_dir())

    nested = find_first_product_dir(root_dir)
    if nested is not None:
        candidates.append(nested)

    scored = sorted(
        ((product_score(path), path) for path in candidates if path.exists()),
        key=lambda item: (item[0], -len(item[1].parts)),
        reverse=True,
    )
    if scored and scored[0][0] > 0:
        return scored[0][1].resolve()

    return root_dir


def product_score(path: Path) -> int:
    score = 0
    if (path / "results_tight.nc").exists():
        score += 8
    if (path / "results_wide.nc").exists():
        score += 5
    if (path / "sbas_results_masked.npz").exists():
        score += 4
    if (path / "parameters.json").exists():
        score += 3
    if (path / "manifest.json").exists():
        score += 2
    if (path / "geotiffs").is_dir():
        score += 2
    return score


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


def find_netcdf_files(project_paths: ProjectPaths) -> list[Path]:
    files = sorted(path for path in project_paths.product_dir.iterdir() if path.is_file())
    return prioritize_netcdf_files(files)


def first_existing_variable(dataset, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in dataset.data_vars:
            return name
    return None


def display_label(name: str) -> str:
    return VARIABLE_LABELS.get(name, name.replace("_", " "))
