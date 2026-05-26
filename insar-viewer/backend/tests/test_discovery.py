"""Tests for project discovery scoring."""
from pathlib import Path

import pytest

from insar_viewer.project.discovery import find_product_dir, product_score

# Adjust to absolute path relative to repo root
REPO_ROOT = Path(__file__).parent.parent.parent.parent  # insar-viewer/../ = InSAR-Data-Analysis-Tool
LEGACY_PROJECT = REPO_ROOT / "Data" / "old" / "project_dam_D"
NEW_PROJECT = REPO_ROOT / "Data" / "project_D_results_only"


def test_product_score_zero_on_empty(tmp_path: Path) -> None:
    assert product_score(tmp_path) == 0


def test_product_score_parameters_only(tmp_path: Path) -> None:
    (tmp_path / "parameters.json").write_text("{}")
    assert product_score(tmp_path) == 3


def test_product_score_full(tmp_path: Path) -> None:
    (tmp_path / "results_tight.nc").write_text("")
    (tmp_path / "results_wide.nc").write_text("")
    (tmp_path / "parameters.json").write_text("")
    (tmp_path / "manifest.json").write_text("")
    (tmp_path / "geotiffs").mkdir()
    assert product_score(tmp_path) == 8 + 5 + 3 + 2 + 2


def test_find_product_dir_legacy(tmp_path: Path) -> None:
    if not LEGACY_PROJECT.exists():
        pytest.skip("Legacy project not present")
    found = find_product_dir(LEGACY_PROJECT)
    assert found == LEGACY_PROJECT.resolve()


def test_find_product_dir_new_standard(tmp_path: Path) -> None:
    if not NEW_PROJECT.exists():
        pytest.skip("New standard project not present")
    found = find_product_dir(NEW_PROJECT)
    # Should find the outputs/project_dam_D/ subfolder
    assert (found / "results_tight.nc").exists() or (found / "results_wide.nc").exists()


def test_find_product_dir_returns_root_when_empty(tmp_path: Path) -> None:
    # A folder with nothing in it should fall back to itself
    result = find_product_dir(tmp_path)
    assert result == tmp_path.resolve()


def test_find_product_dir_prefers_tight_over_wide(tmp_path: Path) -> None:
    inner = tmp_path / "outputs" / "proj_D"
    inner.mkdir(parents=True)
    (inner / "results_tight.nc").write_text("")
    (inner / "results_wide.nc").write_text("")
    (inner / "parameters.json").write_text("")
    (inner / "manifest.json").write_text("")
    (inner / "geotiffs").mkdir()
    result = find_product_dir(tmp_path)
    assert result == inner.resolve()
