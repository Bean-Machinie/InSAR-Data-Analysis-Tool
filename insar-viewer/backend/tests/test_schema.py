"""Tests for Pydantic schema parsing."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from insar_viewer.project.schema import Manifest, Parameters

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # insar-viewer/../ = InSAR-Data-Analysis-Tool
LEGACY_PARAMS = REPO_ROOT / "Data" / "old" / "project_dam_D" / "parameters.json"
LEGACY_MANIFEST = REPO_ROOT / "Data" / "old" / "project_dam_D" / "manifest.json"


def test_parse_legacy_parameters() -> None:
    if not LEGACY_PARAMS.exists():
        pytest.skip("Legacy parameters.json not present")
    data = json.loads(LEGACY_PARAMS.read_text())
    p = Parameters.model_validate(data)
    assert p.project == "project_dam"
    assert p.orbit == "D"
    assert p.scenes.count == 8
    assert len(p.scenes.dates) == 8
    assert p.aoi.raw_wkt.startswith("POLYGON")
    assert len(p.pois) == 1
    assert p.pois[0].name == "aoi_centroid"


def test_parse_legacy_manifest() -> None:
    if not LEGACY_MANIFEST.exists():
        pytest.skip("Legacy manifest.json not present")
    data = json.loads(LEGACY_MANIFEST.read_text())
    m = Manifest.model_validate(data)
    assert m.project == "project_dam"
    assert m.crs == "EPSG:4326"
    assert "velocity_sbas" in m.variables
    assert m.variables["velocity_sbas"].units == "mm/year"


def test_parameters_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        Parameters.model_validate({"project": "test"})  # missing orbit, aoi, time_window, scenes


def test_parameters_extra_fields_allowed() -> None:
    raw = {
        "project": "test",
        "orbit": "A",
        "aoi": {"raw_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"},
        "time_window": {"start": "2024-01-01", "end": "2024-06-01"},
        "scenes": {"count": 2, "dates": ["2024-01-01", "2024-03-01"]},
        "unknown_future_field": "value",
    }
    p = Parameters.model_validate(raw)
    assert p.project == "test"


def test_manifest_unknown_variables_allowed() -> None:
    raw = {
        "project": "p", "orbit": "D",
        "variables": {
            "new_variable_type": {"units": "m/s", "description": "future var"},
        },
    }
    m = Manifest.model_validate(raw)
    assert "new_variable_type" in m.variables
