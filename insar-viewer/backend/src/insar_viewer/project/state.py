"""Active-project singleton. Replaces scattered VIEWER_DATA globals."""
from __future__ import annotations

from pathlib import Path

from .loader import LoadedProject, load_project

_active: LoadedProject | None = None


def open_project(path: Path) -> LoadedProject:
    global _active
    _active = load_project(path)
    return _active


def get_project() -> LoadedProject:
    if _active is None:
        raise RuntimeError("No project loaded.")
    return _active


def close_project() -> None:
    global _active
    _active = None


def is_loaded() -> bool:
    return _active is not None
