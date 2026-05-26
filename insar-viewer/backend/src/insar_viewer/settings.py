"""App-level settings persisted to %APPDATA%\\InSARViewer\\."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from platformdirs import user_data_dir
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

APP_NAME = "InSARViewer"
APP_AUTHOR = "InSARViewer"


def _app_dir() -> Path:
    return Path(user_data_dir(APP_NAME, APP_AUTHOR))


def _settings_path() -> Path:
    return _app_dir() / "settings.json"


def _recent_path() -> Path:
    return _app_dir() / "recent_projects.json"


class AppSettings(BaseModel):
    model_config = {"extra": "ignore"}

    theme: str = "dark"
    default_basemap: str = "esri_satellite"
    default_colormap_velocity: str = "RdBu_r"
    default_colormap_displacement: str = "RdBu_r"
    units: str = "mm"


class RecentProject(BaseModel):
    path: str
    name: str
    last_opened: str


def load_settings() -> AppSettings:
    p = _settings_path()
    if p.exists():
        try:
            return AppSettings.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse settings.json; using defaults.")
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(settings.model_dump_json(indent=2), encoding="utf-8")


def load_recent() -> list[RecentProject]:
    p = _recent_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [RecentProject.model_validate(r) for r in raw]
    except Exception:
        return []


def add_recent(path: str, name: str) -> None:
    from datetime import datetime, timezone

    existing = [r for r in load_recent() if r.path != path]
    entry = RecentProject(
        path=path,
        name=name,
        last_opened=datetime.now(timezone.utc).isoformat(),
    )
    updated = [entry] + existing
    updated = updated[:10]
    p = _recent_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([r.model_dump() for r in updated], indent=2),
        encoding="utf-8",
    )


def setup_logging() -> None:
    log_dir = _app_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
