"""pywebview window + uvicorn thread launcher."""
from __future__ import annotations

import logging
import socket
import threading
import time

import uvicorn
import webview

from .app import create_app
from .settings import setup_logging

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FolderPickerApi:
    """Exposed to JavaScript via pywebview JS bridge."""

    def pick_folder(self) -> str | None:
        result = webview.windows[0].create_file_dialog(
            webview.FOLDER_DIALOG,
            allow_multiple=False,
        )
        if result:
            return result[0]
        return None


def run(dev_mode: bool = False) -> None:
    setup_logging()
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    app = create_app(dev_mode=dev_mode)

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(server_config)

    def _start_server() -> None:
        server.run()

    thread = threading.Thread(target=_start_server, daemon=True)
    thread.start()

    # Wait until server is up (max 10s)
    for _ in range(100):
        try:
            import urllib.request
            urllib.request.urlopen(f"{url}/api/docs", timeout=0.1)
            break
        except Exception:
            time.sleep(0.1)

    logger.info("Server started at %s", url)

    api = _FolderPickerApi()
    window = webview.create_window(
        "InSAR Deformation Viewer",
        url,
        width=1400,
        height=900,
        min_size=(900, 600),
        js_api=api,
    )

    webview.start(debug=dev_mode)

    # Window closed — signal server shutdown
    server.should_exit = True
    thread.join(timeout=5)
