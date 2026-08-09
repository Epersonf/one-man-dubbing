from __future__ import annotations

import threading
import webbrowser

import uvicorn

from config import WEBUI_HOST, WEBUI_PORT
from infra.filesystem_paths import RVC_HUBERT_DIR, RVC_MUTE_DIR
from installer.run_installer import run_full_setup


def _needs_first_run_setup() -> bool:
    return not (RVC_HUBERT_DIR.is_dir() and RVC_MUTE_DIR.is_dir())


def _open_browser_when_ready() -> None:
    webbrowser.open(f"http://{WEBUI_HOST}:{WEBUI_PORT}")


def main() -> None:
    if _needs_first_run_setup():
        print("First run detected — installing dependencies, engines and model weights...")
        run_full_setup()

    from webui.app import app

    threading.Timer(1.5, _open_browser_when_ready).start()
    uvicorn.run(app, host=WEBUI_HOST, port=WEBUI_PORT)


if __name__ == "__main__":
    main()
