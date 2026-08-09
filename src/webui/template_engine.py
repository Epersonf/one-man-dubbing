from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

WEBUI_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEBUI_DIR / "templates"))
