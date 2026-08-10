from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

WEBUI_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBUI_DIR / "static"
templates = Jinja2Templates(directory=str(WEBUI_DIR / "templates"))


def _static_version() -> str:
    # Cache-busting: without this, browsers keep serving a stale app.js
    # after an edit, silently running old (or missing) JS behavior - e.g. a
    # form losing its submit handler and falling back to a native GET.
    mtimes = (path.stat().st_mtime for path in STATIC_DIR.rglob("*") if path.is_file())
    return str(int(max(mtimes, default=0)))


templates.env.globals["static_version"] = _static_version()
