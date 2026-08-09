from __future__ import annotations

import sys
from pathlib import Path

from core.domain.errors import InstallationError
from infra.filesystem_paths import REPO_ROOT
from infra.process_runner import ProcessRunError, run_command


def install_dependencies(requirements_path: Path | None = None) -> None:
    """pip install every dependency listed in the project's requirements.txt."""
    target = requirements_path or (REPO_ROOT / "requirements.txt")
    if not target.is_file():
        raise InstallationError(f"requirements file not found: {target}")

    try:
        run_command([sys.executable, "-m", "pip", "install", "-r", str(target)])
    except ProcessRunError as exc:
        raise InstallationError(f"Failed to install project dependencies: {exc}") from exc
