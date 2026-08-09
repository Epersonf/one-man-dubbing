from __future__ import annotations

import sys

from config import ENGINE_MANIFESTS, EngineManifest
from core.domain.errors import InstallationError
from infra.filesystem_paths import VENDOR_DIR
from infra.process_runner import ProcessRunError, run_command


def install_all_engines() -> None:
    for manifest in ENGINE_MANIFESTS:
        install_engine(manifest)


def install_engine(manifest: EngineManifest) -> None:
    """Clone one engine's repository into vendor/ and install its own
    requirements.txt, if present. Driven entirely by the declarative
    EngineManifest table in config.py - no engine-specific logic here.
    """
    target_dir = VENDOR_DIR / manifest.vendor_subdir

    if target_dir.is_dir():
        return

    try:
        run_command(["git", "clone", "--depth", "1", manifest.git_repository, str(target_dir)])
    except ProcessRunError as exc:
        raise InstallationError(f"Failed to clone {manifest.engine_name}: {exc}") from exc

    engine_requirements = target_dir / "requirements.txt"
    if engine_requirements.is_file():
        try:
            run_command(
                [sys.executable, "-m", "pip", "install", "-r", str(engine_requirements)]
            )
        except ProcessRunError as exc:
            raise InstallationError(
                f"Failed to install dependencies for {manifest.engine_name}: {exc}"
            ) from exc
