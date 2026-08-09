from __future__ import annotations

import sys
from pathlib import Path

from config import ENGINE_MANIFESTS, EngineManifest
from core.domain.errors import InstallationError
from infra.filesystem_paths import VENDOR_DIR
from infra.process_runner import ProcessRunError, run_command


def install_all_engines() -> None:
    for manifest in ENGINE_MANIFESTS:
        install_engine(manifest)


def install_engine(manifest: EngineManifest) -> None:
    """Clone one engine's repository into vendor/ (if not already there)
    and install its own dependency file, if present. Driven entirely by
    the declarative EngineManifest table in config.py - no engine-specific
    logic here.
    """
    target_dir = VENDOR_DIR / manifest.vendor_subdir

    if not target_dir.is_dir():
        try:
            run_command(
                ["git", "clone", "--depth", "1", manifest.git_repository, str(target_dir)]
            )
        except ProcessRunError as exc:
            raise InstallationError(f"Failed to clone {manifest.engine_name}: {exc}") from exc

    _install_engine_requirements(manifest, target_dir)


def _install_engine_requirements(manifest: EngineManifest, target_dir: Path) -> None:
    engine_requirements = target_dir / manifest.requirements_filename
    if not engine_requirements.is_file():
        return

    # Strip any regional mirror pins the vendored file may declare (e.g.
    # "--index-url <mirror>") so installation isn't at the mercy of a
    # mirror's reachability; everything else in the file is used as-is.
    # Also drop onnxruntime-gpu: it's only imported for the DirectML
    # (AMD/Intel) inference backend (see infer/rmvpe.py), unused on the
    # CUDA path this project targets, and its version pin can lag behind
    # the newest Python releases.
    lines = engine_requirements.read_text(encoding="utf8").splitlines()
    skip_prefixes = ("--index-url", "--extra-index-url", "onnxruntime-gpu")
    filtered = [line for line in lines if not line.strip().startswith(skip_prefixes)]
    staged_requirements = target_dir / f".{manifest.engine_name}_requirements.filtered.txt"
    staged_requirements.write_text("\n".join(filtered), encoding="utf8")

    try:
        run_command([sys.executable, "-m", "pip", "install", "-r", str(staged_requirements)])
    except ProcessRunError as exc:
        raise InstallationError(
            f"Failed to install dependencies for {manifest.engine_name}: {exc}"
        ) from exc
    finally:
        staged_requirements.unlink(missing_ok=True)
