from __future__ import annotations

from config import ENGINE_MANIFESTS, EngineManifest
from infra.filesystem_paths import WEIGHTS_DIR
from infra.huggingface_client import download_file


def download_all_weights() -> None:
    for manifest in ENGINE_MANIFESTS:
        download_weights_for(manifest)


def download_weights_for(manifest: EngineManifest) -> None:
    """Download every weight asset declared for one engine's manifest.

    Reads the declarative EngineManifest.weight_assets table instead of
    hardcoding filenames, so a new engine only needs a new manifest entry.
    """
    destination_dir = WEIGHTS_DIR / manifest.engine_name
    for asset in manifest.weight_assets:
        download_file(asset.repo_id, asset.filename, destination_dir)
