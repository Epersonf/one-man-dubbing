from __future__ import annotations

from config import ENGINE_MANIFESTS, EngineManifest
from infra.filesystem_paths import VENDOR_DIR
from infra.huggingface_client import download_and_extract_zip, download_file


def download_all_weights() -> None:
    for manifest in ENGINE_MANIFESTS:
        download_weights_for(manifest)


def download_weights_for(manifest: EngineManifest) -> None:
    """Download every weight asset declared for one engine's manifest.

    Reads the declarative EngineManifest.weight_assets table instead of
    hardcoding filenames, so a new engine only needs a new manifest entry.
    Each asset's local_dir is relative to the engine's own vendor
    directory, since real engine code expects weights alongside its code.
    """
    vendor_dir = VENDOR_DIR / manifest.vendor_subdir
    for asset in manifest.weight_assets:
        destination_dir = vendor_dir / asset.local_dir if asset.local_dir else vendor_dir
        if asset.extract_zip:
            download_and_extract_zip(asset.repo_id, asset.filename, destination_dir)
        else:
            download_file(asset.repo_id, asset.filename, destination_dir)
