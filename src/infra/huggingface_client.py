from __future__ import annotations

from pathlib import Path

from core.domain.errors import DownloadError


def download_file(repo_id: str, filename: str, destination_dir: Path) -> Path:
    """Download a single file from a Hugging Face repo into destination_dir.

    Single wrapper around huggingface_hub so no other module imports it
    directly (DRY, and a single place to translate library errors into
    domain errors).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DownloadError("huggingface_hub is not installed") from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(destination_dir),
        )
    except Exception as exc:
        raise DownloadError(f"Failed to download {filename!r} from {repo_id!r}: {exc}") from exc

    result_path = Path(downloaded_path)
    if not result_path.is_file():
        raise DownloadError(f"Download reported success but file is missing: {result_path}")
    return result_path


def download_and_extract_zip(repo_id: str, filename: str, destination_dir: Path) -> None:
    """Download a .zip asset and extract it into destination_dir, discarding
    the archive. Used for bundled sample packs (e.g. RVC's mute.zip) that
    aren't a single usable file on their own.
    """
    import zipfile

    archive_path = download_file(repo_id, filename, destination_dir)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination_dir)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"Downloaded archive is not a valid zip: {archive_path}") from exc
    finally:
        archive_path.unlink(missing_ok=True)
