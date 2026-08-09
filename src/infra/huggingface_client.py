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


def download_files(repo_id: str, filenames: list[str], destination_dir: Path) -> list[Path]:
    return [download_file(repo_id, filename, destination_dir) for filename in filenames]
