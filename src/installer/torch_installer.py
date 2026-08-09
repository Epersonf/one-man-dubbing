from __future__ import annotations

import sys

from core.domain.errors import InstallationError
from infra.process_runner import ProcessRunError, run_command
from installer.gpu_detector import GpuInfo

_CUDA_INDEX_BY_MAJOR_MINOR: dict[tuple[int, int], str] = {
    (12, 4): "cu124",
    (12, 1): "cu121",
    (11, 8): "cu118",
}
_DEFAULT_CUDA_TAG = "cu121"


def _cuda_tag_for(driver_cuda_version: str) -> str:
    parts = driver_cuda_version.split(".")
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

    for (candidate_major, candidate_minor), tag in sorted(
        _CUDA_INDEX_BY_MAJOR_MINOR.items(), reverse=True
    ):
        if (major, minor) >= (candidate_major, candidate_minor):
            return tag
    return _DEFAULT_CUDA_TAG


def install_torch(gpu_info: GpuInfo) -> None:
    """Install the PyTorch build matching the detected CUDA driver version,
    from the official PyTorch package index.
    """
    cuda_tag = _cuda_tag_for(gpu_info.driver_cuda_version)
    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"

    try:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchaudio",
                "--index-url",
                index_url,
            ]
        )
    except ProcessRunError as exc:
        raise InstallationError(f"Failed to install PyTorch for {cuda_tag}: {exc}") from exc
