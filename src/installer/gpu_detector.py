from __future__ import annotations

import re
from dataclasses import dataclass

from core.domain.errors import InstallationError
from infra.process_runner import ProcessRunError, run_command

_CUDA_VERSION_PATTERN = re.compile(r"CUDA(?: UMD)? Version:\s*([\d.]+)")
_GPU_NAME_PATTERN = re.compile(r"^\s*\d+\s*,\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver_cuda_version: str


def detect_gpu() -> GpuInfo:
    """Detect the NVIDIA GPU and the CUDA version supported by its driver.

    Assumes the NVIDIA driver is already installed on the system; this
    project deliberately does not attempt to install drivers itself.
    """
    try:
        smi_output = run_command(["nvidia-smi"])
    except ProcessRunError as exc:
        raise InstallationError(
            "nvidia-smi did not run successfully. Install the NVIDIA driver first."
        ) from exc

    cuda_match = _CUDA_VERSION_PATTERN.search(smi_output)
    if cuda_match is None:
        raise InstallationError("Could not determine CUDA version from nvidia-smi output.")

    try:
        name_output = run_command(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"]
        )
    except ProcessRunError as exc:
        raise InstallationError("Could not query GPU name from nvidia-smi.") from exc

    name_match = _GPU_NAME_PATTERN.search(name_output)
    gpu_name = name_match.group(1) if name_match else "Unknown NVIDIA GPU"

    return GpuInfo(name=gpu_name, driver_cuda_version=cuda_match.group(1))
