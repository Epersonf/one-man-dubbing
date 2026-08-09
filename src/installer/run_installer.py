from __future__ import annotations

from core.domain.errors import InstallationError
from infra.filesystem_paths import ensure_project_directories
from installer.dependency_installer import install_dependencies
from installer.engine_installer import install_all_engines
from installer.gpu_detector import detect_gpu
from installer.model_downloader import download_all_weights
from installer.torch_installer import install_torch


def run_full_setup() -> None:
    """Run the full one-time setup sequence with no manual intervention,
    other than assuming the NVIDIA driver is already installed.
    """
    ensure_project_directories()

    print("[1/6] Detecting GPU...")
    gpu_info = detect_gpu()
    print(f"      Found {gpu_info.name} (driver CUDA {gpu_info.driver_cuda_version})")

    print("[2/6] Installing PyTorch...")
    install_torch(gpu_info)

    print("[3/6] Installing project dependencies...")
    install_dependencies()

    print("[4/6] Installing engines (RVC, Fish Speech)...")
    install_all_engines()

    print("[5/6] Downloading base model weights...")
    download_all_weights()

    print("[6/6] Validating installation...")
    _validate_installation()

    print("Setup complete. Run 'python main.py' to start OneManDubbing.")


def _validate_installation() -> None:
    try:
        import torch
    except ImportError as exc:
        raise InstallationError("PyTorch import failed after installation.") from exc

    if not torch.cuda.is_available():
        raise InstallationError(
            "PyTorch was installed but no CUDA GPU is visible to it."
        )


if __name__ == "__main__":
    run_full_setup()
