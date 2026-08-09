from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = PROJECT_ROOT.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
REFERENCES_DIR: Path = DATA_DIR / "references"
MODELS_DIR: Path = DATA_DIR / "models"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"

VENDOR_DIR: Path = PROJECT_ROOT / "vendor"
RVC_DIR: Path = VENDOR_DIR / "rvc"
FISH_SPEECH_DIR: Path = VENDOR_DIR / "fish_speech"

WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"


def ensure_project_directories() -> None:
    """Create every directory the app expects to exist, if missing."""
    for directory in (
        REFERENCES_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        VENDOR_DIR,
        WEIGHTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def reference_path_for(voice_name: str) -> Path:
    return REFERENCES_DIR / f"{voice_name}.wav"


def model_dir_for(voice_name: str) -> Path:
    return MODELS_DIR / voice_name


def output_path_for(voice_name: str, job_id: str, extension: str = "mp3") -> Path:
    return OUTPUTS_DIR / voice_name / f"{job_id}.{extension}"
