from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = PROJECT_ROOT.parent

# Runtime state (downloaded weights, cloned engines, user-generated data) is
# never source code, so it lives outside src/ at the repo root, where it can
# be gitignored as a whole instead of poking holes in src/'s ignore rules.
DATA_DIR: Path = REPO_ROOT / "data"
REFERENCES_DIR: Path = DATA_DIR / "references"
MODELS_DIR: Path = DATA_DIR / "models"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"

VENDOR_DIR: Path = REPO_ROOT / "vendor"
RVC_DIR: Path = VENDOR_DIR / "rvc"
FISH_SPEECH_DIR: Path = VENDOR_DIR / "fish_speech"

# RVC hardcodes these paths relative to its own repo root (see
# infer/cli.py and infer/hubert.py in the vendored repo), so downloaded
# weights and per-voice training state live inside vendor/rvc/ itself
# rather than in a generic top-level weights/ directory.
RVC_ASSETS_DIR: Path = RVC_DIR / "assets"
RVC_HUBERT_DIR: Path = RVC_ASSETS_DIR / "hubert_base"
RVC_RMVPE_PATH: Path = RVC_ASSETS_DIR / "rmvpe" / "rmvpe.pt"
RVC_PRETRAINED_DIR: Path = RVC_ASSETS_DIR / "pretrained_v2"
RVC_TRAINED_WEIGHTS_DIR: Path = RVC_ASSETS_DIR / "weights"
RVC_LOGS_DIR: Path = RVC_DIR / "logs"
RVC_MUTE_DIR: Path = RVC_LOGS_DIR / "mute"


def ensure_project_directories() -> None:
    """Create every directory the app expects to exist, if missing."""
    for directory in (
        REFERENCES_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        VENDOR_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def reference_dir_for(voice_name: str) -> Path:
    """Folder holding every reference clip for a voice - a voice can be
    trained from more than one clip, so this is a directory, not a file."""
    return REFERENCES_DIR / voice_name


def legacy_reference_path_for(voice_name: str) -> Path:
    """Where a voice's single reference clip lived before multi-clip
    support (data/references/<voice_name>.wav). Only used to detect and
    migrate voices created before that change.
    """
    return REFERENCES_DIR / f"{voice_name}.wav"


def voice_metadata_path_for(voice_name: str) -> Path:
    return REFERENCES_DIR / f"{voice_name}.json"


def model_dir_for(voice_name: str) -> Path:
    """Per-voice directory holding metadata (and staged actor uploads) for
    every model trained on top of that voice - not the trained model
    binaries themselves, which are engine-specific (see rvc_*_path_for)."""
    return MODELS_DIR / voice_name


def model_metadata_path_for(voice_name: str, model_id: str) -> Path:
    return model_dir_for(voice_name) / f"{model_id}.json"


def output_path_for(voice_name: str, job_id: str, extension: str = "mp3") -> Path:
    return OUTPUTS_DIR / voice_name / f"{job_id}.{extension}"


def rvc_experiment_dir_for(model_id: str) -> Path:
    """RVC's own working directory for one training run (preprocessed
    audio, extracted features, checkpoints), keyed by model_id since a
    voice can have several trained models."""
    return RVC_LOGS_DIR / model_id


def rvc_trained_model_path_for(model_id: str) -> Path:
    """Where RVC writes the final inference-ready model (see
    train/process_ckpt.py's savee(), called from train/train.py)."""
    return RVC_TRAINED_WEIGHTS_DIR / f"{model_id}.pth"
