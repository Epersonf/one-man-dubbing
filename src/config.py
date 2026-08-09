from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HuggingFaceAsset:
    """A single file to fetch from a Hugging Face repo, saved relative to
    the engine's own vendor directory (not a generic shared weights/ dir):
    real engine code (e.g. RVC) hardcodes weight paths relative to its own
    repo root, so weights have to live inside vendor/<engine>/ to be found.
    """

    repo_id: str
    filename: str
    local_dir: str = ""
    extract_zip: bool = False


@dataclass(frozen=True)
class EngineManifest:
    """Declarative description of everything needed to install one engine.

    model_downloader.py and engine_installer.py read a table of these
    instead of hardcoding engine-specific logic. Adding a third conversion
    or synthesis engine in the future means appending one manifest here.
    """

    engine_name: str
    git_repository: str
    vendor_subdir: str
    weight_assets: tuple[HuggingFaceAsset, ...] = field(default_factory=tuple)
    requirements_filename: str = "requirements.txt"


ENGINE_MANIFESTS: tuple[EngineManifest, ...] = (
    EngineManifest(
        engine_name="rvc",
        git_repository="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git",
        vendor_subdir="rvc",
        requirements_filename="requirments_cu128_py312.txt",
        weight_assets=(
            HuggingFaceAsset(
                "lj1995/VoiceConversionWebUI", "hubert_base/config.json", "assets"
            ),
            HuggingFaceAsset(
                "lj1995/VoiceConversionWebUI",
                "hubert_base/preprocessor_config.json",
                "assets",
            ),
            HuggingFaceAsset(
                "lj1995/VoiceConversionWebUI", "hubert_base/pytorch_model.bin", "assets"
            ),
            HuggingFaceAsset("lj1995/VoiceConversionWebUI", "rmvpe.pt", "assets/rmvpe"),
            HuggingFaceAsset(
                "lj1995/VoiceConversionWebUI", "pretrained_v2/f0G40k.pth", "assets"
            ),
            HuggingFaceAsset(
                "lj1995/VoiceConversionWebUI", "pretrained_v2/f0D40k.pth", "assets"
            ),
            HuggingFaceAsset("lj1995/VoiceConversionWebUI", "ffmpeg.exe", ""),
            HuggingFaceAsset("lj1995/VoiceConversionWebUI", "ffprobe.exe", ""),
            HuggingFaceAsset(
                "lj1995/VoiceConversionWebUI", "mute.zip", "logs", extract_zip=True
            ),
        ),
    ),
    EngineManifest(
        engine_name="fish_speech",
        git_repository="https://github.com/fishaudio/fish-speech.git",
        vendor_subdir="fish_speech",
        weight_assets=(
            HuggingFaceAsset("fishaudio/fish-speech-1.5", "model.pth", "weights"),
            HuggingFaceAsset("fishaudio/fish-speech-1.5", "config.json", "weights"),
        ),
    ),
)

DEFAULT_CONVERSION_ENGINE = "rvc"
DEFAULT_SYNTHESIS_ENGINE = "fish_speech"

DEFAULT_OUTPUT_FORMAT = "mp3"
DEFAULT_SAMPLE_RATE = 40_000

# RVC training defaults for this fork's real CLI (see core/engines/rvc_engine.py).
RVC_MODEL_VERSION = "v2"
RVC_FEATURE_DIM = 768
RVC_F0_METHOD = "rmvpe"

WEBUI_HOST = "127.0.0.1"
WEBUI_PORT = 7860
