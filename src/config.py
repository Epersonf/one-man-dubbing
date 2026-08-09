from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HuggingFaceAsset:
    """A single file to fetch from a Hugging Face repo."""

    repo_id: str
    filename: str


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


ENGINE_MANIFESTS: tuple[EngineManifest, ...] = (
    EngineManifest(
        engine_name="rvc",
        git_repository="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git",
        vendor_subdir="rvc",
        weight_assets=(
            HuggingFaceAsset("lj1995/VoiceConversionWebUI", "hubert_base.pt"),
            HuggingFaceAsset("lj1995/VoiceConversionWebUI", "rmvpe.pt"),
            HuggingFaceAsset("lj1995/VoiceConversionWebUI", "pretrained_v2/f0G40k.pth"),
            HuggingFaceAsset("lj1995/VoiceConversionWebUI", "pretrained_v2/f0D40k.pth"),
        ),
    ),
    EngineManifest(
        engine_name="fish_speech",
        git_repository="https://github.com/fishaudio/fish-speech.git",
        vendor_subdir="fish_speech",
        weight_assets=(
            HuggingFaceAsset("fishaudio/fish-speech-1.5", "model.pth"),
            HuggingFaceAsset("fishaudio/fish-speech-1.5", "config.json"),
        ),
    ),
)

DEFAULT_CONVERSION_ENGINE = "rvc"
DEFAULT_SYNTHESIS_ENGINE = "fish_speech"

DEFAULT_OUTPUT_FORMAT = "mp3"
DEFAULT_SAMPLE_RATE = 40_000

WEBUI_HOST = "127.0.0.1"
WEBUI_PORT = 7860
