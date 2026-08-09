from __future__ import annotations

from typing import Protocol

from core.domain.audio_asset import AudioAsset
from core.domain.voice_profile import SynthesisParameters


class VoiceSynthesisEngine(Protocol):
    """Contract for engines that generate voice audio from parameters,
    without a real reference audio (e.g. Fish Speech)."""

    engine_name: str

    def synthesize(
        self,
        text_sample: str,
        parameters: SynthesisParameters,
    ) -> AudioAsset:
        """Generate a synthetic audio that will serve as a reference."""
        ...

    def is_ready(self) -> bool:
        """True if weights/dependencies are already installed."""
        ...
