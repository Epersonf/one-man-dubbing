from __future__ import annotations

from pathlib import Path
from typing import Protocol

from core.domain.audio_asset import AudioAsset
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingJob


class VoiceConversionEngine(Protocol):
    """Contract for engines that train a timbre model from a reference
    and then convert audio (e.g. RVC)."""

    engine_name: str

    def train(
        self,
        reference: AudioAsset,
        config: TrainingConfig,
    ) -> TrainingJob:
        """Start training; returns a trackable/asynchronous job."""
        ...

    def convert(
        self,
        source_audio: AudioAsset,
        trained_model_path: Path,
    ) -> AudioAsset:
        """Apply the trained model to the actor's audio."""
        ...

    def is_ready(self) -> bool:
        """True if weights/dependencies are already installed."""
        ...

    def trained_model_path_for(self, voice_name: str) -> Path:
        """Where this engine writes/looks up a voice's trained model.

        Each engine has its own on-disk layout for trained models (e.g. RVC
        writes into its own vendored repo), so callers ask the engine
        instead of assuming a shared convention.
        """
        ...
