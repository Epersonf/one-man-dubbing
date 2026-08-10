from __future__ import annotations

from pathlib import Path
from typing import Protocol

from core.domain.audio_asset import AudioAsset
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingJob


class VoiceConversionEngine(Protocol):
    """Contract for engines that train a timbre model from one or more
    reference clips and then convert audio (e.g. RVC)."""

    engine_name: str

    def train(
        self,
        voice_name: str,
        reference_paths: list[Path],
        config: TrainingConfig,
        model_id: str,
        resume: bool = False,
    ) -> TrainingJob:
        """Start training a new model, identified by the caller-supplied
        model_id (a voice can have several trained models). Returns a
        trackable/asynchronous job whose job_id equals model_id.

        resume=True skips re-preprocessing reference_paths (assumed
        already done for this model_id in an earlier run) and continues
        training from the latest checkpoint toward config.epochs.
        """
        ...

    def convert(
        self,
        source_audio: AudioAsset,
        trained_model_path: Path,
        job_id: str,
    ) -> AudioAsset:
        """Apply the trained model to the actor's audio, publishing
        progress/log lines under job_id (see infra/job_progress_bus.py) as
        it runs - conversion can take a while and callers want live
        feedback, not just a result at the end.
        """
        ...

    def is_ready(self) -> bool:
        """True if weights/dependencies are already installed."""
        ...

    def trained_model_path_for(self, model_id: str) -> Path:
        """Where this engine writes/looks up one trained model's file.

        Each engine has its own on-disk layout for trained models (e.g. RVC
        writes into its own vendored repo), so callers ask the engine
        instead of assuming a shared convention.
        """
        ...

    def delete_trained_model(self, model_id: str) -> None:
        """Remove every engine-side artifact for one trained model
        (weights, training logs, similarity index). Safe to call even if
        some of those artifacts are already missing.
        """
        ...
