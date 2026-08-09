from __future__ import annotations

import threading

from config import DEFAULT_CONVERSION_ENGINE
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingJob
from core.domain.voice_profile import VoiceProfile
from core.registry.engine_registry import EngineRegistry
from infra.audio_io import load_audio_asset


class TrainingService:
    """Orchestrates Step 2: training a voice conversion model from the
    reference audio produced in Step 1.
    """

    def train(
        self,
        voice_profile: VoiceProfile,
        config: TrainingConfig,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
    ) -> TrainingJob:
        engine = EngineRegistry.get_conversion(engine_name)
        reference_asset = load_audio_asset(voice_profile.reference_audio_path)
        return engine.train(reference_asset, config)

    def start_training_in_background(
        self,
        voice_profile: VoiceProfile,
        config: TrainingConfig,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
    ) -> None:
        """Run training in a separate thread so the UI can poll progress
        (via the job progress bus, keyed by voice name) while it runs.
        """
        thread = threading.Thread(
            target=self.train, args=(voice_profile, config, engine_name), daemon=True
        )
        thread.start()

    def list_available_engines(self) -> list[str]:
        return EngineRegistry.available_conversion()
