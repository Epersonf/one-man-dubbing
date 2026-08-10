from __future__ import annotations

import threading
from datetime import datetime

from config import DEFAULT_CONVERSION_ENGINE
from core.domain.errors import AudioValidationError, TrainingInProgressError
from core.domain.trained_model import TrainedModelInfo
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingJob, TrainingStatus
from core.domain.voice_profile import VoiceProfile
from core.registry.engine_registry import EngineRegistry
from core.services import model_metadata
from core.services.reference_voice_service import ReferenceVoiceService
from infra.audio_io import load_audio_asset
from infra.filesystem_paths import model_dir_for, model_metadata_path_for
from infra.job_progress_bus import progress_bus
from infra.metadata_store import delete_json, read_all_json, read_json

# Module-level, not per-instance: training is GPU/CPU-exclusive, and
# multiple TrainingService() instances exist (one per webui route module),
# so the lock has to be shared by all of them to actually serialize runs.
_training_lock = threading.Lock()


class TrainingService:
    """Orchestrates Step 2: training a voice conversion model from the
    reference clip(s) produced in Step 1. A voice can have several trained
    models (e.g. re-trained with different settings); each is identified
    by its own model_id.
    """

    def __init__(self) -> None:
        self._voice_service = ReferenceVoiceService()

    def train(
        self,
        voice_profile: VoiceProfile,
        config: TrainingConfig,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
        model_id: str | None = None,
        resume: bool = False,
    ) -> TrainingJob:
        engine = EngineRegistry.get_conversion(engine_name)
        reference_paths = self._voice_service.list_reference_clips(voice_profile.name)
        if not reference_paths:
            raise AudioValidationError(f"No reference clips found for voice: {voice_profile.name}")
        for path in reference_paths:
            load_audio_asset(path)

        resolved_model_id = model_id or model_metadata.new_model_id(voice_profile.name)
        try:
            return engine.train(voice_profile.name, reference_paths, config, resolved_model_id, resume)
        finally:
            # Runs on success AND failure: a model is usable for inference
            # the moment the engine has written its first checkpoint, so a
            # crash at epoch 500/2000 still leaves a listed, resumable
            # model instead of silently losing the run.
            model_metadata.save_if_usable(resolved_model_id, voice_profile.name, config, engine_name)

    def start_training_in_background(
        self, voice_profile: VoiceProfile, config: TrainingConfig, engine_name: str = DEFAULT_CONVERSION_ENGINE
    ) -> str:
        model_id = model_metadata.new_model_id(voice_profile.name)
        return self._start_in_background(voice_profile, config, engine_name, model_id, resume=False)

    def resume_training_in_background(
        self,
        voice_profile: VoiceProfile,
        model_id: str,
        config: TrainingConfig,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
    ) -> str:
        return self._start_in_background(voice_profile, config, engine_name, model_id, resume=True)

    def _start_in_background(
        self,
        voice_profile: VoiceProfile,
        config: TrainingConfig,
        engine_name: str,
        model_id: str,
        resume: bool,
    ) -> str:
        """Run training in a separate thread so the UI can poll progress
        while it runs. Raises TrainingInProgressError instead of starting a
        second run if one is already active - running two at once contends
        for the same GPU and can make the whole machine unresponsive.
        """
        if not _training_lock.acquire(blocking=False):
            raise TrainingInProgressError(
                "A training run is already in progress. Wait for it to finish first."
            )

        def _run() -> None:
            try:
                self.train(voice_profile, config, engine_name, model_id, resume)
            except Exception as exc:
                failed_job = TrainingJob(
                    voice_name=voice_profile.name,
                    engine_name=engine_name,
                    job_id=model_id,
                    status=TrainingStatus.FAILED,
                    error_message=str(exc),
                    finished_at=datetime.utcnow(),
                )
                progress_bus.publish(failed_job)
                progress_bus.close(failed_job.job_id)
            finally:
                _training_lock.release()

        threading.Thread(target=_run, daemon=True).start()
        return model_id

    def list_models(self, voice_name: str) -> list[TrainedModelInfo]:
        return [model_metadata.from_metadata(data) for data in read_all_json(model_dir_for(voice_name))]

    def get_model(self, voice_name: str, model_id: str) -> TrainedModelInfo | None:
        data = read_json(model_metadata_path_for(voice_name, model_id))
        return model_metadata.from_metadata(data) if data else None

    def delete_model(self, voice_name: str, model_id: str, engine_name: str) -> None:
        EngineRegistry.get_conversion(engine_name).delete_trained_model(model_id)
        delete_json(model_metadata_path_for(voice_name, model_id))

    def list_available_engines(self) -> list[str]:
        return EngineRegistry.available_conversion()
