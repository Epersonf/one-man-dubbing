from __future__ import annotations

import threading
from datetime import datetime
from uuid import uuid4

from config import DEFAULT_CONVERSION_ENGINE
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingJob, TrainingStatus
from core.domain.trained_model import TrainedModelInfo
from core.domain.voice_profile import VoiceProfile
from core.registry.engine_registry import EngineRegistry
from infra.audio_io import load_audio_asset
from infra.filesystem_paths import model_dir_for, model_metadata_path_for
from infra.job_progress_bus import progress_bus
from infra.metadata_store import delete_json, read_all_json, read_json, write_json


class TrainingService:
    """Orchestrates Step 2: training a voice conversion model from the
    reference audio produced in Step 1. A voice can have several trained
    models (e.g. re-trained with different settings); each is identified
    by its own model_id.
    """

    def train(
        self,
        voice_profile: VoiceProfile,
        config: TrainingConfig,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
        model_id: str | None = None,
    ) -> TrainingJob:
        engine = EngineRegistry.get_conversion(engine_name)
        reference_asset = load_audio_asset(voice_profile.reference_audio_path)
        job = engine.train(reference_asset, config, model_id or new_model_id(voice_profile.name))
        self._save_model_metadata(job, config, engine_name)
        return job

    def start_training_in_background(
        self,
        voice_profile: VoiceProfile,
        config: TrainingConfig,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
    ) -> str:
        """Run training in a separate thread so the UI can poll progress
        while it runs. Returns the model_id immediately; any failure
        (including ones before the engine gets a chance to run, like a
        missing reference file) is published as a FAILED job instead of
        leaving the progress stream hanging with nothing to report.
        """
        model_id = new_model_id(voice_profile.name)

        def _run() -> None:
            try:
                self.train(voice_profile, config, engine_name, model_id)
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

        threading.Thread(target=_run, daemon=True).start()
        return model_id

    def list_models(self, voice_name: str) -> list[TrainedModelInfo]:
        return [_model_info_from_metadata(data) for data in read_all_json(model_dir_for(voice_name))]

    def get_model(self, voice_name: str, model_id: str) -> TrainedModelInfo | None:
        data = read_json(model_metadata_path_for(voice_name, model_id))
        return _model_info_from_metadata(data) if data else None

    def delete_model(self, voice_name: str, model_id: str, engine_name: str) -> None:
        EngineRegistry.get_conversion(engine_name).delete_trained_model(model_id)
        delete_json(model_metadata_path_for(voice_name, model_id))

    def list_available_engines(self) -> list[str]:
        return EngineRegistry.available_conversion()

    def _save_model_metadata(
        self, job: TrainingJob, config: TrainingConfig, engine_name: str
    ) -> None:
        write_json(
            model_metadata_path_for(job.voice_name, job.job_id),
            {
                "model_id": job.job_id,
                "voice_name": job.voice_name,
                "engine_name": engine_name,
                "created_at": (job.finished_at or datetime.utcnow()).isoformat(),
                "epochs": config.epochs,
                "has_similarity_index": config.use_similarity_index,
            },
        )


def new_model_id(voice_name: str) -> str:
    return f"{voice_name}__{uuid4().hex[:8]}"


def _model_info_from_metadata(data: dict) -> TrainedModelInfo:
    return TrainedModelInfo(
        model_id=data["model_id"],
        voice_name=data["voice_name"],
        engine_name=data["engine_name"],
        created_at=datetime.fromisoformat(data["created_at"]),
        epochs=data["epochs"],
        has_similarity_index=data["has_similarity_index"],
    )
