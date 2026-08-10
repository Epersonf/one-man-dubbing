from __future__ import annotations

import threading
from uuid import uuid4

from config import DEFAULT_CONVERSION_ENGINE, DEFAULT_OUTPUT_FORMAT
from core.domain.audio_asset import AudioAsset
from core.domain.dubbing_job import DubbingJob, DubbingStatus
from core.domain.errors import DubbingInProgressError
from core.registry.engine_registry import EngineRegistry
from infra.audio_io import convert_to_format, save_uploaded_audio
from infra.filesystem_paths import model_dir_for, output_path_for
from infra.job_progress_bus import progress_bus

# Module-level: conversion is GPU-exclusive, same reasoning as
# TrainingService's _training_lock.
_dubbing_lock = threading.Lock()


class DubbingService:
    """Orchestrates Step 3: converting the actor's own-voice recording into
    the trained target voice, producing the final output file.
    """

    def dub(
        self,
        voice_name: str,
        model_id: str,
        actor_audio_bytes: bytes,
        job_id: str,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> AudioAsset:
        engine = EngineRegistry.get_conversion(engine_name)

        staged_path = model_dir_for(voice_name) / f"actor_input_{job_id}.wav"
        source_asset = save_uploaded_audio(actor_audio_bytes, staged_path)

        trained_model_path = engine.trained_model_path_for(model_id)
        converted_asset = engine.convert(source_asset, trained_model_path, job_id)

        final_path = output_path_for(voice_name, job_id, output_format)
        result_asset = convert_to_format(converted_asset, final_path, output_format)
        self._finalize_job(job_id, result_asset)
        return result_asset

    def start_dub_in_background(
        self,
        voice_name: str,
        model_id: str,
        actor_audio_bytes: bytes,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> str:
        """Run dubbing in a separate thread so the UI can poll progress
        while it runs, mirroring TrainingService.start_training_in_background.
        """
        if not _dubbing_lock.acquire(blocking=False):
            raise DubbingInProgressError(
                "A dubbing run is already in progress. Wait for it to finish first."
            )
        job_id = uuid4().hex

        def _run() -> None:
            try:
                self.dub(voice_name, model_id, actor_audio_bytes, job_id, engine_name, output_format)
            except Exception as exc:
                failed_job = DubbingJob(
                    job_id=job_id,
                    model_id=model_id,
                    status=DubbingStatus.FAILED,
                    error_message=str(exc),
                )
                progress_bus.publish(failed_job)
                progress_bus.close(job_id)
            finally:
                _dubbing_lock.release()

        threading.Thread(target=_run, daemon=True).start()
        return job_id

    def _finalize_job(self, job_id: str, result_asset: AudioAsset) -> None:
        job = progress_bus.latest_for_job(job_id)
        if job is not None:
            job.status = DubbingStatus.COMPLETED
            job.output_path = result_asset.path
            progress_bus.publish(job)
        progress_bus.close(job_id)
