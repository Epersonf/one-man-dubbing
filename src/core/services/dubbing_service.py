from __future__ import annotations

from config import DEFAULT_CONVERSION_ENGINE, DEFAULT_OUTPUT_FORMAT
from core.domain.audio_asset import AudioAsset
from core.registry.engine_registry import EngineRegistry
from infra.audio_io import convert_to_format, save_uploaded_audio
from infra.filesystem_paths import model_dir_for, output_path_for


class DubbingService:
    """Orchestrates Step 3: converting the actor's own-voice recording into
    the trained target voice, producing the final output file.
    """

    def dub(
        self,
        voice_name: str,
        actor_audio_bytes: bytes,
        job_id: str,
        engine_name: str = DEFAULT_CONVERSION_ENGINE,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> AudioAsset:
        engine = EngineRegistry.get_conversion(engine_name)

        staged_path = model_dir_for(voice_name) / f"actor_input_{job_id}.wav"
        source_asset = save_uploaded_audio(actor_audio_bytes, staged_path)

        trained_model_path = model_dir_for(voice_name) / "model.pth"
        converted_asset = engine.convert(source_asset, trained_model_path)

        final_path = output_path_for(voice_name, job_id, output_format)
        return convert_to_format(converted_asset, final_path, output_format)
