from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from config import DEFAULT_SYNTHESIS_ENGINE
from core.domain.voice_profile import SynthesisParameters, VoiceProfile, VoiceSourceRoute
from core.registry.engine_registry import EngineRegistry
from core.services import voice_backfill
from infra.audio_io import SUPPORTED_EXTENSIONS, save_uploaded_audio
from infra.filesystem_paths import REFERENCES_DIR, reference_dir_for, voice_metadata_path_for
from infra.metadata_store import delete_json, read_all_json, read_json, write_json


class ReferenceVoiceService:
    """Orchestrates Step 1: producing the reference audio clip(s) a voice
    is trained from, regardless of whether they came from an upload
    (Route A, one or more clips) or synthesis (Route B, always one clip).
    Step 2 only ever sees the resulting VoiceProfile + its clip list.
    """

    def create_from_upload(self, voice_name: str, files: list[tuple[str, bytes]]) -> VoiceProfile:
        clip_dir = reference_dir_for(voice_name)
        for index, (filename, raw_bytes) in enumerate(files):
            suffix = Path(filename).suffix or ".wav"
            save_uploaded_audio(raw_bytes, clip_dir / f"clip_{index:02d}{suffix}")

        profile = VoiceProfile(
            name=voice_name, reference_dir=clip_dir, source_route=VoiceSourceRoute.UPLOAD
        )
        self._save_metadata(profile)
        return profile

    def create_from_synthesis(
        self,
        voice_name: str,
        text_sample: str,
        parameters: SynthesisParameters,
        engine_name: str = DEFAULT_SYNTHESIS_ENGINE,
    ) -> VoiceProfile:
        engine = EngineRegistry.get_synthesis(engine_name)
        synthesized_asset = engine.synthesize(text_sample, parameters)

        clip_dir = reference_dir_for(voice_name)
        clip_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(synthesized_asset.path, clip_dir / f"clip_00{synthesized_asset.path.suffix}")

        profile = VoiceProfile(
            name=voice_name,
            reference_dir=clip_dir,
            source_route=VoiceSourceRoute.SYNTHESIS,
            synthesis_parameters=parameters,
        )
        self._save_metadata(profile)
        return profile

    def list_reference_clips(self, voice_name: str) -> list[Path]:
        clip_dir = reference_dir_for(voice_name)
        if not clip_dir.is_dir():
            return []
        return sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)

    def list_voices(self) -> list[VoiceProfile]:
        voice_backfill.backfill_missing_metadata(self._save_metadata)
        return [_voice_profile_from_metadata(data) for data in read_all_json(REFERENCES_DIR)]

    def get_voice(self, voice_name: str) -> VoiceProfile | None:
        voice_backfill.migrate_legacy_single_file(voice_name)
        data = read_json(voice_metadata_path_for(voice_name))
        if data is None and reference_dir_for(voice_name).is_dir():
            voice_backfill.backfill_metadata_for(voice_name, self._save_metadata)
            data = read_json(voice_metadata_path_for(voice_name))
        return _voice_profile_from_metadata(data) if data else None

    def delete_voice(self, voice_name: str) -> None:
        """Remove the voice's reference clips and metadata.

        Does not touch trained models - callers that also want those gone
        should delete them (via TrainingService) before calling this, since
        this service doesn't know about engines/training.
        """
        shutil.rmtree(reference_dir_for(voice_name), ignore_errors=True)
        delete_json(voice_metadata_path_for(voice_name))

    def list_available_synthesis_engines(self) -> list[str]:
        return EngineRegistry.available_synthesis()

    def _save_metadata(self, profile: VoiceProfile) -> None:
        write_json(
            voice_metadata_path_for(profile.name),
            {
                "name": profile.name,
                "source_route": profile.source_route.value,
                "created_at": profile.created_at.isoformat(),
            },
        )


def _voice_profile_from_metadata(data: dict) -> VoiceProfile:
    return VoiceProfile(
        name=data["name"],
        reference_dir=reference_dir_for(data["name"]),
        source_route=VoiceSourceRoute(data["source_route"]),
        created_at=datetime.fromisoformat(data["created_at"]),
    )
