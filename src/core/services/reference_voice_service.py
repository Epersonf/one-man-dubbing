from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from config import DEFAULT_SYNTHESIS_ENGINE
from core.domain.voice_profile import SynthesisParameters, VoiceProfile, VoiceSourceRoute
from core.registry.engine_registry import EngineRegistry
from infra.audio_io import save_uploaded_audio
from infra.filesystem_paths import REFERENCES_DIR, reference_path_for, voice_metadata_path_for
from infra.metadata_store import delete_json, read_all_json, read_json, write_json


class ReferenceVoiceService:
    """Orchestrates Step 1: producing a single reference audio artifact,
    regardless of whether it came from an upload (Route A) or synthesis
    (Route B). Step 2 only ever sees the resulting VoiceProfile.
    """

    def create_from_upload(self, voice_name: str, raw_audio_bytes: bytes) -> VoiceProfile:
        destination = reference_path_for(voice_name)
        asset = save_uploaded_audio(raw_audio_bytes, destination)
        profile = VoiceProfile(
            name=voice_name,
            reference_audio_path=asset.path,
            source_route=VoiceSourceRoute.UPLOAD,
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

        destination = reference_path_for(voice_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(synthesized_asset.path, destination)

        profile = VoiceProfile(
            name=voice_name,
            reference_audio_path=destination,
            source_route=VoiceSourceRoute.SYNTHESIS,
            synthesis_parameters=parameters,
        )
        self._save_metadata(profile)
        return profile

    def list_voices(self) -> list[VoiceProfile]:
        return [_voice_profile_from_metadata(data) for data in read_all_json(REFERENCES_DIR)]

    def get_voice(self, voice_name: str) -> VoiceProfile | None:
        data = read_json(voice_metadata_path_for(voice_name))
        return _voice_profile_from_metadata(data) if data else None

    def delete_voice(self, voice_name: str) -> None:
        """Remove the voice's reference audio and metadata.

        Does not touch trained models - callers that also want those gone
        should delete them (via TrainingService) before calling this, since
        this service doesn't know about engines/training.
        """
        reference_path_for(voice_name).unlink(missing_ok=True)
        delete_json(voice_metadata_path_for(voice_name))

    def list_available_synthesis_engines(self) -> list[str]:
        return EngineRegistry.available_synthesis()

    def _save_metadata(self, profile: VoiceProfile) -> None:
        write_json(
            voice_metadata_path_for(profile.name),
            {
                "name": profile.name,
                "reference_audio_path": str(profile.reference_audio_path),
                "source_route": profile.source_route.value,
                "created_at": profile.created_at.isoformat(),
            },
        )


def _voice_profile_from_metadata(data: dict) -> VoiceProfile:
    return VoiceProfile(
        name=data["name"],
        reference_audio_path=Path(data["reference_audio_path"]),
        source_route=VoiceSourceRoute(data["source_route"]),
        created_at=datetime.fromisoformat(data["created_at"]),
    )
