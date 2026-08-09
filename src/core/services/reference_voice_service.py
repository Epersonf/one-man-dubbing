from __future__ import annotations

import shutil

from config import DEFAULT_SYNTHESIS_ENGINE
from core.domain.voice_profile import SynthesisParameters, VoiceProfile, VoiceSourceRoute
from core.registry.engine_registry import EngineRegistry
from infra.audio_io import save_uploaded_audio
from infra.filesystem_paths import reference_path_for


class ReferenceVoiceService:
    """Orchestrates Step 1: producing a single reference audio artifact,
    regardless of whether it came from an upload (Route A) or synthesis
    (Route B). Step 2 only ever sees the resulting VoiceProfile.
    """

    def create_from_upload(self, voice_name: str, raw_audio_bytes: bytes) -> VoiceProfile:
        destination = reference_path_for(voice_name)
        asset = save_uploaded_audio(raw_audio_bytes, destination)
        return VoiceProfile(
            name=voice_name,
            reference_audio_path=asset.path,
            source_route=VoiceSourceRoute.UPLOAD,
        )

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

        return VoiceProfile(
            name=voice_name,
            reference_audio_path=destination,
            source_route=VoiceSourceRoute.SYNTHESIS,
            synthesis_parameters=parameters,
        )

    def list_available_synthesis_engines(self) -> list[str]:
        return EngineRegistry.available_synthesis()
