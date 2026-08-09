from __future__ import annotations

from core.domain.errors import EngineNotFoundError
from core.engines.voice_conversion_engine import VoiceConversionEngine
from core.engines.voice_synthesis_engine import VoiceSynthesisEngine


class EngineRegistry:
    """Central lookup from engine name to concrete implementation.

    The UI lists available engines by consulting this registry - it never
    imports FishSpeechEngine or RvcEngine directly. Adding a new engine is:
    implement the interface, register one line, done.
    """

    _synthesis: dict[str, VoiceSynthesisEngine] = {}
    _conversion: dict[str, VoiceConversionEngine] = {}

    @classmethod
    def register_synthesis(cls, engine: VoiceSynthesisEngine) -> None:
        cls._synthesis[engine.engine_name] = engine

    @classmethod
    def register_conversion(cls, engine: VoiceConversionEngine) -> None:
        cls._conversion[engine.engine_name] = engine

    @classmethod
    def available_synthesis(cls) -> list[str]:
        return list(cls._synthesis.keys())

    @classmethod
    def available_conversion(cls) -> list[str]:
        return list(cls._conversion.keys())

    @classmethod
    def get_synthesis(cls, name: str) -> VoiceSynthesisEngine:
        try:
            return cls._synthesis[name]
        except KeyError as exc:
            raise EngineNotFoundError(f"Synthesis engine not registered: {name}") from exc

    @classmethod
    def get_conversion(cls, name: str) -> VoiceConversionEngine:
        try:
            return cls._conversion[name]
        except KeyError as exc:
            raise EngineNotFoundError(f"Conversion engine not registered: {name}") from exc


def bootstrap_default_engines() -> None:
    """Register the built-in engines. Called once at application startup."""
    from core.engines.fish_speech_engine import FishSpeechEngine
    from core.engines.rvc_engine import RvcEngine

    EngineRegistry.register_synthesis(FishSpeechEngine())
    EngineRegistry.register_conversion(RvcEngine())
