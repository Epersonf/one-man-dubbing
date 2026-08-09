from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class VoiceSourceRoute(str, Enum):
    """How the reference voice audio was obtained (Step 1)."""

    UPLOAD = "upload"
    SYNTHESIS = "synthesis"


@dataclass(frozen=True)
class SynthesisParameters:
    """Controls exposed to the UI for zero-shot voice synthesis (Route B)."""

    pitch_base: float = 0.0
    timbre_embedding: str | None = None
    speed: float = 1.0
    accent: str | None = None


@dataclass(frozen=True)
class VoiceProfile:
    """Metadata describing a target voice, regardless of how it was created."""

    name: str
    reference_audio_path: Path
    source_route: VoiceSourceRoute
    created_at: datetime = field(default_factory=datetime.utcnow)
    synthesis_parameters: SynthesisParameters | None = None
