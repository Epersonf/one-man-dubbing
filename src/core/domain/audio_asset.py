from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioAsset:
    """Immutable reference to an audio file on disk plus its known properties."""

    path: Path
    duration_seconds: float
    sample_rate: int

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def exists(self) -> bool:
        return self.path.is_file()
